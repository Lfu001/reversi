import argparse
import random

import numpy as np
import torch
from mcts import MCTS
from reversi import ReversiEnvironment
from tqdm.rich import tqdm
from transformers import AutoConfig, AutoModel

from mcts_grpo.model.configuration_urm import URMConfig
from mcts_grpo.model.modeling_urm import URMModel
from mcts_grpo.settings import Settings
from mcts_grpo.training import GameSetup


class Agent:
    """Base class for evaluation agents."""

    def act(
        self,
        valid_moves: list[np.ndarray],
        current_states: np.ndarray | None = None,
        env: ReversiEnvironment | None = None,
    ) -> list[np.ndarray]:
        """
        Selects a move for each game state.

        Args:
            valid_moves: A list of length batch_size, where each element is a boolean mask
                         of shape (64,) indicating valid moves.
            current_states: Current states (batch, 4, 8, 8). Used for lookahead agents.
            env: ReversiEnvironment instance. Used for lookahead agents.

        Returns:
            A list of length batch_size, where each element is a one-hot encoded policy
            of shape (64,).
        """
        raise NotImplementedError


class RandomAgent(Agent):
    """A simple agent that plays random legal moves."""

    def act(
        self,
        valid_moves: list[np.ndarray],
        current_states: np.ndarray | None = None,
        env: ReversiEnvironment | None = None,
    ) -> list[np.ndarray]:
        actions = []
        for valid_mask in valid_moves:
            legal_indices = np.where(valid_mask)[0]
            action = np.zeros(64, dtype=np.float32)
            if len(legal_indices) > 0:
                chosen_idx = random.choice(legal_indices)
                action[chosen_idx] = 1.0
            actions.append(action)
        return actions


class GreedyAgent(Agent):
    """Agent that chooses the move maximizing the score difference."""

    def act(
        self,
        valid_moves: list[np.ndarray],
        current_states: np.ndarray | None = None,
        env: ReversiEnvironment | None = None,
    ) -> list[np.ndarray]:
        if current_states is None or env is None:
            raise ValueError("GreedyAgent requires current_states and env.")

        actions = []
        # Process each game in the batch
        for i, valid_mask in enumerate(valid_moves):
            legal_indices = np.where(valid_mask)[0]
            action = np.zeros(64, dtype=np.float32)

            if len(legal_indices) > 0:
                best_score = -float("inf")
                best_idx = legal_indices[0]

                # Get single game state (4, 8, 8)
                state = current_states[i]

                # Turn information to know which color is maximizing
                # MCTS uses 1.0 for Dark, -1.0 for Light in Channel 2.
                # Here we just want to maximize (My Disks - Opp Disks).
                # ReversiEnvironment.get_next_state returns state.
                # Channel 0: Dark, Channel 1: Light.
                # If current player is Dark, we max (Dark - Light).
                # If Light, we max (Light - Dark).
                # Wait, get_next_state might normalize the state for the NEXT player?
                # Let's check get_next_state.
                # It returns Absolute Board State if I recall correctly?
                # No, get_state returns Absolute Board State (Channel 0 Dark, Channel 1 Light).
                # But ML models usually expect "My Color" in Ch 0.
                # However, ReversiEnvironment implementation showed:
                # Ch 0: Dark Plane, Ch 1: Light Plane. (Absolute)
                # So we need to know OUR color.

                turn_val = state[
                    2, 0, 0
                ]  # 1.0 if Dark, -1.0 if Light (Wait, check implementation)
                # In get_state_vec:
                # let turn_val = if turn == DiskColor::Dark { 1.0 } else { -1.0 };
                # state_vec[idx + 128] = turn_val;
                # So Yes.

                is_dark = turn_val > 0

                for idx in legal_indices:
                    # Lookahead
                    # Note: get_next_state is static.
                    next_state = ReversiEnvironment.get_next_state(state, idx)
                    # next_state is (4, 8, 8)

                    dark_count = np.sum(next_state[0])
                    light_count = np.sum(next_state[1])

                    if is_dark:
                        score = dark_count - light_count
                    else:
                        score = light_count - dark_count

                    if score > best_score:
                        best_score = score
                        best_idx = idx

                action[best_idx] = 1.0

            actions.append(action)
        return actions


class MobilityMinimizationAgent(Agent):
    """Agent that chooses the move minimizing the opponent's mobility (legal moves)."""

    def act(
        self,
        valid_moves: list[np.ndarray],
        current_states: np.ndarray | None = None,
        env: ReversiEnvironment | None = None,
    ) -> list[np.ndarray]:
        if current_states is None or env is None:
            raise ValueError(
                "MobilityMinimizationAgent requires current_states and env."
            )

        actions = []
        for i, valid_mask in enumerate(valid_moves):
            legal_indices = np.where(valid_mask)[0]
            action = np.zeros(64, dtype=np.float32)

            if len(legal_indices) > 0:
                best_mobility = float("inf")
                best_idx = legal_indices[0]

                state = current_states[i]

                for idx in legal_indices:
                    next_state = ReversiEnvironment.get_next_state(state, idx)
                    # Next state Channel 3 is Legal Moves for the NEXT player (Opponent).
                    # We want to MINIMIZE this sum.

                    opponent_mobility = np.sum(next_state[3])

                    if opponent_mobility < best_mobility:
                        best_mobility = opponent_mobility
                        best_idx = idx

                action[best_idx] = 1.0
            actions.append(action)
        return actions


def evaluate(
    num_games: int = 100,
    batch_size: int = 64,
    mcts_sims: int = 50,
    checkpoint: str | None = None,
    opponent_type: str = "random",
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    env: ReversiEnvironment | None = None,
) -> dict[str, float]:
    """
    Evaluates the MCTS model against a Baseline Agent.

    Args:
        num_games: Total number of games to play.
        batch_size: Number of parallel games.
        mcts_sims: Number of MCTS simulations per move.
        checkpoint: Path to checkpoint directory to load.
        opponent_type: Type of opponent ('random', 'greedy', 'mobility').
        model: Pre-loaded model instance (optional).
        device: Device to run evaluation on (optional).
        env: Pre-loaded environment (optional).

    Returns:
        Dictionary containing win rate and other metrics.
    """
    print(
        f"Starting evaluation: MCTS vs {opponent_type.capitalize()} ({num_games} games)"
    )

    # Initialize Settings
    settings = Settings()
    # Override settings for evaluation
    settings.training.batch_size = batch_size
    settings.mcts.num_simulations = mcts_sims

    if model is None:
        setup = GameSetup(settings)
        accelerator, device, model, _, _, env, _ = setup.initialize()

        # Load checkpoint if provided or available
        if checkpoint:
            print(f"Loading checkpoint from {checkpoint}")
            accelerator.load_state(checkpoint)
        else:
            # Try to find latest checkpoint
            from pathlib import Path

            # Helper to find latest checkpoint
            base_dir = Path("logs/checkpoints")
            if base_dir.exists():
                checkpoints = sorted(
                    [
                        d
                        for d in base_dir.iterdir()
                        if d.is_dir() and d.name.startswith("checkpoint_")
                    ]
                )
                if checkpoints:
                    latest = checkpoints[-1]
                    print(f"Found latest checkpoint: {latest}")
                    try:
                        accelerator.load_state(str(latest))
                        print("Loaded latest checkpoint.")
                    except Exception as e:
                        print(f"Failed to load checkpoint: {e}")
                        print("Running with random weights.")
                else:
                    print(
                        "No checkpoints found in logs/checkpoints. Running with random weights."
                    )
            else:
                print("logs/checkpoints not found. Running with random weights.")
    else:
        # Use provided model
        if device is None:
            device = next(model.parameters()).device

        if env is None:
            env = ReversiEnvironment(batch_size=batch_size)

    model.eval()

    # Initialize Agents
    mcts_agent = MCTS(
        max_inference_batch_size=settings.mcts.max_inference_batch_size,
        states_per_inference=settings.mcts.states_per_inference,
    )

    if opponent_type == "random":
        opponent_agent = RandomAgent()
    elif opponent_type == "greedy":
        opponent_agent = GreedyAgent()
    elif opponent_type == "mobility":
        opponent_agent = MobilityMinimizationAgent()
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    results = {"wins": 0, "losses": 0, "draws": 0}
    games_completed = 0
    pbar = tqdm(total=num_games, desc="Evaluated Games")

    # We will alternate colors every batch to ensure fairness
    # Batch 0: MCTS=Black, Batch 1: MCTS=White, etc.
    batch_idx = 0

    while games_completed < num_games:
        # Determine MCTS color for this batch
        # Alternating: Even batches -> MCTS is Black (Player 1 in my logic, but DiskColor::Dark is usually Black/First)
        # ReversiEnvironment Channel 2: 1.0 for Dark (Black), -1.0 for Light (White)
        mcts_is_black = batch_idx % 2 == 0
        mcts_color_val = 1.0 if mcts_is_black else -1.0

        desc = "MCTS=Black" if mcts_is_black else "MCTS=White"
        pbar.set_description(f"Evaluated Games ({desc})")

        current_states = env.reset()
        dones = np.zeros(batch_size, dtype=bool)

        while not np.all(dones):
            # Check turns
            # State shape: (batch, 4, 8, 8)
            # Channel 2 is turn plane.
            # We can take the mean or just one value since it's a plane
            turns = current_states[:, 2, 0, 0]  # (batch,)

            # Prepare policies
            final_actions = np.zeros((batch_size, 8, 8), dtype=np.float32)

            # Helper to get mask of active games that need action from a specific agent
            # We only care about games that are NOT done.
            active_mask = ~dones

            # Identify which games are MCTS turn
            # MCTS moves if (turn == mcts_color_val)
            # Use a small epsilon for float comparison safety
            is_mcts_turn = (np.abs(turns - mcts_color_val) < 0.1) & active_mask
            is_opponent_turn = (~is_mcts_turn) & active_mask

            # MCTS Step
            if np.any(is_mcts_turn):
                # Extract states for MCTS
                # mcts.run_simulations expects full batch usually, or we can filter?
                # The MCTS wrapper usually takes the whole batch or a list of states.
                # Looking at `self_play.py`, it passes `current_states` (full batch).
                # `mcts.run_simulations` returns pi, q for ALL states provided.
                # Efficiency optimization: We could theoretically only run MCTS on relevant states,
                # but the current MCTS implementation might expect fixed batch size or mapping.
                # However, MCTS.run_simulations takes `states` and returns results.
                # If we pass a subset, we get a subset back.

                mcts_indices = np.where(is_mcts_turn)[0]
                mcts_states_subset = current_states[mcts_indices]

                pi_subset, _ = mcts_agent.run_simulations(
                    model=model,
                    states=mcts_states_subset,
                    device=device,
                    num_simulations=settings.mcts.num_simulations,
                    dirichlet_epsilon=0.0,  # Disable noise for evaluation
                    dirichlet_alpha=settings.mcts.dirichlet_alpha,
                    c_puct=settings.mcts.c_puct,
                )

                # MCTS returns (N, 64) -> reshape to (N, 8, 8)
                pi_subset_reshaped = pi_subset.reshape(-1, 8, 8)

                # Assign to final actions
                # We need to make sure we map back to correct indices
                for i, idx in enumerate(mcts_indices):
                    final_actions[idx] = pi_subset_reshaped[i]

            # Opponent Step
            if np.any(is_opponent_turn):
                opp_indices = np.where(is_opponent_turn)[0]
                opp_states_subset = current_states[opp_indices]

                # We need valid moves. Format: Channel 3 of state
                valid_moves_subset = opp_states_subset[:, 3, :, :]  # (N, 8, 8)
                valid_moves_flat = valid_moves_subset.reshape(-1, 64) > 0.5
                valid_moves_list = list(valid_moves_flat)

                # Pass necessary info for Lookahead agents
                opp_actions = opponent_agent.act(
                    valid_moves=valid_moves_list,
                    current_states=opp_states_subset,
                    env=env,
                )

                for i, idx in enumerate(opp_indices):
                    final_actions[idx] = opp_actions[i].reshape(8, 8)

            # Perform Step
            # Use deterministic step for evaluation
            next_states, step_dones = env.step_batch(final_actions, deterministic=True)

            # Handle Dones
            new_dones_indices = np.where(step_dones & ~dones)[0]
            for idx in new_dones_indices:
                # Calculate winner
                black_count = np.sum(next_states[idx][0])
                white_count = np.sum(next_states[idx][1])

                winner = 0  # 0=Draw, 1=Black, -1=White
                if black_count > white_count:
                    winner = 1
                elif white_count > black_count:
                    winner = -1

                # Did MCTS win?
                # MCTS is Black (1) if mcts_is_black=True
                # MCTS is White (-1) if mcts_is_black=False

                mcts_val = 1 if mcts_is_black else -1

                if winner == mcts_val:
                    results["wins"] += 1
                elif winner == 0:
                    results["draws"] += 1
                else:
                    results["losses"] += 1

                games_completed += 1
                pbar.update(1)

            current_states = next_states
            dones = step_dones

            if games_completed >= num_games:
                break

        batch_idx += 1

    pbar.close()

    total = results["wins"] + results["losses"] + results["draws"]
    win_rate = results["wins"] / total if total > 0 else 0.0

    print("\nEvaluation Results:")
    print(f"Total Games: {total}")
    print(f"Wins: {results['wins']}")
    print(f"Losses: {results['losses']}")
    print(f"Draws: {results['draws']}")
    print(f"Win Rate: {win_rate:.2%}")

    results["total_games"] = total
    results["win_rate"] = win_rate
    return results


def _calculate_score_diff(state: np.ndarray) -> int:
    """Calculates Black - White score difference."""
    black = np.sum(state[0])
    white = np.sum(state[1])
    return int(black - white)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "greedy", "mobility"],
        help="Opponent type",
    )
    args = parser.parse_args()

    AutoConfig.register("urm", URMConfig)
    AutoModel.register(URMConfig, URMModel)
    # model = AutoModel.from_pretrained("reversi_zero_model_final").to("mps")

    evaluate(
        args.num_games,
        args.batch_size,
        checkpoint=args.checkpoint,
        opponent_type=args.opponent,
        mcts_sims=800,
        # model=model,
    )
