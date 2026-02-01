"""Evaluation script for MCTS model against various opponents.

This script evaluates the performance of an MCTS-based model by playing
games against different baseline agents (random, greedy, mobility).
"""

import argparse
import random

import numpy as np
import torch
from accelerate.utils import set_seed
from mcts import MCTS
from reversi import ReversiEnvironment
from tqdm.rich import tqdm
from transformers import AutoConfig, AutoModel

from mcts_grpo.model.configuration_urm import URMConfig
from mcts_grpo.model.modeling_urm import URMModel
from mcts_grpo.settings import Settings
from mcts_grpo.training import GameSetup


def get_initial_state() -> np.ndarray:
    """Returns the initial Reversi game state.

    Returns:
        A (4, 8, 8) array representing the initial board state.
        - Channel 0: Dark disks (1 where disk exists)
        - Channel 1: Light disks (1 where disk exists)
        - Channel 2: Turn indicator (1 for Dark, -1 for Light)
        - Channel 3: Legal moves for current player
    """
    state = np.zeros((4, 8, 8), dtype=np.int8)

    # Initial disk positions (center 2x2)
    # Dark at d5, e4; Light at d4, e5
    state[0, 3, 4] = 1  # Dark at e4 (row 3, col 4)
    state[0, 4, 3] = 1  # Dark at d5 (row 4, col 3)
    state[1, 3, 3] = 1  # Light at d4 (row 3, col 3)
    state[1, 4, 4] = 1  # Light at e5 (row 4, col 4)

    # Dark moves first
    state[2, :, :] = 1

    # Initial legal moves for Dark: c4, d3, e6, f5
    state[3, 2, 3] = 1  # d3
    state[3, 3, 2] = 1  # c4
    state[3, 4, 5] = 1  # f5
    state[3, 5, 4] = 1  # e6

    return state


def is_game_over(state: np.ndarray) -> bool:
    """Check if the game is over (no legal moves for current player)."""
    return state[3].sum() == 0


def get_winner(state: np.ndarray) -> int:
    """Determine the winner of a finished game.

    Returns:
        1 for Dark win, -1 for Light win, 0 for draw.
    """
    dark_count = state[0].sum()
    light_count = state[1].sum()

    if dark_count > light_count:
        return 1
    elif light_count > dark_count:
        return -1
    else:
        return 0


def select_random_move(state: np.ndarray) -> int:
    """Select a random legal move."""
    legal_moves = state[3].flatten()
    legal_indices = np.where(legal_moves > 0)[0]
    return int(random.choice(legal_indices))


def select_greedy_move(state: np.ndarray) -> int:
    """Select the move that maximizes disk count difference for current player."""
    legal_moves = state[3].flatten()
    legal_indices = np.where(legal_moves > 0)[0]

    turn_val = state[2, 0, 0]
    is_dark = turn_val > 0

    best_score = -float("inf")
    best_idx = legal_indices[0]

    for idx in legal_indices:
        next_state = ReversiEnvironment.get_next_state(state, idx)
        dark_count = next_state[0].sum()
        light_count = next_state[1].sum()

        if is_dark:
            score = dark_count - light_count
        else:
            score = light_count - dark_count

        if score > best_score:
            best_score = score
            best_idx = idx

    return int(best_idx)


def select_mobility_move(state: np.ndarray) -> int:
    """Select the move that minimizes opponent's mobility (legal moves)."""
    legal_moves = state[3].flatten()
    legal_indices = np.where(legal_moves > 0)[0]

    best_mobility = float("inf")
    best_idx = legal_indices[0]

    for idx in legal_indices:
        next_state = ReversiEnvironment.get_next_state(state, idx)
        # Channel 3 of next_state contains legal moves for the NEXT player (opponent)
        opponent_mobility = next_state[3].sum()

        if opponent_mobility < best_mobility:
            best_mobility = opponent_mobility
            best_idx = idx

    return int(best_idx)


def play_single_game(
    mcts_agent: MCTS,
    model: torch.nn.Module,
    device: torch.device,
    settings: Settings,
    mcts_is_dark: bool,
    opponent_type: str,
    initial_state: np.ndarray | None = None,
) -> int:
    """Play a single game and return the result for MCTS.

    Args:
        mcts_agent: MCTS instance.
        model: The neural network model.
        device: Device to run inference on.
        settings: MCTS settings.
        mcts_is_dark: True if MCTS plays as Dark (first player).
        opponent_type: Type of opponent ('random', 'greedy', 'mobility').
        initial_state: Optional initial state to start from (4, 8, 8).

    Returns:
        1 for MCTS win, 0 for draw, -1 for MCTS loss.
    """
    state = initial_state if initial_state is not None else get_initial_state()
    mcts_color_val = 1 if mcts_is_dark else -1

    while not is_game_over(state):
        turn_val = state[2, 0, 0]
        is_mcts_turn = (turn_val > 0) == mcts_is_dark

        if is_mcts_turn:
            # MCTS move
            # Need to expand state to batch dimension (1, 4, 8, 8)
            state_batch = state[np.newaxis, :].astype(np.float32)

            pi, _, _ = mcts_agent.run_simulations(
                model=model,
                states=state_batch,
                device=device,
                num_simulations=settings.mcts.num_simulations,
                dirichlet_epsilon=0.0,  # Disable noise for evaluation
                dirichlet_alpha=settings.mcts.dirichlet_alpha,
                c_puct=settings.mcts.c_puct,
            )

            # Select the move with highest probability among legal moves
            legal_mask = state[3].flatten() > 0
            pi_masked = np.where(legal_mask, pi[0], -np.inf)
            action_idx = int(np.argmax(pi_masked))
        else:
            # Opponent move
            if opponent_type == "random":
                action_idx = select_random_move(state)
            elif opponent_type == "greedy":
                action_idx = select_greedy_move(state)
            elif opponent_type == "mobility":
                action_idx = select_mobility_move(state)
            else:
                raise ValueError(f"Unknown opponent type: {opponent_type}")

        # Apply the action using stateless API
        state = ReversiEnvironment.get_next_state(state, action_idx)

    # Determine winner
    winner = get_winner(state)

    # Return result from MCTS perspective
    if winner == mcts_color_val:
        return 1  # MCTS win
    elif winner == 0:
        return 0  # Draw
    else:
        return -1  # MCTS loss


def _generate_random_opening(max_random_moves: int = 6) -> np.ndarray:
    """Generate a random opening by playing random moves from initial state.

    Args:
        max_random_moves: Maximum number of random moves to play (1 to this value).

    Returns:
        A game state array of shape (4, 8, 8) after random moves.
    """
    temp_env = ReversiEnvironment(batch_size=1)
    num_moves = random.randint(1, max_random_moves)

    for _ in range(num_moves):
        states, _ = temp_env.step_batch(
            np.ones((8, 8), dtype=float), deterministic=False
        )

    return states[0]


def evaluate(
    num_games: int = 100,
    mcts_sims: int = 50,
    checkpoint: str | None = None,
    opponent_type: str = "random",
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    max_random_moves: int = 6,
    quiet: bool = False,
) -> dict[str, float]:
    """Evaluates the MCTS model against a baseline agent.

    Args:
        num_games: Total number of games to play (must be even).
        mcts_sims: Number of MCTS simulations per move.
        checkpoint: Path to checkpoint directory to load.
        opponent_type: Type of opponent ('random', 'greedy', 'mobility').
        model: Pre-loaded model instance (optional).
        device: Device to run evaluation on (optional).
        max_random_moves: Max random moves for opening randomization (1 to this).
        quiet: If True, suppress print statements.

    Returns:
        Dictionary containing win rate and other metrics.
    """
    assert num_games % 2 == 0, "num_games must be even for fair paired evaluation"

    if not quiet:
        print(
            f"Starting evaluation: MCTS vs {opponent_type.capitalize()} ({num_games} games)"
        )

    # Initialize Settings
    settings = Settings()
    settings.mcts.num_simulations = mcts_sims

    if model is None:
        setup = GameSetup(settings)
        accelerator, device, model, _, _, _, _ = setup.initialize()

        # Load checkpoint if provided or available
        if checkpoint:
            print(f"Loading checkpoint from {checkpoint}")
            accelerator.load_state(checkpoint)
        else:
            # Try to find latest checkpoint
            from pathlib import Path

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

    model.eval()

    # Initialize MCTS agent
    mcts_agent = MCTS(
        max_inference_batch_size=settings.mcts.max_inference_batch_size,
        states_per_inference=settings.mcts.states_per_inference,
    )

    results = {"wins": 0, "losses": 0, "draws": 0}

    # Play paired games: same opening, alternating colors for fairness
    num_pairs = num_games // 2
    for pair_idx in tqdm(range(num_pairs), desc="Evaluating (pairs)", disable=quiet):
        # Generate random opening for this pair
        opening_state = _generate_random_opening(max_random_moves)

        # Game 1: MCTS as Dark
        result1 = play_single_game(
            mcts_agent=mcts_agent,
            model=model,
            device=device,
            settings=settings,
            mcts_is_dark=True,
            opponent_type=opponent_type,
            initial_state=opening_state.copy(),
        )

        # Game 2: MCTS as Light (same opening)
        result2 = play_single_game(
            mcts_agent=mcts_agent,
            model=model,
            device=device,
            settings=settings,
            mcts_is_dark=False,
            opponent_type=opponent_type,
            initial_state=opening_state.copy(),
        )

        for result in [result1, result2]:
            if result == 1:
                results["wins"] += 1
            elif result == 0:
                results["draws"] += 1
            else:
                results["losses"] += 1

    total = results["wins"] + results["losses"] + results["draws"]
    win_rate = results["wins"] / total if total > 0 else 0.0

    if not quiet:
        print("\nEvaluation Results:")
        print(f"Total Games: {total}")
        print(f"Wins: {results['wins']}")
        print(f"Losses: {results['losses']}")
        print(f"Draws: {results['draws']}")
        print(f"Win Rate: {win_rate:.2%}")

    results["total_games"] = total
    results["win_rate"] = win_rate
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=100)
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
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=50,
        help="Number of MCTS simulations per move",
    )
    parser.add_argument(
        "--max-random-moves",
        type=int,
        default=6,
        help="Max random moves for opening randomization (1 to this value)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    AutoConfig.register("urm", URMConfig)
    AutoModel.register(URMConfig, URMModel)

    evaluate(
        args.num_games,
        mcts_sims=args.mcts_sims,
        checkpoint=args.checkpoint,
        opponent_type=args.opponent,
        max_random_moves=args.max_random_moves,
    )
