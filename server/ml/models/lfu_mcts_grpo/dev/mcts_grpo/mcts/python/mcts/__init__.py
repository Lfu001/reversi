"""
MCTS Python Library
Provides high-level Python interface to the Rust MCTS implementation.
"""

from typing import Any, Optional

import numpy as np

from ._core import MCTS as _RustMCTS

__all__ = ["MCTS", "RustMCTS"]


class MCTS:
    """
    Monte Carlo Tree Search implementation.

    This is the main public interface for the MCTS library, providing
    a convenient API for running MCTS simulations with neural network models.
    """

    def __init__(self, max_inference_batch_size: int, states_per_inference: int):
        """
        Initialize MCTS with configuration parameters.

        Args:
            max_inference_batch_size: Maximum batch size for inference worker
            states_per_inference: Number of states to process per inference call
        """
        self.max_inference_batch_size = max_inference_batch_size
        self.states_per_inference = states_per_inference
        self._rust_mcts = _RustMCTS(max_inference_batch_size, states_per_inference)

    def run_simulations(
        self,
        model: Any,
        states: np.ndarray,
        device: Any,
        num_simulations: int,
        dirichlet_epsilon: float,
        dirichlet_alpha: float,
        c_puct: float,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run MCTS simulations for a batch of states.

        Args:
            model: Neural network model for policy and value evaluation
            states: Batch of game states, shape [B, C, H, W]
            device: Device to run the model on (e.g., torch.device)
            num_simulations: Number of MCTS simulations per move
            dirichlet_epsilon: Epsilon for Dirichlet noise
            dirichlet_alpha: Alpha parameter for Dirichlet distribution
            c_puct: Exploration constant for PUCT algorithm
            seed: Random seed for reproducibility (optional)

        Returns:
            pi: Improved policy distributions, shape [B, num_actions]
            q_values: Q-values for each action, shape [B, num_actions]
        """

        # Create inference callback that wraps the model
        def inference_callback(
            batch_states: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            Inference callback for the Rust MCTS.

            Args:
                batch_states: Batch of states from MCTS, shape [N, C, H, W]

            Returns:
                policy: Policy logits, shape [N, num_actions]
                value: State values, shape [N, 1]
            """
            # Import torch here to avoid hard dependency
            try:
                import torch

                is_torch_available = True
            except ImportError:
                is_torch_available = False

            if is_torch_available and device is not None:
                with torch.no_grad():
                    # Convert to torch tensor
                    states_tensor = torch.from_numpy(batch_states).to(device)

                    # Run model inference
                    policy_logits, value = model(states_tensor)

                    # Convert back to numpy if needed
                    if isinstance(policy_logits, torch.Tensor):
                        policy = policy_logits.cpu().numpy()
                        value = value.cpu().numpy()
                    else:
                        policy = policy_logits
                        value = value
            else:
                # Fallback for non-PyTorch models or when device is None
                policy, value = model(batch_states)

            # Ensure correct shapes: policy should be (N, 64), value should be (N, 1)
            batch_size = batch_states.shape[0]
            policy = policy.reshape(batch_size, -1).astype(np.float32)
            value = value.reshape(batch_size, -1).astype(np.float32)

            return policy, value

        # Run MCTS with the Rust implementation
        # Rust returns (pi, q_values) where pi is already normalized
        pi, q_values = self._rust_mcts.run(
            states.astype(np.float32),
            inference_callback,
            num_simulations,
            dirichlet_epsilon,
            dirichlet_alpha,
            c_puct,
            seed,
        )

        # Convert to float32 for consistency with PyTorch
        return pi.astype(np.float32), q_values.astype(np.float32)


# Low-level API: Direct access to Rust implementation
# For advanced users who want full control and don't need the convenience wrapper
RustMCTS = _RustMCTS
