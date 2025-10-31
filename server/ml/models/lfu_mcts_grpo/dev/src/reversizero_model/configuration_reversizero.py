from transformers import PretrainedConfig


class ReversiZeroConfig(PretrainedConfig):
    model_type = "reversi_zero"

    def __init__(
        self,
        # Model Architecture
        num_channels: int = 128,
        num_residual_blocks: int = 8,
        # MCTS
        num_simulations: int = 10,
        dirichlet_epsilon: float = 0.25,
        dirichlet_alpha: float = 0.3,
        c_puct: float = 1.25,
        # Training
        batch_size: int = 512,
        replay_buffer_size: int = 600_000,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        total_training_steps: int = 3_000,
        games_per_iteration: int = 512,
        training_steps_per_iteration: int = 60,
        lambda_value: float = 3.0,
        # Tree-GRPO
        lambda_grpo: float = 1.0,
        grpo_num_pairs: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.num_residual_blocks = num_residual_blocks
        self.num_simulations = num_simulations
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.total_training_steps = total_training_steps
        self.games_per_iteration = games_per_iteration
        self.training_steps_per_iteration = training_steps_per_iteration
        self.lambda_value = lambda_value
        self.lambda_grpo = lambda_grpo
        self.grpo_num_pairs = grpo_num_pairs
