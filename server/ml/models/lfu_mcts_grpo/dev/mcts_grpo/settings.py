from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class ModelConfig(BaseModel):
    """Model architecture configuration for URM (Universal Reasoning Model)."""

    hidden_size: int = Field(description="Hidden dimension size")
    num_attention_heads: int = Field(description="Number of attention heads")
    num_layers: int = Field(description="Number of transformer layers (shared)")
    num_inner_loops: int = Field(description="Number of inner loop iterations")
    truncation_steps: int = Field(description="Forward-only steps for TBPTL")
    expansion_ratio: int = Field(description="MLP expansion ratio")
    conv_kernel_size: int = Field(description="Depthwise convolution kernel size")
    dropout: float = Field(description="Dropout rate")
    layer_norm_eps: float = Field(description="Layer normalization epsilon")


class MCTSConfig(BaseModel):
    """Configuration for Monte Carlo Tree Search."""

    num_simulations: int = Field(description="Number of MCTS simulations per move")
    dirichlet_epsilon: float = Field(description="Epsilon for Dirichlet noise")
    dirichlet_alpha: float = Field(
        description="Alpha parameter for Dirichlet distribution"
    )
    c_puct: float = Field(description="Exploration constant for PUCT algorithm")
    max_inference_batch_size: int = Field(
        description="Maximum batch size for inference worker"
    )
    states_per_inference: int = Field(
        description="Number of states to process per inference call"
    )
    parallel_games: int = Field(description="Number of parallel games during self-play")
    games_per_iteration: int = Field(
        description="Number of games to complete per iteration"
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    train_batch_size: int = Field(description="Training batch size")
    replay_buffer_size: int = Field(description="Size of the experience replay buffer")
    learning_rate: float = Field(description="Learning rate for the optimizer")
    weight_decay: float = Field(description="Weight decay for the optimizer")
    total_training_steps: int = Field(description="Total number of training steps")
    training_steps_per_iteration: int = Field(
        description="Training steps per iteration"
    )
    warmup_steps: int = Field(description="Number of warmup steps for LR scheduler")
    lambda_value: float = Field(description="Lambda value for value loss")


class TreeGRPOConfig(BaseModel):
    """Configuration for Tree-GRPO algorithm."""

    lambda_grpo: float = Field(description="Lambda value for GRPO loss")
    clip_epsilon: float = Field(
        default=0.2, description="Clipping epsilon for PPO-style objective"
    )


class Settings(BaseSettings):
    """
    Main settings class that loads configurations from config.toml.
    All configuration values should be defined in config.toml.
    """

    # Model configuration
    model: ModelConfig

    # Sub-configurations
    mcts: MCTSConfig
    training: TrainingConfig
    grpo: TreeGRPOConfig

    # Configuration file path
    model_config = SettingsConfigDict(
        toml_file=Path(__file__).resolve().parent.parent / "config.toml"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)
