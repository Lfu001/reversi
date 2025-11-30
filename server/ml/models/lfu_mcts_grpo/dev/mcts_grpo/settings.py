from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    num_channels: int = Field(description="Number of channels in the neural network")
    num_residual_blocks: int = Field(
        description="Number of residual blocks in the network"
    )


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


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    batch_size: int = Field(description="Training batch size")
    replay_buffer_size: int = Field(description="Size of the experience replay buffer")
    learning_rate: float = Field(description="Learning rate for the optimizer")
    weight_decay: float = Field(description="Weight decay for the optimizer")
    total_training_steps: int = Field(description="Total number of training steps")
    games_per_iteration: int = Field(
        description="Number of self-play games per training iteration"
    )
    training_steps_per_iteration: int = Field(
        description="Training steps per iteration"
    )
    lambda_value: float = Field(description="Lambda value for value loss")


class TreeGRPOConfig(BaseModel):
    """Configuration for Tree-GRPO algorithm."""

    lambda_grpo: float = Field(description="Lambda value for GRPO loss")
    grpo_num_pairs: int = Field(description="Number of state-action pairs for GRPO")


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
