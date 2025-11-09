"""
トレーニング関連モジュール
"""

from .executor import TrainingExecutor
from .game_result import GameResultProcessor, Winner
from .self_play import SelfPlayExecutor
from .setup import GameSetup

__all__ = [
    "GameSetup",
    "SelfPlayExecutor",
    "GameResultProcessor",
    "TrainingExecutor",
    "Winner",
]
