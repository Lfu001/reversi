class ReversiError(Exception):
    """Base exception for all Reversi-related errors."""


class InvalidMoveError(ReversiError):
    """Raised when an invalid move is attempted."""
