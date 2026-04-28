class TDDataError(RuntimeError):
    """Raised when input market data is invalid or unavailable."""


class TDSignalError(RuntimeError):
    """Raised when signal generation or signal inputs are invalid."""
