"""Small shared helpers for the dashboard's persisted execution clocks."""

# A single precision rule prevents live, completed, failed, and replayed stages
# from presenting the same stopwatch value with different visual precision.
STOPWATCH_DECIMALS = 3


def round_stopwatch(seconds: float) -> float:
    """Persist a stopwatch value at the dashboard's three-decimal precision."""
    return round(float(seconds), STOPWATCH_DECIMALS)


def format_stopwatch(seconds: float) -> str:
    """Render a stopwatch value with a fixed fractional width and seconds unit."""
    return f"{float(seconds):.{STOPWATCH_DECIMALS}f}s"
