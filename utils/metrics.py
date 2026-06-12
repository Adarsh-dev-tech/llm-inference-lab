def calculate_tps(tokens: int, duration_ms: float) -> float:
    """Calculates throughput (tokens per second) given token count and duration in ms."""
    if duration_ms <= 0:
        return 0.0
    return round((tokens / duration_ms) * 1000.0, 2)

def to_ms(seconds: float) -> float:
    """Converts duration in seconds to milliseconds, rounded to 2 decimal places."""
    return round(seconds * 1000.0, 2)
