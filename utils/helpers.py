import numpy as np


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def minutes(hours):
    return hours * 60.0


def hours(minutes_value):
    return minutes_value / 60.0
