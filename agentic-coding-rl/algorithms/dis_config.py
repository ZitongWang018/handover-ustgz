"""DIS defaults shared by the lightweight launcher and the tensor implementation."""
import math

DEFAULT_DIS_RATIO_LOW = 0.2
DEFAULT_DIS_RATIO_HIGH = 4.0


def validate_dis_bounds(lower: float, upper: float) -> None:
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < 1 < upper):
        raise ValueError("DIS requires finite ratio bounds with 0 < lower < 1 < upper")


def dis_settings(config: dict) -> tuple[float, float]:
    # Previously created local.json files remain usable without regeneration.
    training = config["training"]
    lower = float(training.get("dis_ratio_low", DEFAULT_DIS_RATIO_LOW))
    upper = float(training.get("dis_ratio_high", DEFAULT_DIS_RATIO_HIGH))
    validate_dis_bounds(lower, upper)
    return lower, upper
