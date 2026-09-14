"""Race control flags and track status enumerations."""

from enum import Enum


class RaceFlag(str, Enum):
    """Track condition and race control status flags."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    VSC = "VSC"
    SAFETY_CAR = "SAFETY_CAR"
