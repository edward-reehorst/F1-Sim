"""Traffic dynamics, dirty air effects, and pace-offset overtaking logic."""

import math
from pydantic import BaseModel, Field


class OvertakeEvent(BaseModel):
    """Details of an attempted or completed on-track overtake."""

    lap: int = Field(ge=1, description="Race lap on which overtake was attempted")
    attacker_id: str = Field(description="Driver ID of trailing attacker")
    defender_id: str = Field(description="Driver ID of leading defender")
    position_gained: int = Field(ge=1, description="Track position fought for (e.g. 1 for P1)")
    pace_delta: float = Field(description="Net pace advantage of attacker (LapTime_defender - LapTime_attacker)")
    probability: float = Field(ge=0.0, le=1.0, description="Calculated probability of successful pass")
    success: bool = Field(description="Whether the overtake succeeded")
    description: str = Field(description="Human-readable summary of the event")


def compute_overtake_threshold(
    circuit_overtaking_difficulty: float,
    base_threshold_seconds: float = 0.35,
) -> float:
    """Calculate the pace delta required to launch a competitive overtake attempt.

    Formula: threshold = base_threshold * (1.0 + circuit_difficulty)
    At easy circuits (Monza ~0.25): threshold = 0.35 * 1.25 = 0.4375s
    At hard circuits (Monaco ~0.95): threshold = 0.35 * 1.95 = 0.6825s
    """
    return base_threshold_seconds * (1.0 + circuit_overtaking_difficulty)


def compute_overtake_probability(
    pace_delta: float,
    threshold: float,
    sensitivity: float = 8.0,
) -> float:
    """Compute the probability of an overtake based purely on pace offset.

    P(Overtake) = 1 / (1 + exp(-sensitivity * (pace_delta - threshold)))

    If pace_delta <= 0 (trailing car is slower or equal): returns 0.0.
    """
    if pace_delta <= 0.0:
        return 0.0

    exponent = -sensitivity * (pace_delta - threshold)
    # Clamp exponent to prevent numerical overflow in extreme deltas
    exponent = max(-50.0, min(50.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


def evaluate_overtake(
    lap: int,
    attacker_id: str,
    defender_id: str,
    position: int,
    pace_delta: float,
    circuit_difficulty: float,
    base_threshold: float,
    rng_value: float,
    sensitivity: float = 8.0,
) -> tuple[bool, OvertakeEvent]:
    """Evaluate whether an overtake attempt succeeds given the pace delta and an RNG roll [0, 1)."""
    threshold = compute_overtake_threshold(
        circuit_overtaking_difficulty=circuit_difficulty,
        base_threshold_seconds=base_threshold,
    )
    prob = compute_overtake_probability(
        pace_delta=pace_delta,
        threshold=threshold,
        sensitivity=sensitivity,
    )

    success = (prob > 0.0) and (rng_value < prob)

    if success:
        desc = (
            f"Lap {lap}: {attacker_id.upper()} overtook {defender_id.upper()} for P{position} "
            f"(delta: +{pace_delta:.3f}s, prob: {prob * 100:.1f}%)"
        )
    else:
        desc = (
            f"Lap {lap}: {attacker_id.upper()} challenged {defender_id.upper()} for P{position} "
            f"but could not pass (delta: +{pace_delta:.3f}s, prob: {prob * 100:.1f}%)"
        )

    event = OvertakeEvent(
        lap=lap,
        attacker_id=attacker_id,
        defender_id=defender_id,
        position_gained=position,
        pace_delta=pace_delta,
        probability=prob,
        success=success,
        description=desc,
    )

    return success, event
