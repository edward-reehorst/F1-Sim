"""Offline regularized-least-squares calibration against observed telemetry."""

from __future__ import annotations

import random

import numpy as np
from pydantic import BaseModel, Field

from f1_sim.engine.physics import compute_clean_air_lap_time
from f1_sim.loaders import load_all_compounds, load_all_drivers, load_all_teams, load_circuit
from f1_sim.models.calibration import CalibrationConfig, GridPenalty, GridSlot
from f1_sim.models.telemetry import LapObservation, TelemetryDataset


def _starting_grid_from_dataset(
    dataset: TelemetryDataset,
    grid_penalties: list[GridPenalty] | None = None,
) -> list[GridSlot]:
    """Recover the starting grid from a qualifying dataset.

    Only datasets recorded entirely in a qualifying session carry grid positions;
    race/practice datasets leave the preset grid in place. One slot per driver,
    ordered by position. When ``grid_penalties`` are given they are applied to the
    qualifying classification first (cumulative grid drops); penalty driver IDs
    match either the observation's ``source.driver_code`` or its local driver slug.
    """
    if not dataset.observations or any(o.session != "qualifying" for o in dataset.observations):
        return []
    slots: dict[str, GridSlot] = {}
    for obs in dataset.observations:
        if obs.position is None or obs.driver_id in slots:
            continue
        slots[obs.driver_id] = GridSlot(position=obs.position, driver_id=obs.driver_id, team_id=obs.team_id)

    if grid_penalties:
        from f1_sim.tuning.fastf1 import apply_grid_penalties

        classification = {driver_id: slot.position for driver_id, slot in slots.items()}
        code_to_slug: dict[str, str] = {}
        for obs in dataset.observations:
            code = str((obs.source or {}).get("driver_code", "")).strip().upper()
            if code:
                code_to_slug.setdefault(code, obs.driver_id)
        penalties: list[GridPenalty] = []
        for p in grid_penalties:
            slug = code_to_slug.get(p.driver_id.strip().upper(), p.driver_id)
            if slug not in classification:
                continue
            penalties.append(p.model_copy(update={"driver_id": slug}))
        classified = apply_grid_penalties(classification, penalties)
        for driver_id, position in classified.items():
            slots[driver_id] = slots[driver_id].model_copy(update={"position": position})

    return sorted(slots.values(), key=lambda s: (s.position, s.driver_id))

DEFAULT_FUEL_PENALTY_PER_KG = 0.033
MAX_FUEL_PENALTY_PER_KG = 0.10
_REFERENCE_COMPOUND = "soft"
_NOTE_TEAM_OFFSETS = (
    "team_pace_offsets are not identifiable from a single-circuit dataset (each driver "
    "races for exactly one team); auto-fit pace deltas are attributed to drivers. Team "
    "offsets remain available in the overlay for manual/user-provided settings."
)


class CalibrationReport(BaseModel):
    """Diagnostics for a completed offline calibration fit."""

    circuit_id: str = Field(description="Circuit the fit was performed against")
    n_observations: int = Field(description="Green-flag observations after outlier filtering")
    n_fit_observations: int = Field(description="Observations used to fit the parameters")
    n_holdout_observations: int = Field(default=0, description="Observations held out for validation")
    in_sample_rmse: float = Field(description="Root-mean-square error on the fit set, in seconds")
    holdout_rmse: float | None = Field(default=None, description="Root-mean-square error on the holdout set, if any")
    residuals: list[float] = Field(default_factory=list, description="Fit-set residuals (observed - predicted) in seconds")
    anchor_driver_id: str | None = Field(default=None, description="Driver anchored at 0.0s pace offset")
    fitted_driver_offsets: dict[str, float] = Field(default_factory=dict, description="Recovered relative driver pace offsets in seconds")
    fitted_compound_base_deltas: dict[str, float] = Field(default_factory=dict, description="Recovered compound base delta corrections in seconds")
    fitted_compound_wear_multipliers: dict[str, float] = Field(default_factory=dict, description="Recovered compound wear rate multipliers")
    fitted_compound_cliff_adjustments: dict[str, int] = Field(default_factory=dict, description="Recovered cliff lap adjustments in laps")
    circuit_base_adjust: float = Field(default=0.0, description="Recovered circuit base lap time correction in seconds")
    fuel_penalty_per_kg: float | None = Field(default=None, description="Recovered fuel penalty, or None if untuned")
    ridge_lambda: float = Field(description="Ridge regularization strength used for the linear solve")
    config: CalibrationConfig = Field(description="The fitted overlay, ready to persist or layer into a RaceConfig")
    note: str = Field(default=_NOTE_TEAM_OFFSETS, description="Identifiability caveats")


def _driver_mode(driver_ids: list[str]) -> str:
    """Most common driver (deterministic tie-break by lexicographic order)."""
    counts: dict[str, int] = {}
    for d in driver_ids:
        counts[d] = counts.get(d, 0) + 1
    return sorted(counts, key=lambda d: (-counts[d], d))[0]


def _thin_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Force exact symmetry of a Gram matrix computed in floating point."""
    return (matrix + matrix.T) * 0.5


def _solve_ridge(X: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    """Standardized ridge least-squares solve; returns coefficients on the original X scale."""
    p = X.shape[1]
    if p == 0:
        return np.zeros(0)
    rms = np.sqrt(np.mean(X * X, axis=0))
    rms_safe = np.where(rms < 1e-12, 1.0, rms)
    Xs = X / rms_safe

    if ridge > 0.0:
        gram = _thin_symmetric(Xs.T @ Xs) + ridge * np.eye(p)
        beta_std = np.linalg.solve(gram, Xs.T @ target)
    else:
        beta_std, _, _, _ = np.linalg.lstsq(Xs, target, rcond=None)

    return beta_std / rms_safe


class _Fit:
    """Encapsulates precomputed arrays and the linear model over the full (post-filter) sample."""

    def __init__(self, obs: list[LapObservation], circuit_id: str) -> None:
        self.circuit = load_circuit(circuit_id)
        self.drivers = load_all_drivers()
        self.teams = load_all_teams()
        self.compounds = load_all_compounds()
        self.obs = obs

        n = len(obs)
        self.y = np.array([o.lap_time for o in obs], dtype=float)
        self.age = np.array([o.tire_age_at_lap_start for o in obs], dtype=float)
        self.fuel = np.array(
            [(o.fuel_remaining_kg if o.fuel_remaining_kg is not None else 0.0) for o in obs],
            dtype=float,
        )
        self.has_fuel_all = all(o.fuel_remaining_kg is not None for o in obs)
        self.comp_keys = [o.compound.lower() for o in obs]

        wear = np.array(
            [
                self.drivers[o.driver_id].tire_wear_multiplier if o.driver_id in self.drivers else 1.0
                for o in obs
            ],
            dtype=float,
        )

        self.y_default = np.zeros(n)
        self.cliff_default = np.zeros(n)
        for i, o in enumerate(obs):
            key = o.compound.lower()
            tire = self.compounds[key]
            self.y_default[i] = compute_clean_air_lap_time(
                circuit=self.circuit,
                team=self.teams[o.team_id],
                driver=self.drivers[o.driver_id],
                tire=tire,
                tire_age=o.tire_age_at_lap_start,
                fuel_mass_kg=self.fuel[i],
                fuel_penalty_per_kg=DEFAULT_FUEL_PENALTY_PER_KG,
            )
            self.cliff_default[i] = tire.cliff_penalty_factor * max(0, o.tire_age_at_lap_start - tire.cliff_lap) ** 2

        # Precompute per-compound cliff candidate deltas plus default-form bases.
        self.compounds_present = sorted({k for k in self.comp_keys})
        self.comp_mask: dict[str, np.ndarray] = {}
        self.cliff_default_k: dict[str, np.ndarray] = {}
        self.cliff_penalty: dict[str, float] = {}
        self.cliff_lap: dict[str, int] = {}
        for key in self.compounds_present:
            tire = self.compounds[key]
            mask = np.array([k == key for k in self.comp_keys], dtype=bool)
            self.comp_mask[key] = mask
            self.cliff_default_k[key] = tire.cliff_penalty_factor * np.maximum(0, self.age - tire.cliff_lap) ** 2
            self.cliff_penalty[key] = tire.cliff_penalty_factor
            self.cliff_lap[key] = tire.cliff_lap

    def cliff_delta(self, adjustments: dict[str, int]) -> np.ndarray:
        """Total cliff-term change (candidate - default) across all compounds."""
        delta = np.zeros(self.y.shape[0])
        for key in self.compounds_present:
            adj = adjustments.get(key, 0)
            candidate = self.cliff_penalty[key] * np.maximum(0, self.age - (self.cliff_lap[key] + adj)) ** 2
            delta += self.comp_mask[key] * (candidate - self.cliff_default_k[key])
        return delta

    def design(self, fit_mask: np.ndarray, fit_fuel: bool, anchor_driver: str | None = None) -> tuple[np.ndarray, dict[str, str]]:
        """Design matrix columns for the model, plus a name per column."""
        idx = np.where(fit_mask)[0]
        n = idx.size
        if n == 0:
            raise ValueError("No in-sample observations selected for fitting")

        fit_drivers = sorted({self.obs[i].driver_id for i in idx})
        anchor = anchor_driver if (anchor_driver and anchor_driver in fit_drivers) else _driver_mode(
            [self.obs[i].driver_id for i in idx]
        )
        self.anchor_driver = anchor
        driver_cols = [d for d in fit_drivers if d != anchor]

        fit_compounds = sorted({self.comp_keys[i] for i in idx})
        base_cols = [c for c in fit_compounds if c != _REFERENCE_COMPOUND]

        columns: dict[str, np.ndarray] = {}
        columns["base_adjust"] = np.ones(n, dtype=float)
        col_names: dict[str, str] = {"base_adjust": "circuit base adjust"}

        for _j, d in enumerate(driver_cols):
            name = f"driver:{d}"
            columns[name] = np.array([self.obs[i].driver_id == d for i in idx], dtype=float)

        for c in base_cols:
            name = f"compound_base:{c}"
            columns[name] = np.array([self.comp_keys[i] == c for i in idx], dtype=float)

        wear_mult = np.array(
            [
                self.drivers[self.obs[i].driver_id].tire_wear_multiplier if self.obs[i].driver_id in self.drivers else 1.0
                for i in idx
            ],
            dtype=float,
        )
        for c in fit_compounds:
            tire = self.compounds[c]
            age_at = np.array([self.obs[i].tire_age_at_lap_start for i in idx], dtype=float)
            mask = np.array([self.comp_keys[i] == c for i in idx], dtype=float)
            basis = mask * tire.wear_rate * self.circuit.tire_wear_factor * wear_mult * age_at
            if np.max(np.abs(basis)) > 0:
                columns[f"wear:{c}"] = basis

        if fit_fuel and self.has_fuel_all and np.max(np.abs(np.array([self.fuel[i] for i in idx]))) > 0:
            columns["fuel"] = np.array([self.fuel[i] for i in idx], dtype=float)

        col_names.update({name: name for name in columns if name != "base_adjust"})
        col_order = list(columns)
        X = np.column_stack([columns[name] for name in col_order])
        return X, col_order


def calibrate_from_dataset(
    dataset: TelemetryDataset,
    circuit_id: str,
    *,
    holdout: float = 0.0,
    seed: int = 42,
    ridge: float = 0.5,
    anchor_driver_id: str | None = None,
    max_cliff_adjust: int = 8,
    fit_fuel_penalty: bool = True,
    grid_penalties: list[GridPenalty] | None = None,
) -> CalibrationReport:
    """Fit a ``CalibrationConfig`` overlay against observed laps for a circuit.

    The model exploits the linear structure of the additive lap-time formula:
    prediction = bundled default lap time + base_adjust + per-driver offset +
    per-compound base correction + per-compound wear correction * age basis +
    fuel correction, with per-compound cliff thresholds selected by deterministic
    grid search. A ridge penalty pulls parameters toward the bundled defaults to
    avoid overfitting sparse sessions. The fit is deterministic for a given dataset
    and seed: no stochastic search or sampling is used outside the optional holdout split.
    """
    if not 0.0 <= holdout < 1.0:
        raise ValueError(f"holdout must be in [0, 1), got {holdout}")
    if dataset.circuit_id and dataset.circuit_id != circuit_id:
        raise ValueError(
            f"Dataset circuit '{dataset.circuit_id}' does not match requested circuit '{circuit_id}'. "
            "Pass the circuit the observations were recorded on."
        )

    obs = dataset.filter_for_calibration()
    if not obs:
        raise ValueError(
            "No green-flag observations remain after outlier filtering; cannot calibrate."
        )

    fit = _Fit(obs, circuit_id)
    n = len(obs)

    if holdout > 0.0:
        order = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(order)
        n_holdout = int(round(n * holdout))
        hold_idx = np.array(sorted(order[:n_holdout]), dtype=int)
        fit_idx = np.array(sorted(order[n_holdout:]), dtype=int)
    else:
        fit_idx = np.arange(n)
        hold_idx = np.array([], dtype=int)

    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[fit_idx] = True
    holdout_mask = np.zeros(n, dtype=bool)
    holdout_mask[hold_idx] = True

    # Anchoring: reference driver at 0.0s offset, reference compound base at 0s.
    fit_drivers = {fit.obs[i].driver_id for i in fit_idx}
    anchor_driver = (
        anchor_driver_id
        if anchor_driver_id and anchor_driver_id in fit_drivers
        else _driver_mode([fit.obs[i].driver_id for i in fit_idx])
    )
    X, col_order = fit.design(fit_mask, fit_fuel_penalty, anchor_driver=anchor_driver)

    def _target(adjustments: dict[str, int]) -> np.ndarray:
        return fit.y[fit_idx] - fit.y_default[fit_idx] - fit.cliff_delta(adjustments)[fit_idx]

    def _solve(adjustments: dict[str, int]) -> tuple[np.ndarray, np.ndarray, float]:
        target = _target(adjustments)
        beta = _solve_ridge(X, target, ridge)
        pred = fit.y_default[fit_idx] + fit.cliff_delta(adjustments)[fit_idx] + X @ beta
        rmse = float(np.sqrt(np.mean((fit.y[fit_idx] - pred) ** 2)))
        return beta, pred, rmse

    # Per-compound cliff-threshold grid search (deterministic coordinate ascent).
    # Competing structures with statistically identical fit are resolved toward the
    # bundled default (smallest |adjustment|), so unobservable cliffs stay at 0.
    adjustments: dict[str, int] = {c: 0 for c in fit.compounds_present}
    for _ in range(4):
        changed = False
        for key in sorted(adjustments):
            scored: dict[int, float] = {}
            for candidate in range(-max_cliff_adjust, max_cliff_adjust + 1):
                adjustments[key] = candidate
                _, _, rmse = _solve(adjustments)
                scored[candidate] = rmse
            best = min(scored, key=lambda c: (scored[c], abs(c), c))
            if best != adjustments[key]:
                adjustments[key] = best
                changed = True
        if not changed:
            break

    beta, _, _ = _solve(adjustments)

    # Resolve fitted parameter values from the beta vector.
    base_adjust = float(beta[0]) if beta.size else 0.0
    driver_offsets: dict[str, float] = {}
    compound_base: dict[str, float] = {}
    wear_scale: dict[str, float] = {}
    fuel_penalty: float | None = None

    by_name = dict(zip(col_order, beta, strict=True))
    for name, value in by_name.items():
        if name.startswith("driver:"):
            driver_offsets[name.split(":", 1)[1]] = float(value)
        elif name.startswith("compound_base:"):
            compound_base[name.split(":", 1)[1]] = float(value)
        elif name.startswith("wear:"):
            compound_key = name.split(":", 1)[1]
            wear_scale[compound_key] = max(0.0, 1.0 + float(value))
        elif name == "fuel":
            fuel_penalty = min(MAX_FUEL_PENALTY_PER_KG, max(0.0, DEFAULT_FUEL_PENALTY_PER_KG + float(value)))

    # Anchor driver explicitly at zero.
    if anchor_driver is not None:
        driver_offsets[anchor_driver] = 0.0

    cliff_adjustments = {k: int(v) for k, v in adjustments.items() if v != 0}

    config = CalibrationConfig(
        driver_pace_offsets=driver_offsets,
        compound_base_deltas=compound_base,
        compound_wear_multipliers=wear_scale,
        compound_cliff_adjustments=cliff_adjustments,
        circuit_base_adjust=base_adjust,
        fuel_penalty_per_kg=fuel_penalty,
        starting_grid=_starting_grid_from_dataset(dataset, grid_penalties),
    )

    # Re-evaluate predictions with the constrained overlay so the report matches
    # exactly what build_race_config(calibration=...) will produce.
    predictions = np.array(
        [
            compute_clean_air_lap_time(
                circuit=fit.circuit,
                team=fit.teams[fit.obs[i].team_id],
                driver=fit.drivers[fit.obs[i].driver_id],
                tire=fit.compounds[fit.comp_keys[i]],
                tire_age=fit.obs[i].tire_age_at_lap_start,
                fuel_mass_kg=fit.fuel[i],
                fuel_penalty_per_kg=DEFAULT_FUEL_PENALTY_PER_KG,
                calibration=config,
            )
            for i in range(n)
        ],
        dtype=float,
    )
    residuals = fit.y - predictions
    in_sample_rmse = float(
        np.sqrt(np.mean(residuals[fit_idx] ** 2)) if fit_idx.size else float("nan")
    )
    holdout_rmse = float(np.sqrt(np.mean(residuals[hold_idx] ** 2))) if hold_idx.size else None

    return CalibrationReport(
        circuit_id=circuit_id,
        n_observations=n,
        n_fit_observations=int(fit_idx.size),
        n_holdout_observations=int(hold_idx.size),
        in_sample_rmse=in_sample_rmse,
        holdout_rmse=holdout_rmse,
        residuals=[float(r) for r in residuals[fit_idx]],
        anchor_driver_id=anchor_driver,
        fitted_driver_offsets=driver_offsets,
        fitted_compound_base_deltas=compound_base,
        fitted_compound_wear_multipliers=wear_scale,
        fitted_compound_cliff_adjustments=cliff_adjustments,
        circuit_base_adjust=base_adjust,
        fuel_penalty_per_kg=fuel_penalty,
        ridge_lambda=ridge,
        config=config,
    )