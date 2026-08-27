from __future__ import annotations

import argparse
import io
import math
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


NASA_ZIP_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

MIRROR_URLS = {
    "train_FD001.txt": (
        "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/"
        "master/notebooks/jupyter/Dataset/CMAPSSData/train_FD001.txt"
    ),
    "test_FD001.txt": (
        "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/"
        "master/notebooks/jupyter/Dataset/CMAPSSData/test_FD001.txt"
    ),
    "RUL_FD001.txt": (
        "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/"
        "master/notebooks/jupyter/Dataset/CMAPSSData/RUL_FD001.txt"
    ),
}

COLUMNS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

INFORMATIVE_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]

EXPECTED_FD001 = {
    "train_rows": 20631,
    "test_rows": 13096,
    "train_units": 100,
    "test_units": 100,
    "rul_rows": 100,
}


@dataclass
class RULModel:
    estimator: ExtraTreesRegressor
    feature_columns: List[str]
    calibration_residuals: np.ndarray
    calibration_abs_error_quantile90: float
    rul_cap: float
    train_units: Tuple[int, ...]
    calibration_units: Tuple[int, ...]


@dataclass(frozen=True)
class RULMetrics:
    rmse: float
    mae: float
    interval90_coverage: float
    interval90_mean_width: float


@dataclass(frozen=True)
class EngineRisk:
    engine_id: int
    predicted_rul: float
    lower90_rul: float
    upper90_rul: float
    failure_probability_by_period: Tuple[float, ...]


@dataclass(frozen=True)
class MaintenancePlan:
    periods: Tuple[int, ...]
    assignments: Dict[int, int | None]
    period_loads: Dict[int, int]
    expected_cost: float
    objective_status: str


@dataclass(frozen=True)
class EndToEndResult:
    metrics: RULMetrics
    optimized_plan: MaintenancePlan
    baseline_plan: MaintenancePlan
    engine_risks: Tuple[EngineRisk, ...]


def _request_bytes(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "JORS-Academy-Predictive-Maintenance/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_fd001(data_dir: str | Path) -> Dict[str, Path]:
    """
    Download NASA C-MAPSS FD001.

    Primary source: NASA Open Data legacy ZIP.
    Fallback: public GitHub mirror containing the same C-MAPSS text files.

    Raw data are not committed by this project.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        name: data_dir / name
        for name in MIRROR_URLS
    }
    if all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
        return paths

    try:
        payload = _request_bytes(NASA_ZIP_URL)
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = zf.namelist()
            for target_name, target_path in paths.items():
                matches = [
                    name for name in names
                    if name.endswith(target_name)
                ]
                if not matches:
                    raise FileNotFoundError(
                        f"{target_name} missing from NASA archive"
                    )
                with zf.open(matches[0]) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        return paths
    except Exception:
        pass

    for name, url in MIRROR_URLS.items():
        payload = _request_bytes(url)
        paths[name].write_bytes(payload)

    return paths


def parse_cmapss_text(path_or_buffer) -> pd.DataFrame:
    frame = pd.read_csv(
        path_or_buffer,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
    )
    if frame.shape[1] != 26:
        raise ValueError(f"expected 26 columns, got {frame.shape[1]}")
    if frame.isna().any().any():
        raise ValueError("C-MAPSS data contain missing values")
    frame["unit"] = frame["unit"].astype(int)
    frame["cycle"] = frame["cycle"].astype(int)
    if (frame["unit"] <= 0).any() or (frame["cycle"] <= 0).any():
        raise ValueError("unit and cycle identifiers must be positive")
    return frame


def load_fd001(data_dir: str | Path, *, download: bool = True):
    data_dir = Path(data_dir)
    if download:
        paths = download_fd001(data_dir)
    else:
        paths = {
            name: data_dir / name
            for name in MIRROR_URLS
        }

    train = parse_cmapss_text(paths["train_FD001.txt"])
    test = parse_cmapss_text(paths["test_FD001.txt"])
    rul = pd.read_csv(
        paths["RUL_FD001.txt"],
        sep=r"\s+",
        header=None,
        names=["rul"],
        engine="python",
    )
    rul = rul.dropna().reset_index(drop=True)
    rul["rul"] = rul["rul"].astype(float)

    validate_fd001_dimensions(train, test, rul)
    return train, test, rul


def validate_fd001_dimensions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rul: pd.DataFrame,
) -> None:
    checks = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_units": train["unit"].nunique(),
        "test_units": test["unit"].nunique(),
        "rul_rows": len(rul),
    }
    for key, expected in EXPECTED_FD001.items():
        if checks[key] != expected:
            raise ValueError(
                f"FD001 integrity check failed for {key}: "
                f"{checks[key]} != {expected}"
            )


def add_training_rul(
    train: pd.DataFrame,
    *,
    rul_cap: float = 125.0,
) -> pd.DataFrame:
    if rul_cap <= 0:
        raise ValueError("rul_cap must be positive")
    out = train.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul_raw"] = max_cycle - out["cycle"]
    out["rul"] = np.minimum(out["rul_raw"], rul_cap).astype(float)
    return out


def engineer_features(
    frame: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.DataFrame:
    if window < 2:
        raise ValueError("window must be >= 2")
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing C-MAPSS columns: {missing}")

    out = frame.sort_values(["unit", "cycle"]).copy()
    grouped = out.groupby("unit", sort=False)

    for sensor in INFORMATIVE_SENSORS:
        out[f"{sensor}_mean{window}"] = (
            grouped[sensor]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out[f"{sensor}_std{window}"] = (
            grouped[sensor]
            .rolling(window, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        lagged = grouped[sensor].shift(window - 1)
        elapsed = np.minimum(out["cycle"] - 1, window - 1).clip(lower=1)
        out[f"{sensor}_trend{window}"] = (
            (out[sensor] - lagged.fillna(out[sensor]))
            / elapsed
        )

    return out


def model_feature_columns(window: int = 20) -> List[str]:
    columns = ["cycle"]
    for sensor in INFORMATIVE_SENSORS:
        columns.extend(
            [
                sensor,
                f"{sensor}_mean{window}",
                f"{sensor}_std{window}",
                f"{sensor}_trend{window}",
            ]
        )
    return columns


def split_engine_ids(
    engine_ids: Sequence[int],
    *,
    calibration_fraction: float = 0.30,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.array(sorted(set(map(int, engine_ids))), dtype=int)
    if len(ids) < 4:
        raise ValueError("need at least four engines")
    if not 0.1 <= calibration_fraction <= 0.5:
        raise ValueError("calibration_fraction must be in [0.1,0.5]")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    n_cal = max(2, int(round(len(ids) * calibration_fraction)))
    calibration = np.sort(shuffled[:n_cal])
    train = np.sort(shuffled[n_cal:])
    if len(set(train) & set(calibration)):
        raise RuntimeError("engine-level split leakage")
    return train, calibration


def _last_rows(frame: pd.DataFrame) -> pd.DataFrame:
    idx = frame.groupby("unit")["cycle"].idxmax()
    return frame.loc[idx].sort_values("unit").reset_index(drop=True)


def calibration_snapshots(
    labeled_features: pd.DataFrame,
    calibration_units: Sequence[int],
    *,
    seed: int,
    rul_cap: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in calibration_units:
        engine = labeled_features[
            labeled_features["unit"] == int(unit)
        ]
        if engine.empty:
            raise ValueError(f"missing calibration engine {unit}")

        target_rul = float(rng.uniform(0.0, rul_cap))
        idx = (engine["rul"] - target_rul).abs().idxmin()
        rows.append(labeled_features.loc[idx])

    out = pd.DataFrame(rows).reset_index(drop=True)
    if out["unit"].nunique() != len(calibration_units):
        raise RuntimeError("calibration snapshot engine duplication")
    return out


def split_conformal_abs_quantile(
    residuals: np.ndarray,
    *,
    alpha: float = 0.10,
) -> float:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1 or len(residuals) < 2:
        raise ValueError("need at least two residuals")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")

    scores = np.sort(np.abs(residuals))
    n = len(scores)
    k = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
    return float(scores[k - 1])


def fit_rul_model(
    train_raw: pd.DataFrame,
    *,
    seed: int = 42,
    window: int = 20,
    rul_cap: float = 125.0,
) -> RULModel:
    labeled = add_training_rul(train_raw, rul_cap=rul_cap)
    features = engineer_features(labeled, window=window)
    cols = model_feature_columns(window)

    model_units, calibration_units = split_engine_ids(
        features["unit"].unique(),
        calibration_fraction=0.30,
        seed=seed,
    )

    fit_rows = features[features["unit"].isin(model_units)]
    cal_rows = features[features["unit"].isin(calibration_units)]

    estimator = ExtraTreesRegressor(
        n_estimators=160,
        min_samples_leaf=2,
        max_features=0.75,
        random_state=seed,
        n_jobs=-1,
    )
    estimator.fit(fit_rows[cols], fit_rows["rul"])

    cal_points = calibration_snapshots(
        cal_rows,
        calibration_units,
        seed=seed + 10_000,
        rul_cap=rul_cap,
    )
    cal_pred = np.clip(
        estimator.predict(cal_points[cols]),
        0.0,
        rul_cap,
    )
    residuals = cal_points["rul"].to_numpy(dtype=float) - cal_pred
    abs_q90 = split_conformal_abs_quantile(
        residuals,
        alpha=0.10,
    )

    return RULModel(
        estimator=estimator,
        feature_columns=cols,
        calibration_residuals=np.asarray(residuals, dtype=float),
        calibration_abs_error_quantile90=abs_q90,
        rul_cap=float(rul_cap),
        train_units=tuple(map(int, model_units)),
        calibration_units=tuple(map(int, calibration_units)),
    )


def predict_test_endpoints(
    model: RULModel,
    test_raw: pd.DataFrame,
    rul_labels: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.DataFrame:
    features = engineer_features(test_raw, window=window)
    terminal = _last_rows(features)

    if len(terminal) != len(rul_labels):
        raise ValueError("test engine count and RUL label count differ")

    prediction = np.clip(
        model.estimator.predict(terminal[model.feature_columns]),
        0.0,
        model.rul_cap,
    )
    true_raw = rul_labels["rul"].to_numpy(dtype=float)
    true_capped = np.minimum(true_raw, model.rul_cap)

    q = model.calibration_abs_error_quantile90
    lower = np.clip(prediction - q, 0.0, model.rul_cap)
    upper = np.clip(prediction + q, 0.0, model.rul_cap)

    return pd.DataFrame(
        {
            "unit": terminal["unit"].to_numpy(dtype=int),
            "cycle": terminal["cycle"].to_numpy(dtype=int),
            "predicted_rul": prediction,
            "true_rul_raw": true_raw,
            "true_rul_capped": true_capped,
            "lower90_rul": lower,
            "upper90_rul": upper,
        }
    )


def evaluate_predictions(predictions: pd.DataFrame) -> RULMetrics:
    y = predictions["true_rul_capped"].to_numpy(dtype=float)
    p = predictions["predicted_rul"].to_numpy(dtype=float)
    lower = predictions["lower90_rul"].to_numpy(dtype=float)
    upper = predictions["upper90_rul"].to_numpy(dtype=float)

    rmse = float(math.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    coverage = float(np.mean((y >= lower) & (y <= upper)))
    width = float(np.mean(upper - lower))

    return RULMetrics(
        rmse=rmse,
        mae=mae,
        interval90_coverage=coverage,
        interval90_mean_width=width,
    )


def failure_probability(
    predicted_rul: float,
    horizon_cycles: float,
    calibration_residuals: np.ndarray,
) -> float:
    residuals = np.asarray(calibration_residuals, dtype=float)
    if residuals.ndim != 1 or len(residuals) < 2:
        raise ValueError("need at least two calibration residuals")
    threshold = float(horizon_cycles) - float(predicted_rul)
    return float(np.mean(residuals <= threshold))


def build_engine_risks(
    predictions: pd.DataFrame,
    model: RULModel,
    periods: Sequence[int],
) -> Tuple[EngineRisk, ...]:
    periods = tuple(map(int, periods))
    if any(p <= 0 for p in periods) or tuple(sorted(periods)) != periods:
        raise ValueError("periods must be positive and sorted")

    risks = []
    for row in predictions.itertuples(index=False):
        probs = tuple(
            failure_probability(
                row.predicted_rul,
                period,
                model.calibration_residuals,
            )
            for period in periods
        )
        if any(
            probs[i] > probs[i + 1] + 1e-12
            for i in range(len(probs) - 1)
        ):
            raise RuntimeError("failure probability must be nondecreasing")
        risks.append(
            EngineRisk(
                engine_id=int(row.unit),
                predicted_rul=float(row.predicted_rul),
                lower90_rul=float(row.lower90_rul),
                upper90_rul=float(row.upper90_rul),
                failure_probability_by_period=probs,
            )
        )
    return tuple(risks)


def optimize_maintenance_schedule(
    engine_risks: Sequence[EngineRisk],
    *,
    periods: Sequence[int] = (10, 20, 30, 40, 50),
    capacity_per_period: int | Sequence[int] = 8,
    preventive_cost: float = 25_000.0,
    planned_downtime_cost: float = 10_000.0,
    failure_cost: float = 250_000.0,
) -> MaintenancePlan:
    risks = list(engine_risks)
    periods = tuple(map(int, periods))
    n = len(risks)
    T = len(periods)
    if n == 0 or T == 0:
        raise ValueError("empty engine set or planning horizon")

    if isinstance(capacity_per_period, int):
        capacities = [int(capacity_per_period)] * T
    else:
        capacities = list(map(int, capacity_per_period))
        if len(capacities) != T:
            raise ValueError("capacity vector length mismatch")
    if any(c < 0 for c in capacities):
        raise ValueError("maintenance capacities must be nonnegative")

    def x_idx(i, t):
        return i * T + t

    u_offset = n * T

    def u_idx(i):
        return u_offset + i

    nvars = n * T + n
    c = np.zeros(nvars, dtype=float)

    for i, risk in enumerate(risks):
        if len(risk.failure_probability_by_period) != T:
            raise ValueError("risk period count mismatch")
        for t in range(T):
            p_fail = risk.failure_probability_by_period[t]
            c[x_idx(i, t)] = (
                preventive_cost
                + planned_downtime_cost
                + p_fail * failure_cost
            )
        c[u_idx(i)] = (
            risk.failure_probability_by_period[-1] * failure_cost
        )

    rows = []
    lb = []
    ub = []

    for i in range(n):
        row = np.zeros(nvars)
        for t in range(T):
            row[x_idx(i, t)] = 1.0
        row[u_idx(i)] = 1.0
        rows.append(row)
        lb.append(1.0)
        ub.append(1.0)

    for t in range(T):
        row = np.zeros(nvars)
        for i in range(n):
            row[x_idx(i, t)] = 1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(float(capacities[t]))

    result = milp(
        c=c,
        integrality=np.ones(nvars, dtype=int),
        bounds=Bounds(
            np.zeros(nvars),
            np.ones(nvars),
        ),
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(lb),
            np.asarray(ub),
        ),
        options={"time_limit": 60.0},
    )

    if result.x is None:
        raise RuntimeError("maintenance MILP returned no solution")

    assignments: Dict[int, int | None] = {}
    period_loads = {period: 0 for period in periods}

    for i, risk in enumerate(risks):
        assigned = None
        for t, period in enumerate(periods):
            if result.x[x_idx(i, t)] > 0.5:
                assigned = period
                period_loads[period] += 1
                break
        assignments[risk.engine_id] = assigned

    status = "OPTIMAL" if result.status == 0 else "FEASIBLE_LIMIT"
    return MaintenancePlan(
        periods=periods,
        assignments=assignments,
        period_loads=period_loads,
        expected_cost=float(result.fun),
        objective_status=status,
    )


def threshold_baseline_schedule(
    engine_risks: Sequence[EngineRisk],
    *,
    periods: Sequence[int] = (10, 20, 30, 40, 50),
    capacity_per_period: int | Sequence[int] = 8,
    threshold_rul: float = 50.0,
    preventive_cost: float = 25_000.0,
    planned_downtime_cost: float = 10_000.0,
    failure_cost: float = 250_000.0,
) -> MaintenancePlan:
    periods = tuple(map(int, periods))
    T = len(periods)
    if isinstance(capacity_per_period, int):
        capacities = [capacity_per_period] * T
    else:
        capacities = list(capacity_per_period)

    remaining = list(map(int, capacities))
    assignments = {r.engine_id: None for r in engine_risks}

    candidates = sorted(
        [r for r in engine_risks if r.predicted_rul <= threshold_rul],
        key=lambda r: (r.predicted_rul, r.engine_id),
    )

    risk_by_id = {r.engine_id: r for r in engine_risks}
    for risk in candidates:
        for t, period in enumerate(periods):
            if remaining[t] > 0:
                assignments[risk.engine_id] = period
                remaining[t] -= 1
                break

    loads = {
        period: sum(v == period for v in assignments.values())
        for period in periods
    }

    total = 0.0
    for engine_id, assigned in assignments.items():
        risk = risk_by_id[engine_id]
        if assigned is None:
            total += risk.failure_probability_by_period[-1] * failure_cost
        else:
            t = periods.index(assigned)
            total += (
                preventive_cost
                + planned_downtime_cost
                + risk.failure_probability_by_period[t] * failure_cost
            )

    return MaintenancePlan(
        periods=periods,
        assignments=assignments,
        period_loads=loads,
        expected_cost=float(total),
        objective_status="RULE_BASELINE",
    )


def run_end_to_end(
    data_dir: str | Path,
    *,
    seed: int = 42,
    periods: Sequence[int] = (10, 20, 30, 40, 50),
    capacity_per_period: int = 8,
) -> EndToEndResult:
    train, test, rul = load_fd001(data_dir, download=True)
    model = fit_rul_model(train, seed=seed)
    predictions = predict_test_endpoints(model, test, rul)
    metrics = evaluate_predictions(predictions)

    risks = build_engine_risks(predictions, model, periods)
    optimized = optimize_maintenance_schedule(
        risks,
        periods=periods,
        capacity_per_period=capacity_per_period,
    )
    baseline = threshold_baseline_schedule(
        risks,
        periods=periods,
        capacity_per_period=capacity_per_period,
    )

    return EndToEndResult(
        metrics=metrics,
        optimized_plan=optimized,
        baseline_plan=baseline,
        engine_risks=risks,
    )


def brute_force_schedule_cost(
    engine_risks: Sequence[EngineRisk],
    periods: Sequence[int],
    capacities: Sequence[int],
    *,
    preventive_cost: float,
    planned_downtime_cost: float,
    failure_cost: float,
) -> float:
    import itertools

    risks = list(engine_risks)
    periods = tuple(periods)
    options = list(periods) + [None]
    best = math.inf

    for choices in itertools.product(options, repeat=len(risks)):
        if any(
            sum(choice == period for choice in choices) > capacities[t]
            for t, period in enumerate(periods)
        ):
            continue

        cost = 0.0
        for risk, choice in zip(risks, choices):
            if choice is None:
                cost += risk.failure_probability_by_period[-1] * failure_cost
            else:
                t = periods.index(choice)
                cost += (
                    preventive_cost
                    + planned_downtime_cost
                    + risk.failure_probability_by_period[t] * failure_cost
                )
        best = min(best, cost)

    return float(best)


def synthetic_cmapss(
    *,
    n_engines: int = 24,
    min_life: int = 90,
    max_life: int = 150,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_engines + 1):
        life = int(rng.integers(min_life, max_life + 1))
        offset = rng.normal(0.0, 0.15)
        for cycle in range(1, life + 1):
            age = cycle / life
            values = [
                unit, cycle,
                rng.normal(0, 0.002),
                rng.normal(0, 0.0003),
                100.0,
            ]
            sensors = []
            for sensor in range(1, 22):
                base = 500 + 20 * sensor + offset
                if f"s{sensor}" in INFORMATIVE_SENSORS:
                    direction = 1 if sensor % 2 == 0 else -1
                    value = (
                        base
                        + direction * 12.0 * age
                        + direction * 8.0 * age**3
                        + rng.normal(0, 0.6)
                    )
                else:
                    value = base + rng.normal(0, 0.01)
                sensors.append(value)
            rows.append(values + sensors)

    return pd.DataFrame(rows, columns=COLUMNS)


def synthetic_train_test_fixture(
    *,
    n_train_engines: int = 30,
    n_test_engines: int = 10,
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = n_train_engines + n_test_engines
    full = synthetic_cmapss(
        n_engines=total,
        min_life=100,
        max_life=160,
        seed=seed,
    )

    train = full[full["unit"] <= n_train_engines].copy()

    rng = np.random.default_rng(seed + 1)
    test_parts = []
    labels = []
    for new_id, original_id in enumerate(
        range(n_train_engines + 1, total + 1),
        start=1,
    ):
        engine = full[full["unit"] == original_id].copy()
        max_cycle = int(engine["cycle"].max())
        remaining = int(rng.integers(12, min(55, max_cycle - 20)))
        cutoff = max_cycle - remaining
        truncated = engine[engine["cycle"] <= cutoff].copy()
        truncated["unit"] = new_id
        test_parts.append(truncated)
        labels.append(float(remaining))

    test = pd.concat(test_parts, ignore_index=True)
    rul = pd.DataFrame({"rul": labels})
    return train.reset_index(drop=True), test, rul


NASA_FD001_RAW_SAMPLE = """\
1 1 -0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 2388 100.00 39.06 23.4190
1 2 0.0019 -0.0003 100.0 518.67 642.15 1591.82 1403.14 14.62 21.61 553.75 2388.04 9044.07 1.30 47.49 522.28 2388.07 8131.49 8.4318 0.03 392 2388 100.00 39.00 23.4236
1 3 -0.0043 0.0003 100.0 518.67 642.35 1587.99 1404.20 14.62 21.61 554.26 2388.08 9052.94 1.30 47.27 522.42 2388.03 8133.23 8.4178 0.03 390 2388 100.00 38.95 23.3442
1 4 0.0007 0.0000 100.0 518.67 642.35 1582.79 1401.87 14.62 21.61 554.45 2388.11 9049.48 1.30 47.13 522.86 2388.08 8133.83 8.3682 0.03 392 2388 100.00 38.88 23.3739
1 5 -0.0019 -0.0002 100.0 518.67 642.37 1582.85 1406.22 14.62 21.61 554.00 2388.06 9055.15 1.30 47.28 522.19 2388.04 8133.80 8.4294 0.03 393 2388 100.00 38.90 23.4044
"""


def self_test() -> None:
    sample = parse_cmapss_text(io.StringIO(NASA_FD001_RAW_SAMPLE))
    assert sample.shape == (5, 26)
    assert sample.iloc[0]["unit"] == 1
    assert sample.iloc[0]["cycle"] == 1
    assert math.isclose(sample.iloc[0]["s2"], 641.82, abs_tol=1e-12)

    synthetic = synthetic_cmapss(n_engines=24, seed=11)
    labeled = add_training_rul(synthetic, rul_cap=125)
    assert labeled.groupby("unit")["rul_raw"].min().eq(0).all()

    features = engineer_features(labeled, window=10)
    first_rows = features.groupby("unit").head(1)
    assert np.allclose(first_rows["s2_std10"], 0.0)
    assert np.allclose(first_rows["s2_trend10"], 0.0)

    train_ids, cal_ids = split_engine_ids(
        synthetic["unit"].unique(),
        calibration_fraction=0.25,
        seed=3,
    )
    assert not (set(train_ids) & set(cal_ids))
    assert set(train_ids) | set(cal_ids) == set(synthetic["unit"].unique())

    train_fx, test_fx, rul_fx = synthetic_train_test_fixture(seed=19)
    fx_model = fit_rul_model(
        train_fx,
        seed=19,
        window=10,
        rul_cap=125,
    )
    assert len(fx_model.calibration_residuals) >= 2
    assert not (
        set(fx_model.train_units)
        & set(fx_model.calibration_units)
    )

    fx_predictions = predict_test_endpoints(
        fx_model,
        test_fx,
        rul_fx,
        window=10,
    )
    fx_metrics = evaluate_predictions(fx_predictions)
    assert len(fx_predictions) == 10
    assert np.isfinite(fx_metrics.rmse)
    assert fx_metrics.rmse < 35.0
    fx_risks = build_engine_risks(
        fx_predictions,
        fx_model,
        periods=(10, 20, 30, 40),
    )
    fx_plan = optimize_maintenance_schedule(
        fx_risks,
        periods=(10, 20, 30, 40),
        capacity_per_period=3,
    )
    assert fx_plan.objective_status == "OPTIMAL"

    tiny_residuals = np.array([-3.0, -1.0, 2.0, 4.0])
    assert math.isclose(
        split_conformal_abs_quantile(tiny_residuals, alpha=0.10),
        4.0,
        abs_tol=1e-12,
    )

    residuals = np.array([-12, -5, 0, 3, 10, 18], dtype=float)
    p10 = failure_probability(30, 10, residuals)
    p20 = failure_probability(30, 20, residuals)
    p40 = failure_probability(30, 40, residuals)
    assert p10 <= p20 <= p40
    assert failure_probability(60, 20, residuals) <= p20

    risks = [
        EngineRisk(1, 8, 5, 11, (0.4, 0.8)),
        EngineRisk(2, 18, 12, 25, (0.1, 0.5)),
        EngineRisk(3, 50, 40, 60, (0.0, 0.1)),
    ]
    periods = (10, 20)
    capacities = (1, 1)
    kwargs = dict(
        preventive_cost=20.0,
        planned_downtime_cost=5.0,
        failure_cost=100.0,
    )
    plan = optimize_maintenance_schedule(
        risks,
        periods=periods,
        capacity_per_period=capacities,
        **kwargs,
    )
    oracle = brute_force_schedule_cost(
        risks,
        periods,
        capacities,
        **kwargs,
    )
    assert plan.objective_status == "OPTIMAL"
    assert math.isclose(plan.expected_cost, oracle, abs_tol=1e-9)

    print("Predictive-maintenance ML + scheduling self-test: OK")


def print_result(result: EndToEndResult) -> None:
    m = result.metrics
    opt = result.optimized_plan
    base = result.baseline_plan

    print("=" * 82)
    print("NASA C-MAPSS FD001: RUL PREDICTION + MAINTENANCE OPTIMIZATION")
    print("=" * 82)
    print(f"Capped-RUL RMSE              : {m.rmse:.3f} cycles")
    print(f"Capped-RUL MAE               : {m.mae:.3f} cycles")
    print(f"90% interval observed cover  : {100*m.interval90_coverage:.1f}%")
    print(f"90% interval mean width      : {m.interval90_mean_width:.3f} cycles")
    print()
    print(f"MILP status                  : {opt.objective_status}")
    print(f"Optimized expected cost      : ${opt.expected_cost:,.2f}")
    print(f"Threshold baseline cost      : ${base.expected_cost:,.2f}")
    print(f"Modeled cost reduction       : ${base.expected_cost-opt.expected_cost:,.2f}")
    print()
    print("Optimized maintenance load by period:")
    for period in opt.periods:
        print(f"  cycle {period:>3}: {opt.period_loads[period]} engines")
    print()
    print(
        "Important: maintenance cost is optimized under the calibrated prediction "
        "error model. It is a decision-support experiment, not a certified "
        "airworthiness or safety policy."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity-per-period", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        result = run_end_to_end(
            args.data_dir,
            seed=args.seed,
            capacity_per_period=args.capacity_per_period,
        )
        print_result(result)
