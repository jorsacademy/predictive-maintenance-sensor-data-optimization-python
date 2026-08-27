# Predictive Maintenance from NASA C-MAPSS Sensor Data to Maintenance Optimization

An end-to-end predictive-maintenance decision-support project that connects multivariate degradation telemetry to a capacity-constrained maintenance scheduling MILP.

The project uses the NASA C-MAPSS Turbofan Engine Degradation Simulation dataset, specifically FD001.

## Why this project is different

This is not only an RUL-prediction notebook.

```text
NASA C-MAPSS sensor trajectories
          |
          v
causal rolling feature engineering
          |
          v
engine-level train / calibration split
          |
          v
ExtraTrees RUL regression
          |
          v
split-conformal uncertainty calibration
          |
          v
failure probability by planning horizon
          |
          v
maintenance-capacity MILP
          |
          v
optimized engine / maintenance-period assignments
```

The prediction layer and decision layer are validated separately.

## Data source

NASA's Prognostics Center of Excellence describes C-MAPSS as engine-degradation simulation generated with the Commercial Modular Aero-Propulsion System Simulation.

Official sources:

- https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

FD001 contains:

```text
100 train engine trajectories
100 test engine trajectories
1 operating condition
1 HPC degradation fault mode
20,631 training rows
13,096 test rows
```

Every raw row contains an engine id, cycle, 3 operating settings, and 21 sensor measurements. Training trajectories run to failure. Test trajectories stop before failure and NASA supplies the remaining useful life after the final observed cycle.

C-MAPSS is high-fidelity **simulated** degradation telemetry; this repository does not describe it as field-recorded flight sensor data.

Raw NASA data are not committed to this repository. The downloader first attempts NASA's distribution endpoint. If that endpoint is unavailable, it falls back to a public GitHub mirror of the same text files and applies strict FD001 row-count, engine-count, column-count, and missing-value integrity checks.

## Remaining Useful Life target

For a training engine:

```text
raw RUL = engine final cycle - current cycle
RUL = min(raw RUL, 125)
```

Primary regression metrics therefore use the capped-RUL target. Raw NASA test RUL is retained separately.

## Feature engineering

For 14 informative FD001 sensor channels the code builds causal features using observations available only up to the current cycle:

- current sensor value;
- rolling mean;
- rolling standard deviation;
- rolling end-to-end trend proxy;
- current cycle.

The default rolling window is 20 cycles. All rolling operations are grouped by engine, so features cannot spill across engine boundaries.

## Leakage control

A random row-level split would leak the same engine trajectory into both model fitting and uncertainty calibration. Instead, the 100 training engines are split by engine id:

```text
70 engines -> model fitting
30 engines -> uncertainty calibration
```

The official NASA test engines remain untouched until final evaluation.

## RUL model

The default predictor is an `ExtraTreesRegressor`. It is deliberately classical rather than a deep neural network: fast enough for reproducible CI, nonlinear, and suitable for the engineered sensor interactions.

The repository does not claim that ExtraTrees is state-of-the-art on C-MAPSS.

## Uncertainty calibration

One health-state snapshot is selected per held-out calibration engine, spanning different RUL levels. Prediction residuals are used for a split-conformal absolute-error interval.

For nominal 90% intervals:

```text
k = ceil((n_calibration + 1) * 0.90)
```

Observed test-set coverage is reported separately from the nominal target.

## From RUL to failure risk

For a predicted RUL `r_hat` and future horizon `h`, the empirical failure probability is computed from held-out calibration residuals:

```text
P(RUL <= h)
≈ fraction of calibration residuals e
  for which r_hat + e <= h
```

The resulting probability is monotonically nondecreasing with the planning horizon.

## Maintenance optimization

The decision model considers maintenance opportunities at 10, 20, 30, 40, and 50 future operating cycles.

For every engine, the MILP chooses exactly one option:

```text
maintain in one planning period
or
no maintenance during the horizon
```

Each period has limited maintenance-shop capacity.

Modeled maintenance-period cost:

```text
preventive maintenance cost
+ planned downtime cost
+ P(failure before maintenance) * failure cost
```

No-maintenance cost:

```text
P(failure before final horizon) * failure cost
```

The economic coefficients are educational scenario assumptions, not airline maintenance accounting data.

## Baseline

The MILP is compared with a simple threshold rule:

```text
if predicted RUL <= 50:
    schedule by lowest predicted RUL
    into earliest available maintenance slots
else:
    no planned maintenance
```

Both policies use the same calibrated risk estimates and maintenance-capacity assumptions.

## Exact optimization validation

For a small three-engine/two-period instance, the test suite enumerates every feasible maintenance assignment. The SciPy/HiGHS MILP objective must exactly equal this brute-force oracle.

Therefore the scheduling solver is exact for the stated finite MILP; the upstream RUL/risk model remains statistical and uncertain.

## Validated NASA FD001 result

The full pipeline was executed on GitHub Actions with Python 3.12 against the validated FD001 dimensions. With seed 42 and maintenance capacity 8 engines per period, the run produced:

```text
Capped-RUL RMSE              : 18.180 cycles
Capped-RUL MAE               : 13.005 cycles
90% interval observed cover  : 91.0%
90% interval mean width      : 52.780 cycles

MILP status                  : OPTIMAL
Optimized expected cost      : $5,176,666.67
Threshold baseline cost      : $6,121,666.67
Modeled cost reduction       : $945,000.00

Optimized maintenance load
cycle 10 : 8 engines
cycle 20 : 8 engines
cycle 30 : 8 engines
cycle 40 : 2 engines
cycle 50 : 0 engines
```

These values are reproducible results for this exact model, random seed, calibration split, cost assumptions, and FD001 dataset. The `$945,000` difference is a **modeled expected-cost reduction**, not an observed real-world saving.

## Local validation without network access

The repository includes:

1. five verbatim rows from the public FD001 raw format to test the real 26-column parser;
2. a synthetic C-MAPSS-shaped run-to-failure fixture for offline ML regression tests;
3. a truncated test-fleet fixture with held-out RUL labels;
4. the maintenance brute-force oracle.

The synthetic fixture is explicitly a test fixture and is never presented as NASA data.

## Run

Full NASA FD001 pipeline:

```bash
python predictive_maintenance_cmapss_optimization.py --data-dir data/raw --seed 42
```

Offline self-test:

```bash
python predictive_maintenance_cmapss_optimization.py --self-test
```

Regression suite:

```bash
python -m unittest discover -s tests -v
```

## GitHub Actions

CI performs dependency installation, offline self-tests, the regression suite, real FD001 download/integrity validation, model fitting, official test-set RUL evaluation, calibrated risk construction, and the full 100-engine maintenance MILP.

## Scope and limitations

This repository is an educational predictive-maintenance decision-support model. It is not suitable for airworthiness, dispatch, or safety-critical maintenance decisions.

Important limitations:

- C-MAPSS is simulator-generated, not field telemetry;
- FD001 contains one operating condition and one fault mode;
- the RUL target is capped at 125 cycles;
- residual-based risk assumes calibration errors are relevant to future test engines;
- failure probabilities are empirical and coarse with only 30 calibration engines;
- maintenance costs are illustrative;
- maintenance restores the engine for the modeled planning horizon;
- fleet interactions beyond maintenance-shop capacity are not modeled.

No claim of an optimal maintenance policy under true physical failure dynamics is made.
