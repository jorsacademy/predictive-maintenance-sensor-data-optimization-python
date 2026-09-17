# Repository Overlap Audit — Maintenance and Reliability

This document records portfolio overlap without merging, archiving, renaming, or deleting repositories.

## `predictive-maintenance-sensor-data-optimization-python`

**Keep separate.**

Role: end-to-end predictive-maintenance decision support from multivariate degradation telemetry to a maintenance-capacity MILP.

Distinctive value: real benchmark data pipeline, RUL prediction, conformal calibration, failure-risk construction, and downstream optimization.

## `industrial-maintenance-markov-decision-process-python`

**Keep separate.**

Role: exact infinite-horizon discounted MDP for maintenance under known transition probabilities.

Distinctive value: value iteration, policy iteration, exhaustive stationary-policy enumeration, stationary-distribution diagnostics, and Monte Carlo verification.

## `predictive-maintenance-reinforcement-learning`

**Overlap but justified.**

Role: learned condition-based maintenance policies with DQN/PPO in a richer Gymnasium-style environment.

Distinctive value: explicit RL training/evaluation and comparison against run-to-failure, age-based, and condition-based baselines.

It overlaps conceptually with the maintenance MDP, but the research question differs: exact planning under a known small Markov model versus learned policies in a richer stochastic environment.

## `industrial-maintenance-scheduling-optimizer`

**Keep separate.**

Role: resource-constrained maintenance task scheduling.

Distinctive value: assets/tasks/resources, time windows, resource calendars, priority scoring, greedy scheduling, reporting, and an explicitly labeled LP relaxation.

Scheduling tasks is a different decision layer from choosing when to maintain/replace one degrading asset.

## Audit conclusion

There is no current consolidation candidate in the maintenance family. The repositories cover a useful progression:

`degradation data -> RUL/risk prediction -> sequential maintenance policy -> resource-constrained maintenance scheduling`.

The strongest overlap is between the exact maintenance MDP and the RL maintenance environment, but keeping both creates a valuable exact-DP-versus-RL comparison path rather than redundant code.
