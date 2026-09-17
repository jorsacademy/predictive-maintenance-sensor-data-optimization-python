# Maintenance and Reliability Optimization Research Series

This file maps maintenance-related repositories across prediction, exact/sequential optimization, reinforcement learning, and scheduling. It is an index only: each repository remains independent.

## Sensor-driven predictive maintenance

- `predictive-maintenance-sensor-data-optimization-python` — NASA C-MAPSS telemetry -> RUL prediction -> conformal uncertainty -> failure risk -> capacity-constrained maintenance MILP.

This is the strongest direct bridge from predictive modeling to prescriptive maintenance decisions in the portfolio.

## Sequential maintenance decisions

- `industrial-maintenance-markov-decision-process-python` — fully observed finite-state maintenance MDP solved exactly by value iteration, policy iteration, and exhaustive policy enumeration.
- `predictive-maintenance-reinforcement-learning` — condition-based maintenance environment with DQN/PPO and interpretable baseline policies.

These should remain separate: one is an exact dynamic-programming benchmark under known transition probabilities; the other studies learned policies in a richer RL environment.

## Maintenance scheduling

- `industrial-maintenance-scheduling-optimizer` — resource-constrained preventive/predictive task scheduling with greedy scheduling, resource calendars, metrics, and an LP relaxation.
- `aircraft-maintenance-scheduling-gurobi` — application-specific exact maintenance scheduling in aviation.

Scheduling is downstream of health/risk estimation and is a different decision layer from replacement/maintenance-policy optimization.

## Related dynamic-control projects

- `offline-rl-industrial-process-control` — offline RL from logged control data; related methodologically but not maintenance-specific.
- `safe-rl-constrained-production-control` — constrained RL with explicit safety/feasibility concerns.
- `dynamic-manufacturing-digital-twin-rl` — closed-loop RL in a digital-twin environment.

## Portfolio rule

Do not consolidate maintenance repositories merely because they share the same industrial domain. The key distinction is the decision layer:

`sensor prediction -> failure-risk estimation -> maintenance policy -> maintenance scheduling`.

A repository merge is justified only when both the state/decision model and the computational experiment are substantially duplicated.
