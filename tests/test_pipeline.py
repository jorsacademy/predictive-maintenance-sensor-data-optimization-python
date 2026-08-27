import io
import math
import unittest

import numpy as np
import pandas as pd

from predictive_maintenance_cmapss_optimization import (
    COLUMNS,
    EXPECTED_FD001,
    EngineRisk,
    NASA_FD001_RAW_SAMPLE,
    add_training_rul,
    brute_force_schedule_cost,
    engineer_features,
    evaluate_predictions,
    failure_probability,
    fit_rul_model,
    optimize_maintenance_schedule,
    parse_cmapss_text,
    predict_test_endpoints,
    split_conformal_abs_quantile,
    split_engine_ids,
    synthetic_cmapss,
    synthetic_train_test_fixture,
    validate_fd001_dimensions,
)


class PredictiveMaintenanceTests(unittest.TestCase):
    def test_real_nasa_raw_sample_parser(self):
        df = parse_cmapss_text(io.StringIO(NASA_FD001_RAW_SAMPLE))
        self.assertEqual(df.shape, (5, 26))
        self.assertEqual(list(df.columns), COLUMNS)
        self.assertEqual(int(df.iloc[0]["unit"]), 1)
        self.assertEqual(int(df.iloc[0]["cycle"]), 1)
        self.assertTrue(math.isclose(
            float(df.iloc[0]["s2"]),
            641.82,
            abs_tol=1e-12,
        ))

    def test_rul_generation_ends_at_zero_and_caps_healthy_region(self):
        df = synthetic_cmapss(
            n_engines=6,
            min_life=140,
            max_life=150,
            seed=3,
        )
        labeled = add_training_rul(df, rul_cap=125.0)
        self.assertTrue(
            labeled.groupby("unit")["rul_raw"].min().eq(0).all()
        )
        self.assertLessEqual(float(labeled["rul"].max()), 125.0)

    def test_rolling_features_do_not_cross_engine_boundaries(self):
        df = synthetic_cmapss(
            n_engines=4,
            min_life=20,
            max_life=25,
            seed=5,
        )
        labeled = add_training_rul(df)
        features = engineer_features(labeled, window=10)
        first = features.groupby("unit").head(1)
        np.testing.assert_allclose(first["s2_std10"], 0.0)
        np.testing.assert_allclose(first["s2_trend10"], 0.0)

    def test_engine_level_split_has_no_leakage(self):
        train, cal = split_engine_ids(
            np.arange(1, 101),
            calibration_fraction=0.30,
            seed=42,
        )
        self.assertFalse(set(train) & set(cal))
        self.assertEqual(set(train) | set(cal), set(range(1, 101)))
        self.assertEqual(len(cal), 30)
        self.assertEqual(len(train), 70)

    def test_split_conformal_radius_uses_finite_sample_correction(self):
        residuals = np.array([-3.0, -1.0, 2.0, 4.0])
        q = split_conformal_abs_quantile(
            residuals,
            alpha=0.10,
        )
        self.assertTrue(math.isclose(q, 4.0, abs_tol=1e-12))

    def test_failure_probability_is_monotone(self):
        residuals = np.array([-12, -5, 0, 3, 10, 18], dtype=float)
        probabilities = [
            failure_probability(30.0, horizon, residuals)
            for horizon in (10, 20, 30, 40)
        ]
        self.assertEqual(probabilities, sorted(probabilities))
        self.assertLessEqual(
            failure_probability(60.0, 20.0, residuals),
            failure_probability(30.0, 20.0, residuals),
        )

    def test_maintenance_milp_matches_bruteforce_oracle(self):
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
        result = optimize_maintenance_schedule(
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
        self.assertEqual(result.objective_status, "OPTIMAL")
        self.assertTrue(math.isclose(
            result.expected_cost,
            oracle,
            abs_tol=1e-9,
        ))

    def test_synthetic_train_test_end_to_end(self):
        train, test, rul = synthetic_train_test_fixture(
            n_train_engines=24,
            n_test_engines=8,
            seed=29,
        )
        model = fit_rul_model(
            train,
            seed=29,
            window=10,
            rul_cap=125,
        )
        pred = predict_test_endpoints(
            model,
            test,
            rul,
            window=10,
        )
        metrics = evaluate_predictions(pred)

        self.assertEqual(len(pred), 8)
        self.assertTrue(np.isfinite(metrics.rmse))
        self.assertLess(metrics.rmse, 40.0)
        self.assertFalse(
            set(model.train_units) & set(model.calibration_units)
        )

    def test_fd001_integrity_guard_rejects_wrong_dimensions(self):
        train = pd.DataFrame(np.zeros((10, 26)), columns=COLUMNS)
        test = pd.DataFrame(np.zeros((10, 26)), columns=COLUMNS)
        train["unit"] = 1
        test["unit"] = 1
        rul = pd.DataFrame({"rul": [1.0]})
        with self.assertRaises(ValueError):
            validate_fd001_dimensions(train, test, rul)


if __name__ == "__main__":
    unittest.main()
