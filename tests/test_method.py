import unittest
from types import SimpleNamespace
import numpy as np
from ibpr_bpgs.features import AFeatureContext, relationship_39d
from ibpr_bpgs.regions import voxel_components, region_data, apply
from ibpr_bpgs.suggestions import select_best_competitor, filter_suggestions
from ibpr_bpgs.metrics import propagate, lookup_for, gain, _measure, pooled_ap
from ibpr_bpgs.targets import compact_cache, marginal_targets, fast_gain
from ibpr_bpgs.pipeline import predict
from ibpr_bpgs.training import fit_a, fit_b


def scene():
    n = 12
    pos = np.column_stack([np.arange(n) * 0.03, np.zeros(n), np.zeros(n)]).astype(
        np.float32
    )
    return SimpleNamespace(
        pos=pos,
        pred=np.array([1] * 6 + [2] * 6, np.int32),
        predicted_tree=np.ones(n, bool),
        offset=np.zeros((n, 3), np.float32),
        embedding=np.zeros((n, 5), np.float32),
        tree_probability=np.ones(n, np.float32) * 0.8,
        prediction_count=np.ones(n, np.float32),
        boundary_score=np.ones(n, np.float32) * 0.4,
    )


class MethodTests(unittest.TestCase):
    def test_39_dimensions_coverage_removed(self):
        s = scene()
        c = AFeatureContext(s)
        ids = np.arange(12)
        cand, ev = c.proposals(ids)
        x = relationship_39d(c.features_13d(ids, cand, ev))
        self.assertEqual(x.shape, (12, 4, 39))
        s.prediction_count[:] = 1000
        y = relationship_39d(c.features_13d(ids, cand, ev))
        np.testing.assert_array_equal(x, y)

    def test_vote_tie_background(self):
        p = np.array([2, 2, 1, 1, 3, 0, 0])
        nn = np.array([[0, 1, 2, 3, 4], [0, 2, 4, 5, 6]])
        np.testing.assert_array_equal(propagate(p, nn), [1, 0])

    def test_region_pair_isolation(self):
        xyz = np.array([[0, 0, 0], [0.24, 0, 0], [0.25, 0, 0], [0.26, 0, 0], [3, 0, 0]])
        labels = voxel_components(
            xyz, np.array([1, 1, 1, 2, 1]), np.array([2, 2, 2, 1, 2]), 0.25
        )
        self.assertEqual(labels[0], labels[2])
        self.assertNotEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[4])

    def test_threshold_strict_margin(self):
        p = {
            "valid": np.array([True, True]),
            "target_score": np.array([0.4, 0.4]),
            "score_advantage": np.array([0.02, 0.020001]),
        }
        np.testing.assert_array_equal(filter_suggestions(p)["pool"], [False, True])

    def test_prf_strict_half_and_ap_inclusive(self):
        pred = np.array([1, 1, 1, 1])
        gt = np.array([1, 1, 0, 0])
        m, e = _measure(pred, gt, np.ones(4), pred)
        self.assertEqual((m["TP"], m["FP"], m["FN"]), (0, 1, 1))
        self.assertEqual(pooled_ap([e], 0.5), 1.0)

    def test_exact_marginal_after_vote(self):
        s = scene()
        scores = {
            "point_ids": np.array([4, 5, 6, 7]),
            "candidates": np.array(
                [[1, 2, 0, 0], [1, 2, 0, 0], [2, 1, 0, 0], [2, 1, 0, 0]]
            ),
            "scores": np.array([[0.1, 0.9, -1, -1]] * 4, np.float32),
        }
        d = region_data(s, scores, {"score": 0.4, "margin": 0.02})
        nn = np.array(
            [[0, 1, 4, 5, 6], [4, 5, 6, 7, 8], [3, 4, 5, 6, 7], [7, 8, 9, 10, 11]]
        )
        gt = np.array([1, 2, 1, 2])
        base = propagate(s.pred, nn)
        cache = compact_cache(s, gt, nn, base, d)
        allon = np.ones(len(d["components"]), bool)
        full = apply(s, d, allon)
        mapped = lookup_for(gt, base)
        fullg = gain(gt, base, propagate(full, nn), mapped)["G"]
        targets = marginal_targets(cache, len(allon))
        for j in range(len(allon)):
            keep = allon.copy()
            keep[j] = False
            pred = propagate(apply(s, d, keep), nn)
            self.assertEqual(targets[j], fullg - gain(gt, base, pred, mapped)["G"])
            self.assertEqual(
                fast_gain(cache, keep)["G"], gain(gt, base, pred, mapped)["G"]
            )

    def test_empty_prediction(self):
        s = scene()
        s.pred[:] = 0
        s.predicted_tree[:] = False

        class A:
            n_features_in_ = 39

        class B:
            n_features_in_ = 38

        _, ep, _, _, u, accepted = predict(
            s,
            None,
            A(),
            B(),
            {"A": {"score": 0.4, "margin": 0.02}, "B": {"threshold": 0}},
        )
        self.assertEqual(len(u), 0)
        np.testing.assert_array_equal(ep["AB"], s.pred)

    def test_training_inference_smoke(self):
        rng = np.random.default_rng(123)
        x = rng.normal(size=(100, 39))
        y = (x[:, 0] > 0).astype(int)
        a = fit_a(
            x,
            y,
            {
                "max_iter": 3,
                "min_samples_leaf": 5,
                "early_stopping": False,
                "random_state": 1,
            },
        )
        x = rng.normal(size=(100, 38))
        b = fit_b(x, x[:, 0], np.ones(100), {"n_estimators": 3, "random_state": 1}, 0)
        s = scene()
        source, ep, _, _, _, _ = predict(
            s, None, a, b, {"A": {"score": 0.4, "margin": 0.02}, "B": {"threshold": 0}}
        )
        self.assertEqual(set(source), {"baseline", "A", "AB"})
        self.assertEqual(ep["AB"].shape, (12,))


if __name__ == "__main__":
    unittest.main()
