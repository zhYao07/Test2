"""Run with: python -m unittest discover -s Baseline_v16_v1 -p test_validation_split.py"""

import unittest

import numpy as np
import pandas as pd

from validation_split import select_validation, distribution_report


class ValidationSplitTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(17)
        self.labels = ["common", "rare", "missing", "uncertain"]
        self.frame = pd.DataFrame({
            "StudyInstanceUID": [f"study-{i:04}" for i in range(500)],
            "common": np.where(rng.rand(500) < 0.4, 0.95, 0.025),
            "rare": np.where(rng.rand(500) < 0.08, 0.91, 0.25),
            "missing": rng.choice([0.02, 0.7, np.nan], 500),
            "uncertain": rng.choice([0.5, 0.3, 0.85], 500),
        })

    def test_membership_reproducibility_and_quotas(self):
        train, valid = select_validation(self.frame, self.labels, valid_count=50)
        shuffled = self.frame.sample(frac=1, random_state=4)
        train2, valid2 = select_validation(shuffled, self.labels, valid_count=50)
        pd.testing.assert_frame_equal(train, train2)
        pd.testing.assert_frame_equal(valid, valid2)
        self.assertEqual(len(valid), 50)
        self.assertTrue(set(train.StudyInstanceUID).isdisjoint(valid.StudyInstanceUID))
        self.assertEqual(set(train.StudyInstanceUID) | set(valid.StudyInstanceUID), set(self.frame.StudyInstanceUID))
        # Selection must not alter any target, including NaN and 0.5.
        pd.testing.assert_frame_equal(
            pd.concat([train, valid]).sort_values("StudyInstanceUID").reset_index(drop=True), self.frame)
        report = distribution_report(self.frame, valid, self.labels)
        for row in report["features"][:len(self.labels)]:
            self.assertLessEqual(abs(row["valid_count"] - round(row["target_count"])), 1)
        # Report pair counts are calculated using full-data pair names/order.
        for row in report["features"]:
            if row["feature"].startswith("pair:"):
                left, right = row["feature"][5:].split("+")
                self.assertEqual(row["valid_count"], int(((valid[left] > 0.5) & (valid[right] > 0.5)).sum()))

    def test_locked_gold_studies(self):
        locked = self.frame.StudyInstanceUID.iloc[:12].tolist()
        train, valid = select_validation(self.frame, self.labels, valid_count=50, locked_uids=locked)
        self.assertTrue(set(locked) <= set(valid.StudyInstanceUID))
        self.assertTrue(set(locked).isdisjoint(train.StudyInstanceUID))
        self.assertEqual(len(valid), 50)
        _, same = select_validation(self.frame.sample(frac=1, random_state=3), self.labels,
                                    valid_count=50, locked_uids=locked)
        pd.testing.assert_frame_equal(valid, same)
        _, only_locked = select_validation(self.frame, self.labels, valid_count=12, locked_uids=locked)
        self.assertEqual(set(locked), set(only_locked.StudyInstanceUID))
        for count, uids in ((11, locked), (50, ["missing-study"])):
            with self.assertRaises(ValueError):
                select_validation(self.frame, self.labels, valid_count=count, locked_uids=uids)

    def test_invalid_input(self):
        for frame in (self.frame.iloc[:1], pd.concat([self.frame, self.frame.iloc[:1]]),
                      self.frame.assign(StudyInstanceUID=None)):
            with self.assertRaises(ValueError):
                select_validation(frame, self.labels, valid_count=50)

    def test_sparse_constant_labels(self):
        frame = self.frame.assign(common=0.5, rare=0.025, missing=np.nan)
        train, valid = select_validation(frame, self.labels, valid_count=50)
        self.assertEqual((len(train), len(valid)), (450, 50))
        report = distribution_report(frame, valid, self.labels)
        self.assertTrue(all(np.isfinite(row["deviation"]) for row in report["features"]))


if __name__ == "__main__":
    unittest.main()
