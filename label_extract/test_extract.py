"""Offline checks only: no API calls and no generated real-study labels."""

import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("label_extract_module", Path(__file__).with_name("extract.py"))
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)


def response_value(study):
    return {
        "StudyInstanceUID": study["StudyInstanceUID"], "report_scope": "KNEE",
        "labels": {target: {"report_status": "NOT_MENTIONED", "label": None,
                            "evidence_strength": "INSUFFICIENT", "score_reason": "无报告证据",
                            "evidence": [], "uncertainty_reason": "未提及"}
                   for target in extract.TARGETS},
    }


class FakeResponse:
    def __init__(self, value, tokens=True, status="completed"):
        self.output_text = json.dumps(value)
        self.raw = {"id": "fake-offline", "model": "fake-model", "status": status,
                    "output_text": self.output_text}
        if tokens:
            self.raw["usage"] = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                                 "input_tokens_details": {"cached_tokens": 20},
                                 "output_tokens_details": {"reasoning_tokens": 10}}

    def model_dump(self, mode):
        return copy.deepcopy(self.raw)


class FakeClient:
    def __init__(self, factory):
        self.factory = factory
        self.requests = []
        self.responses = self

    def create(self, **request):
        self.requests.append(request)
        return self.factory(request)

    def close(self):
        pass


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.study = {"StudyInstanceUID": "test-study", "Report": "Knee MRI. Complete ACL tear. No effusion."}
        self.args = extract.parser().parse_args(["--retries", "1"])

    def test_unknown_is_not_negative(self):
        value, warnings = extract.validate_output(response_value(self.study), self.study, extract.response_schema())
        self.assertEqual(warnings, [])
        self.assertTrue(all(item["label"] is None for item in value["labels"].values()))

    def test_positive_evidence_and_below_threshold(self):
        value = response_value(self.study)
        value["labels"]["ACL"] = {"report_status": "PRESENT", "label": 0.94,
                                    "evidence_strength": "DIRECT", "score_reason": "明确完全撕裂",
                                    "evidence": ["Complete ACL tear."], "uncertainty_reason": ""}
        value["labels"]["Effusion"] = {"report_status": "ABSENT", "label": 0.07,
                                         "evidence_strength": "DIRECT", "score_reason": "明确否定",
                                         "evidence": ["No effusion."], "uncertainty_reason": ""}
        checked, warnings = extract.validate_output(value, self.study, extract.response_schema())
        self.assertEqual(warnings, [])
        self.assertEqual(checked["labels"]["ACL"]["label"], 0.94)
        self.assertEqual(checked["labels"]["Effusion"]["label"], 0.07)

    def test_fabricated_quote_and_state_conflict_are_masked(self):
        value = response_value(self.study)
        value["labels"]["ACL"] = {"report_status": "PRESENT", "label": 0.94,
                                    "evidence_strength": "DIRECT", "score_reason": "假证据",
                                    "evidence": ["Fabricated quote"], "uncertainty_reason": ""}
        value["labels"]["Effusion"] = {"report_status": "ABSENT", "label": 0.93,
                                         "evidence_strength": "DIRECT", "score_reason": "状态冲突",
                                         "evidence": ["No effusion."], "uncertainty_reason": ""}
        checked, warnings = extract.validate_output(value, self.study, extract.response_schema())
        self.assertEqual(len(warnings), 2)
        self.assertIsNone(checked["labels"]["ACL"]["label"])
        self.assertIsNone(checked["labels"]["Effusion"]["label"])

    def test_wrong_uid_and_missing_target_rejected(self):
        for mutation in ("uid", "target"):
            value = response_value(self.study)
            if mutation == "uid":
                value["StudyInstanceUID"] = "wrong"
            else:
                del value["labels"]["ACL"]
            with self.assertRaises(Exception):
                extract.validate_output(value, self.study, extract.response_schema())

    def test_non_knee_scope_masks_all(self):
        value = response_value(self.study)
        value["report_scope"] = "OTHER"
        checked, warnings = extract.validate_output(value, self.study, extract.response_schema())
        self.assertEqual(len(warnings), 12)
        self.assertTrue(all(item["label"] is None for item in checked["labels"].values()))

    def test_selection_excludes_gold_duplicates_and_label_columns(self):
        gold = {"StudyInstanceUID": "gold", "Report": "Knee MRI. Tear.",
                **{target: "1" for target in extract.TARGETS}}
        duplicate = {**gold, "StudyInstanceUID": "duplicate", "Report": "KNEE  MRI. tear.",
                     **{target: "" for target in extract.TARGETS}}
        ordinary = {**duplicate, "StudyInstanceUID": "ordinary", "Report": "Knee MRI normal."}
        selected = extract.select_studies([gold, duplicate, ordinary], 1, 42)
        self.assertEqual(selected, [{"StudyInstanceUID": "ordinary", "Report": "Knee MRI normal."}])
        self.assertEqual(extract.select_examples([gold], 1, 42)[0]["image_reference_labels"]["ACL"], 1)

    def test_retry_usage_includes_failed_response_and_no_sdk_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            count = [0]

            def factory(request):
                count[0] += 1
                # First response consumed tokens but returned the wrong UID.
                value = response_value(self.study)
                if count[0] == 1:
                    value["StudyInstanceUID"] = "wrong"
                return FakeResponse(value)

            client = FakeClient(factory)
            with patch.object(extract.time, "sleep"):
                saved = extract.call_study(client, self.study, "prompt", extract.response_schema(), self.args, output)
            self.assertEqual(saved["status"], "SUCCESS")
            self.assertEqual(len(client.requests), 2)
            self.assertEqual(client.requests[0]["text"]["format"]["type"], "json_schema")
            summary = extract.export_results(output, [self.study])
            self.assertEqual(summary["usage_totals_observed"]["total_tokens"], 300)
            self.assertEqual(summary["usage_totals_observed"]["cached_input_tokens"], 40)
            self.assertEqual(summary["masked_labels"], 12)
            self.assertEqual(len(list((output / "raw").glob("*.json"))), 2)
            with (output / "labels.csv").open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["ACL"], "")
            self.assertEqual(row["ACL__mask"], "0")

    def test_missing_usage_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            client = FakeClient(lambda request: FakeResponse(response_value(self.study), tokens=False))
            extract.call_study(client, self.study, "prompt", extract.response_schema(), self.args, output)
            summary = extract.export_results(output, [self.study])
            self.assertIsNone(summary["usage_totals_observed"]["total_tokens"])
            self.assertEqual(summary["attempts_without_total_token_usage"], 1)

    def test_incomplete_response_never_generates_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            client = FakeClient(lambda request: FakeResponse(response_value(self.study), status="incomplete"))
            self.args.retries = 0
            saved = extract.call_study(client, self.study, "prompt", extract.response_schema(), self.args, output)
            self.assertEqual(saved["status"], "ERROR")
            self.assertFalse((output / "results.jsonl").exists())
            summary = extract.export_results(output, [self.study])
            self.assertEqual(summary["successful_studies"], 0)
            self.assertEqual(summary["usage_totals_observed"]["total_tokens"], 150)

    def test_main_resume_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, output = root / "train.csv", root / "output"
            rows = [
                {"StudyInstanceUID": "gold", "Report": "Knee MRI. Gold example.",
                 **{target: "1" for target in extract.TARGETS}},
                {"StudyInstanceUID": self.study["StudyInstanceUID"], "Report": self.study["Report"],
                 **{target: "" for target in extract.TARGETS}},
            ]
            extract.write_csv(input_path, ["StudyInstanceUID", "Report", *extract.TARGETS], rows)
            client = FakeClient(lambda request: FakeResponse(response_value(self.study)))
            fake_module = types.SimpleNamespace(OpenAI=lambda **kwargs: client)
            argv = ["--input", str(input_path), "--output", str(output), "--limit", "1", "--examples", "1"]
            with patch.dict(sys.modules, {"openai": fake_module}), patch.dict(os.environ, {"generate_label": "fake-key"}):
                self.assertEqual(extract.main(argv + ["--prepare-only"]), 0)
                self.assertEqual(len(client.requests), 0)
                self.assertEqual(extract.main(argv), 0)
                self.assertEqual(extract.main(argv + ["--workers", "3"]), 0)
                self.assertEqual(len(client.requests), 1)
                with self.assertRaises(ValueError):
                    extract.main(argv + ["--model", "different-model"])
            for path in output.rglob("*"):
                if path.is_file():
                    self.assertNotIn(b"\r\n", path.read_bytes())

    def test_parallel_calls_and_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            barrier = threading.Barrier(3)
            counter_lock = threading.Lock()
            active, peak = [0], [0]
            studies = [{**self.study, "StudyInstanceUID": f"parallel-{index}"} for index in range(3)]

            def factory(request):
                study = json.loads(request["input"])
                with counter_lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                barrier.wait(timeout=5)
                with counter_lock:
                    active[0] -= 1
                return FakeResponse(response_value(study))

            self.args.workers = 3
            code = extract.run_batch(lambda: FakeClient(factory), studies, "prompt", extract.response_schema(),
                                     self.args, output, set(), extract.time.perf_counter())
            self.assertEqual(code, 0)
            self.assertEqual(peak[0], 3)
            self.assertEqual(len(extract.read_jsonl(output / "attempts.jsonl")), 3)
            self.assertEqual(len(extract.read_jsonl(output / "results.jsonl")), 3)
            self.assertEqual(extract.export_results(output, studies)["usage_totals_observed"]["total_tokens"], 450)

    def test_parallel_failure_stops_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            barrier = threading.Barrier(2)
            self.args.workers, self.args.retries = 2, 0
            studies = [{**self.study, "StudyInstanceUID": f"failure-{index}"} for index in range(6)]

            def factory(request):
                study = json.loads(request["input"])
                barrier.wait(timeout=5)
                return FakeResponse(response_value(study), status="incomplete")

            code = extract.run_batch(lambda: FakeClient(factory), studies, "prompt", extract.response_schema(),
                                     self.args, output, set(), extract.time.perf_counter())
            self.assertEqual(code, 1)
            self.assertEqual(len(extract.read_jsonl(output / "attempts.jsonl")), 2)

    def test_legacy_manifest_checks_actual_saved_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config, prompt, schema, examples = {"model": "fake"}, "prompt", {}, []
            extract.write_json(output / "manifest.json", {"config": config, "fingerprint": "legacy"})
            extract.write_text(output / "prompt.txt", prompt + "\n")
            extract.write_json(output / "schema.json", schema)
            extract.write_json(output / "examples.json", examples)
            extract.write_csv(output / "selected_studies.csv", ["StudyInstanceUID", "Report"], [self.study])
            with self.assertRaises(ValueError):
                extract.check_manifest(output, config, prompt, schema, examples, [self.study], "new")
            with self.assertRaises(ValueError):
                extract.check_manifest(output, config, "changed prompt", schema, examples, [self.study], "new")

    def test_uncertain_score_retained_with_separate_weight(self):
        study = {"StudyInstanceUID": "soft-study", "Report": "Possible ACL tear."}
        value = response_value(study)
        value["labels"]["ACL"] = {"report_status": "UNCERTAIN", "label": 0.63,
                                    "evidence_strength": "PARTIAL", "score_reason": "疑似撕裂",
                                    "evidence": ["Possible ACL tear."], "uncertainty_reason": "程度未给出"}
        checked, warnings = extract.validate_output(value, study, extract.response_schema())
        self.assertFalse(warnings)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            client = FakeClient(lambda request: FakeResponse(checked))
            extract.call_study(client, study, "prompt", extract.response_schema(), self.args, output)
            summary = extract.export_results(output, [study])
            with (output / "labels.csv").open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["ACL"], "0.63")
            self.assertEqual(row["ACL__mask"], "1")
            self.assertEqual(row["ACL__weight"], "0.35")
            self.assertEqual(summary["soft_score_distribution"]["ACL"]["mean"], 0.63)

    def test_soft_score_endpoints_and_invalid_numbers_rejected(self):
        for score in [0, 1, -0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(score=score):
                value = response_value(self.study)
                value["labels"]["ACL"].update(label=score, report_status="PRESENT",
                                              evidence=["Complete ACL tear."], evidence_strength="DIRECT")
                with self.assertRaises(Exception):
                    extract.validate_output(value, self.study, extract.response_schema())

    def test_unmentioned_score_never_used_as_supervision(self):
        value = response_value(self.study)
        value["labels"]["PF OA"].update(label=0.5, evidence_strength="PARTIAL")
        checked, warnings = extract.validate_output(value, self.study, extract.response_schema())
        self.assertIsNone(checked["labels"]["PF OA"]["label"])
        self.assertEqual(checked["labels"]["PF OA"]["evidence_strength"], "INSUFFICIENT")
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
