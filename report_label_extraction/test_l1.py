"""Offline safety/integration tests with synthetic reports, never model validation."""

import copy
import csv
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jsonschema

import l1


CONFIG = {"base_url": "http://localhost:8000/v1", "model": "synthetic-test-only",
          "model_version": "test-revision", "auth_required": False,
          "context_window_tokens": 131072, "max_output_tokens": 10000}


def label(cat, status="NOT_MENTIONED", quote=""):
    return {"report_status": status, "evidence": [quote] if quote else [],
            "anatomy": {"code": "UNKNOWN" if status == "NOT_MENTIONED" else l1.ANATOMY[cat], "text": ""},
            "severity": {"grade": "UNSPECIFIED", "text": ""}, "extent": "",
            "temporal_status": "UNSPECIFIED", "contradiction": False,
            "facts": {name: {"value": None, "evidence": []} for name in l1.FACTS[cat]},
            "llm_meets_criteria": "UNKNOWN", "soft_score": None, "review_reason": []}


def fact(item, name, value, quote):
    item["facts"][name] = {"value": value, "evidence": [quote]}


def response(report):
    obj = {"report_hash": l1.digest(report), "report_scope": {"code": "KNEE", "evidence": [report]},
           "labels": {c: label(c) for c in l1.CATEGORIES}}
    if "ACL and MCL intact." in report:
        for cat in ["ACL", "MCL"]:
            item = label(cat, "ABSENT", "ACL and MCL intact.")
            fact(item, "class_absence_supported", True, "ACL and MCL intact.")
            item["llm_meets_criteria"] = "NO"
            obj["labels"][cat] = item
    if "Conclusion: no fracture." in report:
        item = label("Fracture", "ABSENT", "no fracture.")
        fact(item, "class_absence_supported", True, "no fracture.")
        item["llm_meets_criteria"] = "NO"
        obj["labels"]["Fracture"] = item
    return obj


class ClinicalMappingTests(unittest.TestCase):
    def test_absent_vs_unmentioned(self):
        report = "No knee effusion."
        absent = label("Effusion", "ABSENT", report)
        fact(absent, "class_absence_supported", True, report)
        absent["llm_meets_criteria"] = "NO"
        result = l1.audit_label("Effusion", absent, report)
        self.assertEqual((result["target_score"], result["weight"], result["mask"]), (.05, 1., 1))
        unknown = l1.audit_label("Synovitis", label("Synovitis"), report)
        self.assertEqual((unknown["target_score"], unknown["weight"], unknown["mask"]), (None, 0., 0))

    def test_limited_negatives_never_become_confident_or_soft_no(self):
        for cat, report in [("Medial OA", "No focal full-thickness cartilage defects."),
                            ("ACL", "No complete ACL tear."),
                            ("Fracture", "No displaced fracture.")]:
            item = label(cat, "ABSENT", report)
            item["soft_score"] = .01
            item["llm_meets_criteria"] = "NO"
            result = l1.audit_label(cat, item, report)
            self.assertEqual((result["meets_criteria"], result["target_score"], result["weight"], result["mask"]),
                             ("UNKNOWN", None, 0., 0))
            self.assertIn("negative_scope_insufficient", result["review_reason"])

    def test_absence_fact_cannot_coexist_with_present_status(self):
        report = "ACL tear."
        item = label("ACL", "PRESENT", report)
        fact(item, "class_absence_supported", True, report)
        self.assertEqual(l1.audit_label("ACL", item, report)["mask"], 0)

    def test_mild_effusion_is_present_but_subthreshold(self):
        report = "Small knee joint effusion."
        item = label("Effusion", "PRESENT", report)
        item["severity"] = {"grade": "MILD", "text": "Small"}
        result = l1.audit_label("Effusion", item, report)
        self.assertEqual(item["report_status"], "PRESENT")
        self.assertEqual((result["meets_criteria"], result["target_score"], result["weight"]), ("NO", .05, 1.))

    def test_fluid_needs_distension_and_baker_needs_size(self):
        report = "Moderate knee effusion. Baker cyst."
        item = label("Effusion", "PRESENT", "Moderate knee effusion.")
        item["severity"] = {"grade": "MODERATE", "text": "Moderate"}
        self.assertEqual(l1.map_criteria("Effusion", item)[0], "UNKNOWN")
        fact(item, "joint_distension", True, "Moderate knee effusion.")
        self.assertEqual(l1.map_criteria("Effusion", item)[0], "YES")
        cyst = label("Baker's", "PRESENT", "Baker cyst.")
        fact(cyst, "typical_baker_location", True, "Baker cyst.")
        self.assertEqual(l1.map_criteria("Baker's", cyst)[0], "UNKNOWN")

    def test_oa_requires_depth_extent_and_compartment(self):
        report = "Medial compartment full-thickness cartilage loss over 12 mm."
        item = label("Medial OA", "PRESENT", report)
        fact(item, "cartilage_loss_gt50pct", True, "full-thickness cartilage loss")
        item["soft_score"] = .7
        result = l1.audit_label("Medial OA", item, report)
        self.assertEqual((result["meets_criteria"], result["weight"], result["mask"]), ("UNKNOWN", .25, 1))
        fact(item, "cartilage_extent_ge10mm", True, "over 12 mm")
        self.assertEqual(l1.map_criteria("Medial OA", item)[0], "YES")
        item["anatomy"]["code"] = "LATERAL_TIBIOFEMORAL"
        self.assertEqual(l1.audit_label("Medial OA", item, report)["mask"], 0)

    def test_high_grade_mcl_requires_acute(self):
        report = "High-grade MCL tear, age indeterminate."
        item = label("MCL", "PRESENT", report)
        fact(item, "high_grade_tear", True, "High-grade MCL tear")
        self.assertEqual(l1.map_criteria("MCL", item)[0], "UNKNOWN")
        fact(item, "acute_injury", True, "High-grade MCL tear")
        self.assertEqual(l1.map_criteria("MCL", item)[0], "YES")

    def test_meniscal_diagnosis_accepted_without_inventing_slice_evidence(self):
        report = "Rotura del menisco medial."
        item = label("Medial Meniscus", "PRESENT", report)
        fact(item, "definite_meniscal_tear", True, report)
        self.assertEqual(l1.audit_label("Medial Meniscus", item, report)["meets_criteria"], "YES")

    def test_history_only_even_negative_is_not_current_supervision(self):
        report = "History: no fracture last year."
        for status in ["PRESENT", "ABSENT"]:
            item = label("Fracture", status, report)
            item["temporal_status"] = "HISTORY_ONLY"
            item["soft_score"] = .1
            self.assertEqual(l1.audit_label("Fracture", item, report)["mask"], 0)

    def test_clinical_question_not_a_finding(self):
        report = "Clinical history: acute ACL tear? Findings: intact ligaments."
        item = label("ACL", "PRESENT", "acute ACL tear?")
        fact(item, "high_grade_tear", True, "acute ACL tear?")
        result = l1.audit_label("ACL", item, report)
        self.assertEqual(result["mask"], 0)
        self.assertIn("evidence_in_history_or_technique_section", result["review_reason"])

    def test_contradictions_always_masked(self):
        report = "ACL torn. ACL intact."
        item = label("ACL", "PRESENT", "ACL torn.")
        item["evidence"].append("ACL intact.")
        item["contradiction"] = True
        item["soft_score"] = .99
        fact(item, "high_grade_tear", True, "ACL torn.")
        self.assertEqual(l1.audit_label("ACL", item, report)["mask"], 0)

    def test_synovitis_not_inferred_from_effusion(self):
        report = "Large effusion and Hoffa fat pad edema."
        item = label("Synovitis", "PRESENT", report)
        item["llm_meets_criteria"] = "YES"
        self.assertEqual(l1.map_criteria("Synovitis", item)[0], "UNKNOWN")

    def test_contusion_not_every_marrow_edema(self):
        report = "Degenerative subchondral marrow edema."
        item = label("Contusion", "PRESENT", report)
        fact(item, "degenerative_only", True, report)
        self.assertEqual(l1.map_criteria("Contusion", item)[0], "NO")
        other = label("Contusion", "PRESENT", report)
        self.assertEqual(l1.map_criteria("Contusion", other)[0], "UNKNOWN")

    def test_uncertain_not_forced_negative_and_confidence_does_not_raise_weight(self):
        report = "Possible medial meniscus tear."
        item = label("Medial Meniscus", "UNCERTAIN", report)
        item["soft_score"] = .99
        item["llm_meets_criteria"] = "YES"
        result = l1.audit_label("Medial Meniscus", item, report)
        self.assertEqual((result["target_score"], result["weight"], result["mask"]), (.99, .25, 1))


class ValidationTests(unittest.TestCase):
    def test_codex_backend_requires_authorization_without_api_key(self):
        with tempfile.TemporaryDirectory(prefix="rsna_codex_config_") as temp:
            path = Path(temp) / "config.json"
            config = {"backend": "codex_cli", "model": "gpt-6.1-sol", "model_version": "test",
                      "context_window_tokens": 1050000, "allow_agent_execution": False}
            l1.write_json(path, config)
            with patch("l1.shutil.which", return_value="codex.exe"):
                _, state = l1.preflight(path)
            self.assertEqual(state["missing"], ["authorization:allow_agent_execution"])
            with self.assertRaises(l1.FatalEndpointError):
                l1.request_codex(config, "Report", "hash")

    def test_codex_adapter_uses_requested_model_readonly_schema_no_shell(self):
        import subprocess
        def fake_run(command, **kwargs):
            self.assertIn("gpt-6.1-sol", command)
            self.assertIn("read-only", command)
            self.assertIn("--ephemeral", command)
            self.assertIn("--ignore-user-config", command)
            self.assertIn("shell_tool", command)
            self.assertIn("multi_agent", command)
            self.assertIn("model_providers.l1-chatgpt.supports_websockets=false", command)
            settings = [a for a in command if a.startswith("model_instructions_file=")]
            self.assertEqual(len(settings), 1)
            instructions_path = Path(json.loads(settings[0].split("=", 1)[1]))
            instructions = instructions_path.read_text(encoding="utf-8")
            self.assertIn(l1.SYSTEM_PROMPT, instructions)
            self.assertIn("Do not use tools", instructions)
            self.assertNotIn(l1.SYSTEM_PROMPT, kwargs["input"])
            self.assertNotIn(b"\r", instructions_path.read_bytes())
            self.assertIn("Findings complete. Final conclusion.", kwargs["input"])
            path = Path(command[command.index("--output-last-message") + 1])
            with path.open("w", encoding="utf-8", newline="\n") as f:
                f.write(l1.compact(response("Findings complete. Final conclusion.")))
            events = [{"type": "thread.started", "thread_id": "test-only"},
                      {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 34}}]
            return subprocess.CompletedProcess(command, 0, "\n".join(l1.compact(e) for e in events), "")
        config = {**CONFIG, "backend": "codex_cli", "model": "gpt-6.1-sol", "allow_agent_execution": True}
        with patch("l1.subprocess.run", side_effect=fake_run):
            result = l1.request_codex(config, "Findings complete. Final conclusion.", "hash")
        self.assertEqual(result["backend"], "codex_cli")
        self.assertEqual(result["usage"]["total_tokens"], 46)

    def test_codex_cli_failure_does_not_log_stderr_or_retry_all_reports(self):
        import subprocess
        config = {**CONFIG, "backend": "codex_cli", "allow_agent_execution": True}
        with patch("l1.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "secret-never-log")):
            with self.assertRaises(l1.FatalEndpointError) as caught:
                l1.request_codex(config, "Report", "hash")
        self.assertNotIn("secret-never-log", str(caught.exception))

    def test_codex_quota_failure_distinct_from_missing_credentials(self):
        import subprocess
        config = {**CONFIG, "backend": "codex_cli", "allow_agent_execution": True}
        with patch("l1.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "usage limit; secret-never-log")):
            with self.assertRaisesRegex(l1.FatalEndpointError, "available model quota") as caught:
                l1.request_codex(config, "Report", "hash")
        self.assertNotIn("secret-never-log", str(caught.exception))

    def test_language_hints_cover_additional_scripts_and_negation(self):
        examples = {"el": "ΕΥΡΗΜΑΤΑ Χωρίς συλλογή υγρού. Μηνίσκοι φυσιολογικοί.",
                    "tr": "Bulgular: Diz eklemi normaldir. Sonuç: menisküs normal.",
                    "bg": "МР находка: Няма данни за руптура. Ставен излив.",
                    "hr_sr_bs": "Nalazi se uredna morfologija koljena i hrskavice zgloba."}
        for expected, report in examples.items():
            self.assertEqual(l1.language_hint(report)[0], expected)
        for report in ["Χωρίς συλλογή", "Няма руптура", "yoktur", "bez rupture"]:
            self.assertTrue(l1.NEGATION.search(report))

    def test_literal_and_whitespace_evidence_only(self):
        report = "No\n  épanchement."
        found = l1.locate_quote(report, "No épanchement.")
        self.assertEqual(report[found["start"]:found["end"]], report)
        self.assertEqual(found["match"], "whitespace_only")
        self.assertIsNone(l1.locate_quote(report, "No effusion."))
        self.assertIsNone(l1.locate_quote(report, "no épanchement."))

    def test_non_knee_scope_masks_all_twelve_items(self):
        report = "Shoulder MRI: humerus fracture."
        obj = response(report)
        obj["report_scope"] = {"code": "OTHER", "evidence": ["Shoulder MRI"]}
        item = label("Fracture", "PRESENT", "humerus fracture.")
        fact(item, "fracture_line_present", True, "humerus fracture.")
        fact(item, "acute_injury", True, "humerus fracture.")
        obj["labels"]["Fracture"] = item
        row = {"report": report, "report_hash": l1.digest(report)}
        result = l1.extract_report(CONFIG, row, lambda *args: {"content": l1.compact(obj), "finish_reason": "stop"})
        self.assertEqual(result["parse_status"], "OK")
        self.assertEqual(sum(a["mask"] for a in result["audits"].values()), 0)
        self.assertTrue(all("report_scope_other" in a["review_reason"] for a in result["audits"].values()))

    def test_fabricated_evidence_and_asserted_facts_masked(self):
        report = "ACL normal."
        item = label("ACL", "ABSENT", "ACL intact.")
        self.assertEqual(l1.audit_label("ACL", item, report)["mask"], 0)
        item = label("ACL", "PRESENT", report)
        item["facts"]["high_grade_tear"]["value"] = True
        self.assertEqual(l1.audit_label("ACL", item, report)["mask"], 0)

    def test_schema_missing_category_bad_enum_uid_and_nonfinite(self):
        obj = response("A report")
        l1.validate_response(obj, l1.digest("A report"))
        bad = copy.deepcopy(obj)
        del bad["labels"]["MCL"]
        with self.assertRaises(jsonschema.ValidationError):
            l1.validate_response(bad, obj["report_hash"])
        with self.assertRaises(ValueError):
            l1.validate_response(obj, "wrong")
        bad = copy.deepcopy(obj)
        bad["labels"]["MCL"]["report_status"] = "NO"
        with self.assertRaises(jsonschema.ValidationError):
            l1.validate_response(bad, obj["report_hash"])
        with self.assertRaises(ValueError):
            l1.strict_json_loads('{"score":NaN}')
        with self.assertRaises(ValueError):
            l1.strict_json_loads('{"score":1,"score":0}')

    def test_two_repair_retries_then_error_not_negative(self):
        report = "Not a JSON response"
        row = {"report": report, "report_hash": l1.digest(report)}
        calls = []
        def requester(*args):
            calls.append(args)
            return {"content": "invalid", "finish_reason": "stop"}
        with patch("l1.time.sleep"):
            result = l1.extract_report(CONFIG, row, requester)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["parse_status"], "ERROR")
        self.assertEqual(result["retry_count"], 2)
        self.assertEqual(l1.error_label(result["error"])["mask"], 0)

    def test_repair_re_reads_original_and_recovers(self):
        report = "No abnormality.\nConclusion retained."
        row = {"report": report, "report_hash": l1.digest(report)}
        calls = []
        def requester(config, text, report_hash, repair):
            calls.append((text, repair))
            return {"content": "invalid" if len(calls) == 1 else l1.compact(response(report)), "finish_reason": "stop"}
        with patch("l1.time.sleep"):
            result = l1.extract_report(CONFIG, row, requester)
        self.assertEqual(result["parse_status"], "OK")
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c[0] == report for c in calls))

    def test_no_input_or_output_truncation(self):
        report = "Findings. Last conclusion is critical."
        row = {"report": report, "report_hash": l1.digest(report)}
        never = lambda *args: self.fail("Oversize reports must not be sent truncated")
        result = l1.extract_report({**CONFIG, "max_input_chars": 5}, row, never)
        self.assertEqual(result["segmentation_status"], "OVERSIZE_EXCLUDED")
        result = l1.extract_report({**CONFIG, "context_window_tokens": 5}, row, never)
        self.assertEqual(result["error"], "CONTEXT_BUDGET_EXCEEDED_NO_TRUNCATION")
        with patch("l1.time.sleep"):
            result = l1.extract_report(CONFIG, row, lambda *args: {"content": l1.compact(response(report)), "finish_reason": "length"})
        self.assertEqual(result["parse_status"], "ERROR")

    def test_endpoint_failure_stops_instead_of_making_4407_errors(self):
        def requester(*args):
            raise l1.FatalEndpointError("Unavailable")
        with self.assertRaises(l1.FatalEndpointError):
            l1.extract_report(CONFIG, {"report": "Report", "report_hash": "hash"}, requester)

    def test_http_payload_contains_complete_report_schema_and_no_gold(self):
        report = "Hallazgos:\nSin derrame.\nImpresión: normal."
        class FakeHTTPResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return l1.compact({"model": "test", "choices": [{"message": {"content": l1.compact(response(report))},
                                                                 "finish_reason": "stop"}]}).encode()
        def urlopen(request, timeout):
            obj = json.loads(request.data)
            original = json.loads(obj["messages"][1]["content"])
            self.assertEqual(set(original), {"report", "report_hash"})
            self.assertEqual(original["report"], report)
            self.assertEqual(obj["response_format"]["json_schema"]["schema"], l1.response_schema())
            return FakeHTTPResponse()
        with patch("l1.urllib.request.urlopen", side_effect=urlopen):
            result = l1.request_model(CONFIG, report, l1.digest(report))
        self.assertEqual(result["finish_reason"], "stop")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rsna_l1_test_")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def synthetic_train(self):
        rows = []
        for i in range(200):
            rows.append({l1.UID: str(i), "Report": f"Findings: ACL and MCL intact. Unique report {i}.\nConclusion: no fracture.",
                         **{c: "" for c in l1.CATEGORIES}})
        rows.append({l1.UID: "gold", "Report": rows[0]["Report"], **{c: "0" for c in l1.CATEGORIES}})
        path = self.path / "train.csv"
        l1.write_csv(path, rows, [l1.UID, "Report"] + l1.CATEGORIES)
        return path

    def test_prepare_full_coverage_non_gold_pilot_and_unchanged_source(self):
        train = self.synthetic_train()
        before = train.read_bytes()
        out = self.path / "prepared"
        with patch("l1.near_duplicates", return_value=([], 0)):
            summary = l1.prepare(train, out)
        self.assertEqual(summary["studies"], 201)
        self.assertEqual(summary["gold_studies"], 1)
        pilot = l1.read_csv(out / "pilot_sample.csv")
        self.assertEqual(len(pilot), 180)
        self.assertNotIn("gold", {r[l1.UID] for r in pilot})
        self.assertNotIn("0", {r[l1.UID] for r in pilot})
        self.assertEqual(train.read_bytes(), before)
        self.assertEqual(summary["pilot_sources"], {"language_length": 90, "high_risk": 90})
        self.assertTrue(all(not any(c in r for c in l1.CATEGORIES) for r in l1.read_jsonl(out / "reports_source.jsonl")))
        with self.assertRaises(ValueError):
            l1.prepare(train, out)
        l1.load_prepared(out)
        with (out / "reports_source.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write("\n")
        with self.assertRaises(ValueError):
            l1.load_prepared(out)

    def test_exports_mask_unknown_hard_only_known_lf_and_full_left_coverage(self):
        report = "No effusion."
        obj = response(report)
        obj["labels"]["Effusion"] = label("Effusion", "ABSENT", report)
        fact(obj["labels"]["Effusion"], "class_absence_supported", True, report)
        obj["labels"]["Effusion"]["llm_meets_criteria"] = "NO"
        row = {l1.UID: "one", "report": report, "report_hash": l1.digest(report),
               "report_group_id": l1.digest(report.lower()), "language": "en"}
        result = l1.extract_report(CONFIG, row, lambda *args: {"content": l1.compact(obj), "finish_reason": "stop"})
        raw, long, metrics = l1.exports(self.path, [row, {**row, l1.UID: "two"}], {row["report_hash"]: result}, CONFIG, "test")
        self.assertEqual(len(raw), 2)
        self.assertEqual(len(long), 24)
        self.assertEqual(metrics["duplicate_requests_saved"], 1)
        training = l1.read_csv(self.path / "labels_training.csv")
        self.assertEqual(training[0]["Effusion"], "0.05")
        self.assertEqual(training[0]["Synovitis"], "")
        self.assertEqual(training[0]["Synovitis__mask"], "0")
        hard = l1.read_csv(self.path / "labels_training_hard.csv")
        self.assertEqual(hard[0]["Effusion"], "0")
        self.assertEqual(hard[0]["Synovitis"], "")
        for file in self.path.iterdir():
            self.assertNotIn(b"\r", file.read_bytes())

    def test_gate_blocks_pending_reviews_then_passes_complete_reviews(self):
        train = self.synthetic_train()
        out = self.path / "prepared"
        with patch("l1.near_duplicates", return_value=([], 0)):
            meta = l1.prepare(train, out)
        _, records = l1.load_prepared(out)
        chosen = {r[l1.UID] for r in l1.read_csv(out / "pilot_sample.csv")}
        rows = [r for r in records if r[l1.UID] in chosen]
        cache = {r["report_hash"]: l1.extract_report(CONFIG, r, lambda config, text, *args:
                  {"content": l1.compact(response(text)), "finish_reason": "stop"}) for r in rows}
        stage = out / "pilot"
        stage.mkdir()
        raw, _, _ = l1.exports(stage, rows, cache, CONFIG, "test")
        l1.review_template(stage, raw, rows)
        l1.write_json(stage / "run.json", {"run_fingerprint": "test"})
        l1.write_jsonl(stage / "extraction_cache.jsonl", [])
        self.assertFalse(l1.evaluate_gate(out)["passed"])
        review = l1.read_csv(stage / "text_review.csv")
        for row in review:
            row.update({"disposition": "KEEP", "reviewer": "synthetic_test_only", **{k: "PASS" for k in l1.AUDIT_DIMENSIONS}})
        l1.write_csv(stage / "text_review.csv", review, list(review[0]))
        self.assertTrue(l1.evaluate_gate(out)["passed"])
        all_masked = copy.deepcopy(review)
        for entry in all_masked:
            entry.update({"disposition": "MASK", "notes": "Synthetic all-MASK gate regression"})
        l1.write_csv(stage / "text_review.csv", all_masked, list(all_masked[0]))
        self.assertFalse(l1.evaluate_gate(out)["passed"])
        failing = copy.deepcopy(review)
        for entry in failing:
            if entry["category"] == "ACL":
                entry.update({"disposition": "MASK", "notes": "Synthetic systematic semantic failure", "criteria_correct": "FAIL"})
        l1.write_csv(stage / "text_review.csv", failing, list(failing[0]))
        self.assertFalse(l1.evaluate_gate(out)["passed"])
        # False-negative NOT_MENTIONED and wrong-body errors count, even masked.
        for wrong_scope in [False, True]:
            missed = copy.deepcopy(raw)
            for item in missed:
                if wrong_scope:
                    item["report_scope"]["code"] = "OTHER"
                else:
                    item["labels"]["ACL"]["report_status"] = "NOT_MENTIONED"
            l1.write_jsonl(stage / "labels_raw.jsonl", missed)
            gate = l1.evaluate_gate(out)
            self.assertFalse(gate["passed"])
            self.assertIn("Mentioned-item text error rate above predeclared 5% threshold", gate["reasons"])
        l1.write_jsonl(stage / "labels_raw.jsonl", raw)
        review.pop()
        l1.write_csv(stage / "text_review.csv", review, list(review[0]))
        self.assertFalse(l1.evaluate_gate(out)["passed"])

    def test_crash_cache_tail_preserved_recovered_and_middle_corruption_rejected(self):
        path = self.path / "cache.jsonl"
        complete = b'{"report_hash":"complete"}\n'
        damaged = complete + b'{"report_hash":"unfinished'
        path.write_bytes(damaged)
        self.assertEqual(l1.read_cache(path), [{"report_hash": "complete"}])
        self.assertEqual(path.read_bytes(), complete)
        backups = list(self.path.glob("cache.jsonl.incomplete-*.bin"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), damaged)
        path.write_bytes(b'{"bad":\n' + complete)
        with self.assertRaisesRegex(ValueError, "middle cache"):
            l1.read_cache(path)
        path.write_bytes(complete.rstrip(b'\n'))
        self.assertEqual(l1.read_cache(path), [{"report_hash": "complete"}])
        self.assertEqual(path.read_bytes(), complete)

    def test_full_run_blocks_without_gate_before_network(self):
        train = self.synthetic_train()
        out = self.path / "prepared"
        with patch("l1.near_duplicates", return_value=([], 0)):
            l1.prepare(train, out)
        config_path = self.path / "config.json"
        l1.write_json(config_path, CONFIG)
        with patch("l1.request_model", side_effect=AssertionError("Must not contact endpoint")):
            with self.assertRaisesRegex(ValueError, "passed 180-report"):
                l1.run(out, config_path, "full")

    def test_auc_ties_and_fixed_gold_placeholder(self):
        self.assertEqual(l1.auc([0, 1, 0, 1], [.5, .5, .5, .5]), .5)
        self.assertEqual(l1.auc([0, 1], [.1, .9]), 1.)
        self.assertIsNone(l1.auc([0, 0], [.1, .9]))
        gold = [{l1.UID: "a", **{c: "0" for c in l1.CATEGORIES}},
                {l1.UID: "b", **{c: "1" for c in l1.CATEGORIES}}]
        check = l1.gold_check([], gold)
        self.assertEqual(check["gold_overlap"], 0)
        self.assertIsNone(check["macro_available_auc"])
        self.assertEqual(check["macro_all_gold_placeholder_auc"], .5)

    def test_literal_secrets_refused_and_missing_config_explicit(self):
        path = self.path / "config.json"
        l1.write_json(path, {**CONFIG, "api_key": "never-log-this"})
        with self.assertRaisesRegex(ValueError, "literal credentials"):
            l1.preflight(path)
        _, state = l1.preflight()
        self.assertFalse(state["ready"])
        self.assertIn("base_url", state["missing"])
        self.assertIn("model", state["missing"])
        self.assertIn("model_version", state["missing"])

    def test_complete_mocked_lifecycle_cache_gold_gate_and_tamper_check(self):
        # Entire pipeline uses synthetic reports and a fake transport in TEMP.
        # This is deliberately not a real model pilot or a production output.
        train = self.synthetic_train()
        out = self.path / "prepared"
        config_path = self.path / "config.json"
        l1.write_json(config_path, CONFIG)
        calls = []
        def request_model(config, report, report_hash, repair=None):
            calls.append(report_hash)
            return {"content": l1.compact(response(report)), "finish_reason": "stop", "usage": {}}
        def extract_report(config, row):
            return original_extract(config, row, requester=request_model)
        original_extract = l1.extract_report
        with contextlib.redirect_stdout(io.StringIO()), patch("l1.near_duplicates", return_value=([], 0)):
            l1.prepare(train, out)
            with patch("l1.extract_report", side_effect=extract_report):
                l1.run(out, config_path, "pilot")
                self.assertEqual(len(calls), 180)
                review_path = out / "pilot/text_review.csv"
                review = l1.read_csv(review_path)
                for row in review:
                    row.update({"disposition": "KEEP", "reviewer": "synthetic_test_only",
                                **{k: "PASS" for k in l1.AUDIT_DIMENSIONS}})
                # Explicit text-audit rejection is carried to full and duplicate UIDs.
                masked = next(r for r in review if r["category"] == "ACL")
                masked.update({"disposition": "MASK", "notes": "Synthetic mask propagation test"})
                l1.write_csv(review_path, review, list(review[0]))
                self.assertTrue(l1.evaluate_gate(out)["passed"])
                l1.run(out, config_path, "full")
                self.assertEqual(len(calls), 200)  # 180 cached +20 new unique texts, not 201 calls.
                metrics = json.loads((out / "full/labels_metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["studies"], 201)
                self.assertEqual(metrics["actual_label_rows"], 2412)
                self.assertEqual(metrics["gold_check"]["gold_overlap"], 1)
                with review_path.open("a", encoding="utf-8", newline="\n") as f:
                    f.write("\n")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    l1.run(out, config_path, "full")


if __name__ == "__main__":
    unittest.main()
