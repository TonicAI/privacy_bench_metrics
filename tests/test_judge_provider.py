"""Offline tests for the judge provider selection in run_eval_native and score_realism_llm.

No test touches the network: the `anthropic` (and, where needed, `boto3`) modules are replaced in
`sys.modules` by fakes for the duration of each test.

    python -m unittest discover -s tests
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from synthesis_evaluation import native, run_eval_native, score_realism_llm
from synthesis_evaluation.load import load_jsonl

ENV_KEYS = ("ANTHROPIC_API_KEY", "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE")

GOOD_VERDICTS = {"verdicts": [
    {"label": "NAME_GIVEN", "surface": "Megan", "coherent": True,
     "values": [{"value": "Damon", "coherent": True}], "issues": [], "confidence": "sure"},
    {"label": "NAME_FAMILY", "surface": "Donovan", "coherent": True,
     "values": [{"value": "Stouds", "coherent": True}], "issues": [], "confidence": "sure"},
]}


class _Block:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _Usage:
    input_tokens = 100
    output_tokens = 50
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _Message:
    def __init__(self, text: str):
        self.content = [_Block(text)]
        self.usage = _Usage()


class _Stream:
    def __init__(self, reply):
        self._reply = reply

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        if isinstance(self._reply, Exception):
            raise self._reply
        return _Message(self._reply)


class _Messages:
    def __init__(self, owner):
        self._owner = owner

    def stream(self, **kwargs):
        self._owner.requests.append(kwargs)
        reply = self._owner.reply
        if callable(reply) and not isinstance(reply, Exception):
            reply = reply(kwargs)
        return _Stream(reply)


def _fake_client_class(name: str, reply):
    class FakeClient:
        instances: list = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.requests: list = []
            self.reply = reply
            self.messages = _Messages(self)
            type(self).instances.append(self)

    FakeClient.__name__ = name
    return FakeClient


def _fake_anthropic(reply=json.dumps(GOOD_VERDICTS), bedrock_reply=None):
    mod = types.ModuleType("anthropic")
    mod.Anthropic = _fake_client_class("Anthropic", reply)
    mod.AnthropicBedrock = _fake_client_class("AnthropicBedrock", reply if bedrock_reply is None else bedrock_reply)
    return mod


class FakeStatusError(Exception):
    """Shaped like the SDK's APIStatusError; a mid-stream error frame arrives with status 200 and an error code."""

    def __init__(self, status_code: int, code=None):
        super().__init__("Megan Donovan works at Acme00")
        self.status_code = status_code
        self.body = {"type": "error", "error": {"type": code, "message": "Megan"}} if code else None


class APIConnectionError(Exception):
    pass


class TransportError(Exception):
    pass


class ReadError(TransportError):
    pass


def _replies(*items):
    """A reply that answers successive calls with ``items`` in turn, repeating the last."""
    queue = list(items)

    def reply(_kwargs):
        return queue.pop(0) if len(queue) > 1 else queue[0]
    return reply


def _is_org_call(kwargs) -> bool:
    return kwargs["system"][0]["text"] == score_realism_llm.ORG_SYSTEM_PROMPT


def _user_text(kwargs) -> str:
    return kwargs["messages"][0]["content"]


def _org_verdict(surface: str) -> str:
    return json.dumps({"verdicts": [{"label": "ORGANIZATION", "surface": surface, "coherent": True,
                                     "values": [], "issues": [], "confidence": "sure"}]})


ORG_GROUP = "Acme Holdings"


def _write_fixture(root: Path, org_surfaces: int = 0, unowned: bool = False) -> dict:
    """One email with a character's given and family name, both detected and replaced; with ``org_surfaces``,
    a second email naming that many distinct surfaces of one org group, each detected and replaced; with
    ``unowned``, a third email naming an organization outside the roster, detected and replaced."""
    text = "Hi Megan Donovan"
    gold = {"file": {"kind": "eml", "path": "email/a.eml", "container": None}, "text": text, "meta": {"row_id": "r1"},
            "ground_truth_spans": [
                {"text": "Megan", "start": 3, "end": 8, "label": "NAME_GIVEN", "characters": ["c1"],
                 "location": {"kind": "eml", "file_char_start": 3, "file_char_end": 8}},
                {"text": "Donovan", "start": 9, "end": 16, "label": "NAME_FAMILY", "characters": ["c1"],
                 "location": {"kind": "eml", "file_char_start": 9, "file_char_end": 16}}]}
    preds = {"file": {"kind": "eml", "path": "email/a.eml", "container": None}, "spans": [
        {"label": "NAME_GIVEN", "text": "Megan", "new_text": "Damon",
         "location": {"kind": "eml", "file_char_start": 3, "file_char_end": 8}},
        {"label": "NAME_FAMILY", "text": "Donovan", "new_text": "Stouds",
         "location": {"kind": "eml", "file_char_start": 9, "file_char_end": 16}}]}
    chars = {"c1": {"canonical_name": "Megan Donovan", "first_names": ["Megan"], "last_names": ["Donovan"],
                    "organizations": []}}
    paths = {"ground_truth": root / "ground_truth.jsonl", "predictions": root / "predictions.jsonl",
             "characters": root / "characters.json", "runs": root / "runs"}
    gold_rows, pred_rows = [gold], [preds]
    if org_surfaces:
        names = [f"Acme{i:02d}" for i in range(org_surfaces)]
        org_text = " ".join(names)
        loc = lambda i: {"kind": "eml", "file_char_start": 7 * i, "file_char_end": 7 * i + 6}
        gold_rows.append({"file": {"kind": "eml", "path": "email/b.eml", "container": None}, "text": org_text,
                          "meta": {"row_id": "r2"}, "ground_truth_spans": [
                              {"text": n, "start": 7 * i, "end": 7 * i + 6, "label": "ORGANIZATION", "characters": [],
                               "org_group": ORG_GROUP, "location": loc(i)} for i, n in enumerate(names)]})
        pred_rows.append({"file": {"kind": "eml", "path": "email/b.eml", "container": None}, "spans": [
            {"label": "ORGANIZATION", "text": n, "new_text": f"Zeta{i:02d}", "location": loc(i)}
            for i, n in enumerate(names)]})
    if unowned:
        loc = {"kind": "eml", "file_char_start": 0, "file_char_end": 6}
        gold_rows.append({"file": {"kind": "eml", "path": "email/c.eml", "container": None}, "text": "Globex",
                          "meta": {"row_id": "r3"}, "ground_truth_spans": [
                              {"text": "Globex", "start": 0, "end": 6, "label": "ORGANIZATION", "characters": [],
                               "location": loc}]})
        pred_rows.append({"file": {"kind": "eml", "path": "email/c.eml", "container": None}, "spans": [
            {"label": "ORGANIZATION", "text": "Globex", "new_text": "Initech", "location": loc}]})
    paths["ground_truth"].write_text("".join(json.dumps(r) + "\n" for r in gold_rows))
    paths["predictions"].write_text("".join(json.dumps(r) + "\n" for r in pred_rows))
    paths["characters"].write_text(json.dumps(chars))
    return paths


BEDROCK = ("--judge-provider", "bedrock", "--judge-model", "global.anthropic.claude-opus-5-5",
           "--judge-region", "us-east-1")


ORIGINAL_RETRY_SLEEP = score_realism_llm._retry_sleep


class JudgeCliTestCase(unittest.TestCase):
    org_surfaces = 0
    unowned = False

    def setUp(self):
        self._model = score_realism_llm.MODEL
        self._tmp = tempfile.TemporaryDirectory()
        self.paths = _write_fixture(Path(self._tmp.name), org_surfaces=self.org_surfaces,
                                    unowned=self.unowned)
        env = mock.patch.dict(os.environ, {})
        env.start()
        self.addCleanup(env.stop)
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        sleep = mock.patch.object(score_realism_llm, "_retry_sleep", return_value=False)
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)

    def tearDown(self):
        score_realism_llm.MODEL = self._model
        self._tmp.cleanup()

    def run_dir(self, name: str = "run") -> Path:
        return self.paths["runs"] / name

    def run_main(self, *extra: str, name: str = "run", anthropic_module=None, credentials_problem=None):
        argv = ["run_eval_native", "--predictions", str(self.paths["predictions"]),
                "--ground-truth", str(self.paths["ground_truth"]), "--characters", str(self.paths["characters"]),
                "--run-name", name, "--runs-dir", str(self.paths["runs"]), "--workers", "2", *extra]
        module = anthropic_module if anthropic_module is not None else _fake_anthropic()
        stdout, stderr = io.StringIO(), io.StringIO()
        code = 0
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, {"anthropic": module}), \
                mock.patch.object(score_realism_llm, "bedrock_credentials_problem", return_value=credentials_problem), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = run_eval_native.main()
            except SystemExit as exc:
                code = exc.code
        if isinstance(code, str):
            stderr.write(code)
            code = 1
        return code, stdout.getvalue(), stderr.getvalue(), module

    def results(self, name: str = "run") -> dict:
        return json.loads((self.run_dir(name) / "results.json").read_text())


class ArgumentValidationTests(JudgeCliTestCase):
    def assert_rejected(self, *extra: str, message: str):
        code, _, err, module = self.run_main(*extra)
        self.assertNotEqual(code, 0)
        self.assertIn(message, err)
        self.assertFalse(self.run_dir().exists())
        self.assertEqual(module.AnthropicBedrock.instances, [])
        self.assertEqual(module.Anthropic.instances, [])

    def test_bedrock_requires_judge_model(self):
        os.environ["AWS_REGION"] = "us-east-1"
        self.assert_rejected("--judge-provider", "bedrock", message="--judge-model is required")

    def test_bedrock_requires_a_region(self):
        self.assert_rejected("--judge-provider", "bedrock", "--judge-model", "global.anthropic.claude-opus-5-5",
                             message="needs a region")

    def test_bedrock_cannot_skip_the_judge(self):
        self.assert_rejected("--judge-provider", "bedrock", "--judge-model", "global.anthropic.claude-opus-5-5",
                             "--judge-region", "us-east-1", "--skip-llm-judge", message="--skip-llm-judge")

    def test_region_flag_is_bedrock_only(self):
        self.assert_rejected("--judge-region", "us-east-1", message="--judge-region applies only")

    def test_unknown_provider(self):
        self.assert_rejected("--judge-provider", "vertex", message="invalid choice")

    def test_missing_credentials_fail_before_any_call(self):
        code, out, err, module = self.run_main(*BEDROCK, credentials_problem="no AWS credentials found")
        self.assertNotEqual(code, 0)
        self.assertIn("bedrock judge unavailable: no AWS credentials found", err)
        self.assertNotIn("loading predictions", out)
        self.assertFalse(self.run_dir().exists())
        self.assertEqual(module.AnthropicBedrock.instances, [])


class BedrockJudgeTests(JudgeCliTestCase):
    def test_constructs_anthropic_bedrock_with_the_region(self):
        os.environ["ANTHROPIC_API_KEY"] = "must-not-be-used"
        code, _, err, module = self.run_main("--judge-provider", "bedrock", "--judge-model",
                                             "global.anthropic.claude-opus-5-5", "--judge-region", "eu-west-1")
        self.assertEqual(code, 0, err)
        self.assertEqual(module.Anthropic.instances, [])
        [client] = module.AnthropicBedrock.instances
        self.assertEqual(client.kwargs, {"aws_region": "eu-west-1", "max_retries": 0})
        [request] = client.requests
        self.assertEqual(request["model"], "global.anthropic.claude-opus-5-5")
        self.assertEqual(request["max_tokens"], 12_000)
        self.assertEqual(request["thinking"], {"type": "adaptive"})
        self.assertEqual(request["system"][0]["cache_control"], {"type": "ephemeral"})

        results = self.results()
        self.assertEqual(results["config"]["judge_provider"], "bedrock")
        self.assertEqual(results["config"]["judge_model"], "global.anthropic.claude-opus-5-5")
        self.assertEqual(results["config"]["judge_effort"], "default")
        self.assertNotIn("output_config", request)
        judge = results["detail"]["judge"]
        self.assertNotIn("skipped_reason", judge)
        self.assertEqual(judge["provider"], "bedrock")
        overall = results["metrics"]["overall"]
        self.assertFalse(results["metrics"]["judge_skipped"])
        self.assertEqual(overall["synthesis_accuracy"], 1.0)
        self.assertEqual(overall["combined_accuracy"], 1.0)
        self.assertNotIn("eu-west-1", json.dumps(results))

    def test_region_falls_back_to_aws_region_then_aws_default_region(self):
        os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
        code, _, err, module = self.run_main("--judge-provider", "bedrock", "--judge-model",
                                             "us.anthropic.claude-opus-5-5", name="default")
        self.assertEqual(code, 0, err)
        self.assertEqual(module.AnthropicBedrock.instances[0].kwargs["aws_region"], "us-west-2")

        os.environ["AWS_REGION"] = "us-east-2"
        code, _, err, module = self.run_main("--judge-provider", "bedrock", "--judge-model",
                                             "us.anthropic.claude-opus-5-5", name="region")
        self.assertEqual(code, 0, err)
        self.assertEqual(module.AnthropicBedrock.instances[0].kwargs["aws_region"], "us-east-2")

    def test_arn_account_id_is_not_recorded(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic.claude-opus-5-5"
        code, out, err, module = self.run_main("--judge-provider", "bedrock", "--judge-model", arn,
                                               "--judge-region", "us-east-1")
        self.assertEqual(code, 0, err)
        self.assertEqual(module.AnthropicBedrock.instances[0].requests[0]["model"], arn)
        results = self.results()
        self.assertEqual(results["config"]["judge_model"],
                         "arn:aws:bedrock:us-east-1:<account>:inference-profile/global.anthropic.claude-opus-5-5")
        self.assertNotIn("123456789012", json.dumps(results))
        self.assertNotIn("123456789012", out)

    def test_failing_call_fails_the_run(self):
        module = _fake_anthropic(bedrock_reply=RuntimeError(
            "AccessDeniedException: arn:aws:sts::123456789012:assumed-role/judge is not authorized"))
        code, _, err, _ = self.run_main(*BEDROCK, anthropic_module=module)
        self.assertNotEqual(code, 0)
        self.assertIn("LLM judge failed, no results written: bedrock judge call for character 1 of 1 failed: "
                      "RuntimeError after 1 attempt(s)", err)
        self.assertNotIn("123456789012", err)
        self.assertNotIn("assumed-role", err)
        self.assertEqual(len(module.AnthropicBedrock.instances[0].requests), 1)
        self.assertFalse((self.run_dir() / "results.json").exists())
        self.assertFalse(self.run_dir().exists())

    def test_unparseable_response_fails_the_run(self):
        module = _fake_anthropic(bedrock_reply="I cannot help with that.")
        code, _, err, _ = self.run_main("--judge-provider", "bedrock", "--judge-model",
                                        "global.anthropic.claude-opus-5-5", "--judge-region", "us-east-1",
                                        anthropic_module=module)
        self.assertNotEqual(code, 0)
        self.assertIn("did not parse", err)
        self.assertFalse(self.run_dir().exists())

    def test_score_raises_when_nothing_to_judge(self):
        with mock.patch.dict(sys.modules, {"anthropic": _fake_anthropic()}), \
                mock.patch.object(score_realism_llm, "bedrock_credentials_problem", return_value=None):
            with self.assertRaises(score_realism_llm.JudgeError):
                score_realism_llm.score([], provider="bedrock", region="us-east-1")


class BedrockReplyCoverageTests(JudgeCliTestCase):
    """R-1 and R-2: a reply that judges none of its pairs fails the run; one that leaves some out is counted."""

    def assert_fails_without_results(self, reply, message: str):
        code, out, err, _ = self.run_main(*BEDROCK, anthropic_module=_fake_anthropic(bedrock_reply=reply))
        self.assertNotEqual(code, 0)
        self.assertIn(message, err)
        self.assertFalse(self.run_dir().exists())
        for gold in ("Megan", "Donovan", "c1"):
            self.assertNotIn(gold, err)
        return out, err

    def test_empty_verdict_list_fails(self):
        self.assert_fails_without_results(json.dumps({"verdicts": []}),
                                          "character 1 of 1 failed: the response judged none of its 2 pairs")

    def test_verdicts_matching_no_pair_fail(self):
        reply = json.dumps({"verdicts": [
            {"label": "NAME_GIVEN", "surface": "megan", "coherent": True, "values": []},
            {"label": "GIVEN_NAME", "surface": "Donovan", "coherent": True, "values": []}]})
        self.assert_fails_without_results(reply, "the response judged none of its 2 pairs")

    def test_non_dict_verdicts_fail(self):
        self.assert_fails_without_results(json.dumps({"verdicts": ["ok", "ok"]}),
                                          "the response judged none of its 2 pairs")

    def test_a_run_with_every_pair_unjudged_fails(self):
        with mock.patch.object(score_realism_llm, "_judged_pairs", return_value=1):
            self.assert_fails_without_results(json.dumps({"verdicts": []}), "bedrock judge left all 2 pairs unjudged")

    def assert_malformed(self, *entries, malformed: int):
        reply = json.dumps({"verdicts": list(entries)})
        self.assert_fails_without_results(
            reply, f"character 1 of 1 failed: {malformed} of the response's {len(entries)} verdicts for its pairs "
                   "are malformed")

    def test_verdict_without_coherent_fails(self):
        good = GOOD_VERDICTS["verdicts"]
        renamed = {k: v for k, v in good[0].items() if k != "coherent"}
        self.assert_malformed({**renamed, "verdict": "coherent"}, good[1], malformed=1)

    def test_string_coherent_fails(self):
        good = GOOD_VERDICTS["verdicts"]
        self.assert_malformed({**good[0], "coherent": "false"}, {**good[1], "coherent": None}, malformed=2)

    def test_string_value_verdict_fails(self):
        good = GOOD_VERDICTS["verdicts"]
        self.assert_malformed({**good[0], "values": [{"value": "Damon", "coherent": "yes"}]}, good[1], malformed=1)

    def test_values_that_are_not_a_list_fail(self):
        good = GOOD_VERDICTS["verdicts"]
        for values in (1, "Damon", {"value": "Damon", "coherent": True}):
            with self.subTest(values=values):
                self.assert_malformed({**good[0], "values": values}, good[1], malformed=1)

    def test_values_may_be_omitted(self):
        good = GOOD_VERDICTS["verdicts"]
        reply = json.dumps({"verdicts": [{k: v for k, v in good[0].items() if k != "values"}, good[1]]})
        code, _, err, _ = self.run_main(*BEDROCK, anthropic_module=_fake_anthropic(bedrock_reply=reply))
        self.assertEqual(code, 0, err)
        self.assertEqual(self.results()["metrics"]["overall"]["synthesis_accuracy"], 1.0)

    def test_an_unexpected_judge_error_leaves_no_run_dir(self):
        with mock.patch.object(score_realism_llm, "score", side_effect=TypeError("boom")):
            with self.assertRaises(TypeError):
                self.run_main(*BEDROCK)
        self.assertFalse(self.run_dir().exists())

    def test_partial_omission_counts_as_skipped_and_is_printed(self):
        reply = json.dumps({"verdicts": GOOD_VERDICTS["verdicts"][:1]})
        code, out, err, _ = self.run_main(*BEDROCK, anthropic_module=_fake_anthropic(bedrock_reply=reply))
        self.assertEqual(code, 0, err)
        self.assertIn("judge: 1 of 2 pairs unjudged (50.0%), counted as skipped", out)
        results = self.results()
        self.assertEqual(results["detail"]["judge"]["per_label_totals"]["NAME_FAMILY"]["skipped"], 1)
        self.assertEqual(results["metrics"]["overall"]["skipped"], 1)

    def test_complete_reply_prints_zero_unjudged(self):
        code, out, err, _ = self.run_main(*BEDROCK)
        self.assertEqual(code, 0, err)
        self.assertIn("judge: 0 of 2 pairs unjudged (0.0%)", out)


class BedrockRetryTests(JudgeCliTestCase):
    """R-3: retryable failures, including mid-stream ones, are retried with backoff; others are not."""

    def run_with(self, reply):
        module = _fake_anthropic(bedrock_reply=reply)
        code, out, err, _ = self.run_main(*BEDROCK, anthropic_module=module)
        return code, out, err, module.AnthropicBedrock.instances[0].requests

    def test_mid_stream_throttle_is_retried(self):
        code, _, err, requests = self.run_with(_replies(FakeStatusError(200, "throttlingException"),
                                                        json.dumps(GOOD_VERDICTS)))
        self.assertEqual(code, 0, err)
        self.assertEqual(len(requests), 2)
        self.assertEqual(self.sleep.call_count, 1)
        self.assertEqual(self.results()["metrics"]["overall"]["synthesis_accuracy"], 1.0)

    def test_retryable_failures_are_retried(self):
        for exc in (FakeStatusError(408), FakeStatusError(429), FakeStatusError(500), FakeStatusError(529),
                    FakeStatusError(503), FakeStatusError(504), FakeStatusError(200, "timeout_error"),
                    FakeStatusError(200, "modelTimeoutException"),
                    FakeStatusError(200, "overloaded_error"), FakeStatusError(200, "internalServerException"),
                    FakeStatusError(200, "modelStreamErrorException"), APIConnectionError("reset"), ReadError("eof")):
            with self.subTest(exc=type(exc).__name__, status=getattr(exc, "status_code", None)):
                self.assertTrue(score_realism_llm.is_retryable(exc))

    def test_other_failures_are_not_retried(self):
        for exc in (FakeStatusError(400), FakeStatusError(401), FakeStatusError(403), FakeStatusError(404),
                    FakeStatusError(409), FakeStatusError(422), FakeStatusError(200, "invalid_request_error"), FakeStatusError(200, "validationException"),
                    FakeStatusError(200, "accessDeniedException"), FakeStatusError(200), RuntimeError("boom"),
                    ValueError("bad")):
            with self.subTest(exc=type(exc).__name__, status=getattr(exc, "status_code", None)):
                self.assertFalse(score_realism_llm.is_retryable(exc))

    def test_access_denied_is_tried_once(self):
        code, _, err, requests = self.run_with(FakeStatusError(403))
        self.assertNotEqual(code, 0)
        self.assertIn("FakeStatusError (HTTP 403) after 1 attempt(s)", err)
        self.assertEqual(len(requests), 1)
        self.sleep.assert_not_called()
        self.assertFalse(self.run_dir().exists())

    def test_retries_are_bounded_with_exponential_backoff(self):
        code, _, err, requests = self.run_with(FakeStatusError(200, "throttlingException"))
        self.assertNotEqual(code, 0)
        self.assertIn("FakeStatusError (HTTP 200, throttlingException) after 5 attempt(s)", err)
        self.assertEqual(len(requests), score_realism_llm.JUDGE_ATTEMPTS)
        delays = [c.args[1] for c in self.sleep.call_args_list]
        self.assertEqual(len(delays), score_realism_llm.JUDGE_ATTEMPTS - 1)
        for attempt, delay in enumerate(delays, 1):
            ceiling = min(score_realism_llm.RETRY_CAP_SEC, score_realism_llm.RETRY_BASE_SEC * 2 ** (attempt - 1))
            self.assertGreaterEqual(delay, ceiling / 2)
            self.assertLessEqual(delay, ceiling)
        self.assertFalse(self.run_dir().exists())

    def test_error_message_carries_no_exception_text(self):
        _, out, err, _ = self.run_with(FakeStatusError(200, "throttlingException"))
        for gold in ("Megan", "Donovan", "Acme"):
            self.assertNotIn(gold, err + out)

    def test_default_provider_does_not_retry(self):
        os.environ["ANTHROPIC_API_KEY"] = "fake"
        module = _fake_anthropic(reply=_replies(FakeStatusError(200, "overloaded_error"), json.dumps(GOOD_VERDICTS)))
        code, _, err, _ = self.run_main(anthropic_module=module)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(module.Anthropic.instances[0].requests), 1)
        self.assertEqual(self.results()["detail"]["judge"]["per_label_totals"]["NAME_GIVEN"]["skipped"], 1)


class BedrockOrgGroupTests(JudgeCliTestCase):
    """R-4 (B2) and R-5: a chunked org group fails the run on any bad chunk, and the error names no org."""

    org_surfaces = 30

    def org_reply(self, first_chunk):
        def reply(kwargs):
            if not _is_org_call(kwargs):
                return json.dumps(GOOD_VERDICTS)
            return first_chunk if "'Acme00'" in _user_text(kwargs) else _org_verdict("Acme29")
        return reply

    def assert_org_failure(self, first_chunk, message: str):
        code, out, err, module = self.run_main(*BEDROCK, anthropic_module=_fake_anthropic(
            bedrock_reply=self.org_reply(first_chunk)))
        self.assertNotEqual(code, 0)
        self.assertIn(f"bedrock judge call for org group 1 of 1 failed: chunk 1 of 2: {message}", err)
        self.assertFalse(self.run_dir().exists())
        self.assertNotIn("Acme", err + out)
        self.assertNotIn(ORG_GROUP, err + out)

    def test_both_chunks_judged(self):
        code, out, err, module = self.run_main(*BEDROCK, anthropic_module=_fake_anthropic(
            bedrock_reply=self.org_reply(_org_verdict("Acme00"))))
        self.assertEqual(code, 0, err)
        org_requests = [r for r in module.AnthropicBedrock.instances[0].requests if _is_org_call(r)]
        self.assertEqual(len(org_requests), 2)
        self.assertIn("judge: 28 of 32 pairs unjudged", out)

    def test_unparseable_first_chunk_fails_the_run(self):
        self.assert_org_failure("not json", "the response did not parse as a verdict list")

    def test_first_chunk_matching_no_pair_fails_the_run(self):
        self.assert_org_failure(_org_verdict("acme00"), "the response judged none of its 25 pairs")

    def test_failing_first_chunk_fails_the_run(self):
        self.assert_org_failure(FakeStatusError(400), "FakeStatusError (HTTP 400) after 1 attempt(s)")

    def test_queued_calls_never_start_after_a_failure(self):
        module = _fake_anthropic(bedrock_reply=FakeStatusError(403))
        code, _, err, _ = self.run_main(*BEDROCK, "--workers", "1", anthropic_module=module)
        for thread in threading.enumerate():
            if thread.name.startswith("ThreadPoolExecutor"):
                thread.join(timeout=5)
        self.assertNotEqual(code, 0)
        self.assertIn("character 1 of 1 failed", err)
        self.assertEqual(len(module.AnthropicBedrock.instances[0].requests), 1)


    def test_an_in_flight_owner_starts_no_further_call_after_a_failure(self):
        org_started, char_failed = threading.Event(), threading.Event()

        def reply(kwargs):
            if not _is_org_call(kwargs):
                org_started.wait(timeout=5)
                char_failed.set()
                return FakeStatusError(403)
            org_started.set()
            char_failed.wait(timeout=5)
            time.sleep(0.2)
            return _org_verdict("Acme00")

        module = _fake_anthropic(bedrock_reply=reply)
        code, _, err, _ = self.run_main(*BEDROCK, "--workers", "2", anthropic_module=module)
        for thread in threading.enumerate():
            if thread.name.startswith("ThreadPoolExecutor"):
                thread.join(timeout=5)
        self.assertNotEqual(code, 0)
        self.assertIn("character 1 of 1 failed", err)
        requests = module.AnthropicBedrock.instances[0].requests
        self.assertEqual([_is_org_call(r) for r in requests], [False, True])


class BedrockCancellationTests(JudgeCliTestCase):
    """R-8 and R-13: once a call fails the run, queued work is cancelled, no further request is sent, and score()
    does not wait for calls in flight. Ordering comes from events, not timing."""

    org_surfaces = 30
    unowned = True

    def test_queued_work_is_cancelled_and_in_flight_call_is_not_waited_for(self):
        org_started, release, org_returned = threading.Event(), threading.Event(), threading.Event()

        def reply(kwargs):
            if _is_org_call(kwargs):
                org_started.set()
                release.wait(timeout=3)
                org_returned.set()
                return _org_verdict("Acme00")
            if kwargs["system"][0]["text"] == score_realism_llm.UNOWNED_SYSTEM_PROMPT:
                return json.dumps({"verdicts": []})
            org_started.wait(timeout=3)
            return FakeStatusError(403)

        shutdowns = []

        class SpyExecutor(concurrent.futures.ThreadPoolExecutor):
            def shutdown(self, wait=True, *, cancel_futures=False):
                shutdowns.append({"wait": wait, "cancel_futures": cancel_futures})
                super().shutdown(wait=wait, cancel_futures=cancel_futures)

        module = _fake_anthropic(bedrock_reply=reply)
        with mock.patch.object(concurrent.futures, "ThreadPoolExecutor", SpyExecutor):
            code, _, err, _ = self.run_main(*BEDROCK, "--workers", "2", anthropic_module=module)
            returned_while_in_flight = not org_returned.is_set()
            release.set()
            for thread in threading.enumerate():
                if thread.name.startswith("ThreadPoolExecutor"):
                    thread.join(timeout=5)
        self.assertNotEqual(code, 0)
        self.assertIn("character 1 of 1 failed", err)
        self.assertTrue(returned_while_in_flight, "score() waited for the in-flight org call")
        self.assertEqual(shutdowns, [{"wait": False, "cancel_futures": True}])
        requests = module.AnthropicBedrock.instances[0].requests
        self.assertEqual(len(requests), 2)
        self.assertFalse(self.run_dir().exists())


class GuardedTests(unittest.TestCase):
    def test_skips_the_owner_once_the_run_has_failed(self):
        stop = threading.Event()
        stop.set()
        process = mock.Mock()
        self.assertIsNone(score_realism_llm._guarded(stop, True, process, "owner"))
        process.assert_not_called()

    def test_a_strict_failure_stops_the_run(self):
        stop = threading.Event()
        score_realism_llm._guarded(stop, True, mock.Mock(return_value=("k", None, "failed", "")), "owner")
        self.assertTrue(stop.is_set())

    def test_a_default_provider_failure_does_not(self):
        stop = threading.Event()
        score_realism_llm._guarded(stop, False, mock.Mock(return_value=("k", None, "failed", "")), "owner")
        self.assertFalse(stop.is_set())


class RetryLoopTests(unittest.TestCase):
    def test_backoff_wakes_as_soon_as_the_run_fails(self):
        stop, sleeping = threading.Event(), threading.Event()
        client = _fake_client_class("AnthropicBedrock", FakeStatusError(429))()
        outcome = []

        def sleep(stop_, delay):
            sleeping.set()
            return ORIGINAL_RETRY_SLEEP(stop_, delay)

        def call():
            try:
                score_realism_llm._call_llm_with_retries(client, "system", "user", stop)
            except BaseException as exc:
                outcome.append(exc)

        with mock.patch.object(score_realism_llm, "_retry_sleep", sleep), \
                mock.patch.object(score_realism_llm, "RETRY_BASE_SEC", 60.0):
            worker = threading.Thread(target=call, daemon=True)
            worker.start()
            self.assertTrue(sleeping.wait(timeout=5))
            stop.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), "the backoff did not wake when the run failed")
        self.assertIsInstance(outcome[0], score_realism_llm._Cancelled)
        self.assertEqual(len(client.requests), 1)


def _real_sdk():
    try:
        import anthropic
        import httpx2
    except ImportError:
        return None
    return anthropic if hasattr(anthropic, "AnthropicBedrock") else None


@unittest.skipUnless(_real_sdk(), "needs anthropic[bedrock] installed")
class RealSdkRequestCountTests(unittest.TestCase):
    """R-12 against the real SDK over a mock transport: a judge call sends at most JUDGE_ATTEMPTS requests."""

    def setUp(self):
        env = mock.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "AKIAEXAMPLE", "AWS_SECRET_ACCESS_KEY": "example"})
        env.start()
        self.addCleanup(env.stop)
        for key in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
            os.environ.pop(key, None)
        sleep = mock.patch.object(score_realism_llm, "_retry_sleep", return_value=False)
        sleep.start()
        self.addCleanup(sleep.stop)

    def count_requests(self, respond) -> int:
        import httpx2
        requests = []

        def handler(request):
            requests.append(request.url.path)
            return respond(request)

        client = score_realism_llm._make_client("bedrock", "us-east-1")
        client._client = httpx2.Client(transport=httpx2.MockTransport(handler))
        with self.assertRaises(score_realism_llm._CallFailed) as caught:
            score_realism_llm._call_llm_with_retries(client, "system", "user", threading.Event())
        self.assertIn(f"after {score_realism_llm.JUDGE_ATTEMPTS} attempt(s)", str(caught.exception))
        self.assertTrue(all(p.endswith("/invoke-with-response-stream") for p in requests))
        return len(requests)

    def test_sustained_throttling(self):
        import httpx2
        self.assertEqual(self.count_requests(lambda r: httpx2.Response(429, json={"message": "slow down"})),
                         score_realism_llm.JUDGE_ATTEMPTS)

    def test_sustained_connection_errors(self):
        import httpx2

        def refuse(request):
            raise httpx2.ConnectError("refused", request=request)
        self.assertEqual(self.count_requests(refuse), score_realism_llm.JUDGE_ATTEMPTS)


class BedrockCredentialsTests(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {})
        env.start()
        self.addCleanup(env.stop)
        for key in ENV_KEYS:
            os.environ.pop(key, None)

    def _boto3(self, credentials):
        session = mock.Mock()
        session.get_credentials.return_value = credentials
        mod = types.ModuleType("boto3")
        mod.Session = mock.Mock(return_value=session)
        return mod

    def check(self, region="us-east-1", credentials=None):
        boto3 = self._boto3(credentials)
        with mock.patch.dict(sys.modules, {"anthropic": _fake_anthropic(), "boto3": boto3}):
            return score_realism_llm.bedrock_credentials_problem(region), boto3

    def test_no_credentials(self):
        problem, boto3 = self.check(credentials=None)
        self.assertIn("no AWS credentials", problem)
        boto3.Session.assert_called_once_with(region_name="us-east-1")

    def test_credentials_present(self):
        problem, _ = self.check(credentials=object())
        self.assertIsNone(problem)

    def test_bearer_token_counts(self):
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "token"
        problem, boto3 = self.check(credentials=None)
        self.assertIsNone(problem)
        boto3.Session.assert_not_called()

    def test_no_region(self):
        problem, _ = self.check(region=None, credentials=object())
        self.assertIn("no AWS region", problem)

    def test_score_raises_on_missing_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_fixture(Path(tmp))
            preds = native.load_predictions(load_jsonl(paths["predictions"]))
            rows, _, _ = native.join(list(load_jsonl(paths["ground_truth"])), preds, threshold=native.MATCH_THRESHOLD)
        module = _fake_anthropic()
        with mock.patch.dict(sys.modules, {"anthropic": module, "boto3": self._boto3(None)}):
            with self.assertRaises(score_realism_llm.JudgeError) as caught:
                score_realism_llm.score(rows, provider="bedrock", region="us-east-1")
        self.assertIn("no AWS credentials", str(caught.exception))
        self.assertEqual(module.AnthropicBedrock.instances, [])


BASE_CONFIG_KEYS = ["predictions", "ground_truth", "characters", "run_name", "format", "overlap_threshold",
                    "judge_model", "n_rows", "tier", "labels"]


class AnthropicDefaultTests(JudgeCliTestCase):
    def test_skips_without_api_key(self):
        code, _, err, module = self.run_main()
        self.assertEqual(code, 0, err)
        self.assertEqual(module.Anthropic.instances, [])
        self.assertEqual(module.AnthropicBedrock.instances, [])
        results = self.results()
        self.assertEqual(results["detail"]["judge"]["skipped_reason"], "ANTHROPIC_API_KEY not set")
        self.assertTrue(results["metrics"]["judge_skipped"])
        self.assertIsNone(results["metrics"]["overall"]["synthesis_accuracy"])
        self.assertEqual(list(results["config"]), BASE_CONFIG_KEYS)
        self.assertEqual(results["config"]["judge_model"], score_realism_llm.MODEL)

    def test_uses_the_direct_api(self):
        os.environ["ANTHROPIC_API_KEY"] = "fake"
        code, out, err, module = self.run_main("--judge-model", "claude-opus-5-5")
        self.assertEqual(code, 0, err)
        self.assertEqual(module.AnthropicBedrock.instances, [])
        [client] = module.Anthropic.instances
        self.assertEqual(client.kwargs, {"max_retries": 6})
        self.assertEqual(client.requests[0]["model"], "claude-opus-5-5")
        results = self.results()
        self.assertEqual(results["metrics"]["overall"]["synthesis_accuracy"], 1.0)
        self.assertEqual(list(results["config"]), BASE_CONFIG_KEYS)
        self.assertEqual(list(results["detail"]["judge"])[:2], ["model", "usage"])
        self.assertIn("LLM judge (claude-opus-5-5) ...", out)
        self.assertNotIn("pairs unjudged", out)

    def test_failing_call_still_counts_as_skipped(self):
        os.environ["ANTHROPIC_API_KEY"] = "fake"
        module = _fake_anthropic(reply=RuntimeError("overloaded"))
        code, _, err, _ = self.run_main(anthropic_module=module)
        self.assertEqual(code, 0, err)
        judge = self.results()["detail"]["judge"]
        self.assertIn("overloaded", judge["per_character"]["c1"]["error"])
        self.assertEqual(judge["per_label_totals"]["NAME_GIVEN"]["skipped"], 1)
        self.assertEqual(self.results()["metrics"]["overall"]["synthesis_accuracy"], 0.0)

    def test_skip_flag(self):
        code, _, err, module = self.run_main("--skip-llm-judge")
        self.assertEqual(code, 0, err)
        self.assertEqual(list(self.results()["config"]), BASE_CONFIG_KEYS)
        self.assertIsNone(self.results()["config"]["judge_model"])


class RedactionTests(unittest.TestCase):
    def test_plain_ids_unchanged(self):
        for model in ("claude-opus-4-7", "global.anthropic.claude-opus-5-5", "anthropic.claude-opus-5-5"):
            self.assertEqual(score_realism_llm.redact_account_ids(model), model)

    def test_arn_account_redacted(self):
        self.assertEqual(
            score_realism_llm.redact_account_ids("arn:aws:bedrock:us-east-1:123456789012:inference-profile/x"),
            "arn:aws:bedrock:us-east-1:<account>:inference-profile/x")
        self.assertEqual(
            score_realism_llm.redact_account_ids("User: arn:aws:sts::123456789012:assumed-role/r/s denied"),
            "User: arn:aws:sts::<account>:assumed-role/r/s denied")


if __name__ == "__main__":
    unittest.main()
