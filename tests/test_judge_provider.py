"""Offline tests for the judge provider selection in run_eval_native and score_realism_llm.

No test touches the network: the `anthropic` (and, where needed, `boto3`) modules are replaced in
`sys.modules` by fakes for the duration of each test.

    python -m unittest discover -s tests
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from synthesis_evaluation import run_eval_native, score_realism_llm

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
        return _Stream(self._owner.reply)


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


def _write_fixture(root: Path) -> dict:
    """One email with a character's given and family name, both detected and replaced."""
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
    paths["ground_truth"].write_text(json.dumps(gold) + "\n")
    paths["predictions"].write_text(json.dumps(preds) + "\n")
    paths["characters"].write_text(json.dumps(chars))
    return paths


class JudgeCliTestCase(unittest.TestCase):
    def setUp(self):
        self._model = score_realism_llm.MODEL
        self._tmp = tempfile.TemporaryDirectory()
        self.paths = _write_fixture(Path(self._tmp.name))
        env = mock.patch.dict(os.environ, {})
        env.start()
        self.addCleanup(env.stop)
        for key in ENV_KEYS:
            os.environ.pop(key, None)

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
        code, _, err, module = self.run_main("--judge-provider", "bedrock", "--judge-model",
                                             "global.anthropic.claude-opus-5-5", "--judge-region", "us-east-1",
                                             credentials_problem="no AWS credentials found")
        self.assertNotEqual(code, 0)
        self.assertIn("no AWS credentials found", err)
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
        self.assertEqual(client.kwargs, {"aws_region": "eu-west-1", "max_retries": score_realism_llm.MAX_RETRIES})
        [request] = client.requests
        self.assertEqual(request["model"], "global.anthropic.claude-opus-5-5")
        self.assertEqual(request["max_tokens"], 12_000)
        self.assertEqual(request["thinking"], {"type": "adaptive"})
        self.assertEqual(request["system"][0]["cache_control"], {"type": "ephemeral"})

        results = self.results()
        self.assertEqual(results["config"]["judge_provider"], "bedrock")
        self.assertEqual(results["config"]["judge_model"], "global.anthropic.claude-opus-5-5")
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
        code, _, err, _ = self.run_main("--judge-provider", "bedrock", "--judge-model",
                                        "global.anthropic.claude-opus-5-5", "--judge-region", "us-east-1",
                                        anthropic_module=module)
        self.assertNotEqual(code, 0)
        self.assertIn("LLM judge failed", err)
        self.assertIn("AccessDeniedException", err)
        self.assertNotIn("123456789012", err)
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
        module = _fake_anthropic()
        with mock.patch.dict(sys.modules, {"anthropic": module, "boto3": self._boto3(None)}):
            with self.assertRaises(score_realism_llm.JudgeError):
                score_realism_llm.score([], provider="bedrock", region="us-east-1")
        self.assertEqual(module.AnthropicBedrock.instances, [])


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
        self.assertEqual(results["config"]["judge_provider"], "anthropic")
        self.assertEqual(results["config"]["judge_model"], score_realism_llm.MODEL)

    def test_uses_the_direct_api(self):
        os.environ["ANTHROPIC_API_KEY"] = "fake"
        code, _, err, module = self.run_main("--judge-model", "claude-opus-5-5")
        self.assertEqual(code, 0, err)
        self.assertEqual(module.AnthropicBedrock.instances, [])
        [client] = module.Anthropic.instances
        self.assertEqual(client.kwargs, {"max_retries": 6})
        self.assertEqual(client.requests[0]["model"], "claude-opus-5-5")
        self.assertEqual(self.results()["metrics"]["overall"]["synthesis_accuracy"], 1.0)

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
        self.assertIsNone(self.results()["config"]["judge_provider"])
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
