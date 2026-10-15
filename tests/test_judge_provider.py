"""Offline tests for the judge provider selection in run_eval_native and score_realism_llm.

No test touches the network: the `anthropic` module is replaced in `sys.modules` by a fake for the duration of
each test.

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

ENV_KEYS = ("ANTHROPIC_API_KEY", "AWS_REGION", "AWS_DEFAULT_REGION")

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


def _fake_anthropic(reply=json.dumps(GOOD_VERDICTS)):
    mod = types.ModuleType("anthropic")
    mod.Anthropic = _fake_client_class("Anthropic", reply)
    mod.AnthropicBedrock = _fake_client_class("AnthropicBedrock", reply)
    return mod


def _write_fixture(root: Path) -> dict:
    """One email with a character's given and family name, both detected and replaced."""
    gold = {"file": {"kind": "eml", "path": "email/a.eml", "container": None}, "text": "Hi Megan Donovan",
            "meta": {"row_id": "r1"}, "ground_truth_spans": [
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


BEDROCK_MODEL = "global.anthropic.claude-opus-5-5"
BEDROCK = ("--judge-provider", "bedrock", "--judge-model", BEDROCK_MODEL, "--judge-region", "us-east-1")


class JudgeProviderTests(unittest.TestCase):
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

    def run_main(self, *extra: str, module=None):
        argv = ["run_eval_native", "--predictions", str(self.paths["predictions"]),
                "--ground-truth", str(self.paths["ground_truth"]), "--characters", str(self.paths["characters"]),
                "--run-name", "run", "--runs-dir", str(self.paths["runs"]), "--workers", "2", *extra]
        module = module if module is not None else _fake_anthropic()
        stdout, stderr = io.StringIO(), io.StringIO()
        code = 0
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, {"anthropic": module}), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = run_eval_native.main()
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue(), module

    def results(self) -> dict:
        return json.loads((self.paths["runs"] / "run" / "results.json").read_text())

    def test_bedrock_uses_the_bedrock_client_without_an_api_key(self):
        code, out, _, module = self.run_main(*BEDROCK)
        self.assertEqual(code, 0)
        self.assertEqual(module.Anthropic.instances, [])
        [client] = module.AnthropicBedrock.instances
        self.assertEqual(client.kwargs, {"aws_region": "us-east-1", "max_retries": score_realism_llm.MAX_RETRIES})
        self.assertTrue(client.requests)
        self.assertTrue(all(r["model"] == BEDROCK_MODEL for r in client.requests))
        results = self.results()
        self.assertEqual(results["config"]["judge_provider"], "bedrock")
        self.assertEqual(results["config"]["judge_model"], BEDROCK_MODEL)
        self.assertEqual(results["detail"]["judge"]["provider"], "bedrock")
        self.assertEqual(results["detail"]["judge"]["per_label_totals"]["NAME_GIVEN"]["coherent"], 1)
        self.assertIn(f"LLM judge (bedrock: {BEDROCK_MODEL})", out)
        self.assertIn("judge: 0 of 2 pairs unjudged", out)

    def test_bedrock_without_a_region_leaves_it_to_the_sdk(self):
        code, _, _, module = self.run_main("--judge-provider", "bedrock", "--judge-model", BEDROCK_MODEL)
        self.assertEqual(code, 0)
        [client] = module.AnthropicBedrock.instances
        self.assertIsNone(client.kwargs["aws_region"])

    def test_bedrock_requires_a_judge_model(self):
        code, _, err, module = self.run_main("--judge-provider", "bedrock")
        self.assertNotEqual(code, 0)
        self.assertIn("--judge-model is required", err)
        self.assertEqual(module.AnthropicBedrock.instances, [])

    def test_a_failed_call_counts_its_pairs_as_skipped(self):
        code, out, _, _ = self.run_main(*BEDROCK, module=_fake_anthropic(reply=RuntimeError("throttled")))
        self.assertEqual(code, 0)
        self.assertEqual(self.results()["detail"]["judge"]["per_label_totals"]["NAME_GIVEN"]["skipped"], 1)
        self.assertIn("judge: 2 of 2 pairs unjudged", out)

    def test_anthropic_is_still_the_default(self):
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        code, out, _, module = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(module.AnthropicBedrock.instances, [])
        [client] = module.Anthropic.instances
        self.assertEqual(client.kwargs, {"max_retries": score_realism_llm.MAX_RETRIES})
        self.assertEqual(self.results()["detail"]["judge"]["provider"], "anthropic")
        self.assertIn("judge: 0 of 2 pairs unjudged", out)

    def test_anthropic_without_a_key_skips_the_judge(self):
        code, _, _, module = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(module.Anthropic.instances, [])
        self.assertEqual(self.results()["detail"]["judge"]["skipped_reason"], "ANTHROPIC_API_KEY not set")


if __name__ == "__main__":
    unittest.main()
