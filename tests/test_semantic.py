import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import semantic.solver as semantic

ENV = {"DOUYIN_CAPTCHA_SEMANTIC_ENABLED": "true",
       "DOUYIN_CAPTCHA_LLM_BASE_URL": "https://vision.example/v1",
       "DOUYIN_CAPTCHA_LLM_MODEL": "test-vision",
       "DOUYIN_CAPTCHA_LLM_API_KEY": "private-key"}
PREDICTION = {"criterion": "选择森林植被", "selected_indices": [6, 8], "confidence": .9}


class SemanticTests(unittest.TestCase):
    def setUp(self):
        # 原有模型协议测试不读取开发者本机的配置。
        patcher = patch.object(semantic, "dotenv_values", return_value={})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_settings_are_separate_from_text_llm_and_can_be_disabled(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "https://text.example", "LLM_MODEL": "text"}, clear=True):
            with self.assertRaisesRegex(semantic.SemanticError, "DOUYIN_CAPTCHA_LLM_BASE_URL"):
                semantic.settings_from_env()
            os.environ["DOUYIN_CAPTCHA_SEMANTIC_ENABLED"] = "false"
            with self.assertRaisesRegex(semantic.SemanticError, "未启用"):
                semantic.settings_from_env()
        with patch.dict(os.environ, ENV, clear=True):
            result = semantic.settings_from_env()
        self.assertEqual(result["timeout_seconds"], 45)
        self.assertEqual(result["min_confidence"], .8)

    def test_invalid_settings_fail_without_exposing_configured_secrets(self):
        for key, value in [("BASE_URL", "ftp://vision.example"),
                           ("BASE_URL", "https://private-user:private-pass@vision.example"),
                           ("BASE_URL", "https://vision.example?key=private-key"),
                           ("MODEL", ""), ("TIMEOUT_SECONDS", "0"), ("TIMEOUT_SECONDS", "NaN"),
                           ("TIMEOUT_SECONDS", "private-key"), ("MIN_CONFIDENCE", "1.1")]:
            with self.subTest(key=key, value=value), patch.dict(os.environ, {**ENV, semantic.PREFIX + key: value}, clear=True):
                with self.assertRaises(semantic.SemanticError) as error:
                    semantic.settings_from_env()
                self.assertNotIn("private-", str(error.exception))

    def test_selection_is_sorted_and_empty_answer_is_allowed_for_abstention(self):
        self.assertEqual(semantic.parse_prediction(json.dumps({**PREDICTION, "selected_indices": [8, 6]})), PREDICTION)
        empty = {**PREDICTION, "selected_indices": [], "confidence": 0}
        self.assertEqual(semantic.parse_prediction(json.dumps(empty)), empty)

    def test_invalid_model_outputs_cannot_become_browser_actions(self):
        invalid = ["not JSON", "[]", "null", "{}"]
        invalid += [json.dumps({**PREDICTION, "selected_indices": indices})
                    for indices in ([0], [10], [True], [1.5], ["1"], [1, 1], [[1]], "1,2")]
        invalid += [json.dumps({**PREDICTION, "confidence": score})
                    for score in (True, "0.9", -1, 2, float("nan"), float("inf"))]
        invalid += [json.dumps({**PREDICTION, "criterion": criterion}) for criterion in ("", None, "a" * 501)]
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(semantic.SemanticError):
                semantic.parse_prediction(content)

    def test_openai_compatible_request_contains_only_title_and_image(self):
        async def exercise():
            seen = []
            async def handle(request):
                seen.append(request)
                return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps(PREDICTION)}}]})
            with patch.dict(os.environ, ENV, clear=True):
                settings = semantic.settings_from_env()
            client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            with patch.object(httpx, "AsyncClient", return_value=client):
                result = await semantic.recognize("有哪些森林生态元素？", b"test-image", settings)
            self.assertEqual(result, PREDICTION)
            self.assertEqual(str(seen[0].url), "https://vision.example/v1/chat/completions")
            self.assertEqual(seen[0].headers["Authorization"], "Bearer private-key")
            self.assertNotIn("cookie", seen[0].headers)
            payload = json.loads(seen[0].content)
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertEqual(payload["model"], "test-vision")
            content = payload["messages"][1]["content"]
            self.assertEqual(json.loads(content[0]["text"]), {"题意": "有哪些森林生态元素？"})
            self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,dGVzdC1pbWFnZQ==")
        asyncio.run(exercise())

    def test_http_errors_truncated_and_malformed_responses_are_sanitized(self):
        async def exercise(response):
            with patch.dict(os.environ, ENV, clear=True):
                settings = semantic.settings_from_env()
            client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response))
            with patch.object(httpx, "AsyncClient", return_value=client):
                with self.assertRaises(semantic.SemanticError) as error:
                    await semantic.recognize("题意", b"image", settings)
                self.assertNotIn("private-key", str(error.exception))
        responses = [httpx.Response(401, text="private-key"), httpx.Response(200, text="private-key"),
                     httpx.Response(200, json={"choices": []}),
                     httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "private-key"}}]}),
                     httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"refusal": "private-key", "content": None}}]})]
        for response in responses:
            asyncio.run(exercise(response))

    def test_model_request_has_total_deadline_and_does_not_swallow_cancellation(self):
        async def exercise(cancel):
            async def handle(request):
                if cancel:
                    raise asyncio.CancelledError()
                await asyncio.sleep(1)
                return httpx.Response(200, json={})
            with patch.dict(os.environ, ENV, clear=True):
                settings = semantic.settings_from_env()
            settings["timeout_seconds"] = .01
            client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            with patch.object(httpx, "AsyncClient", return_value=client), self.assertRaises(
                    asyncio.CancelledError if cancel else semantic.SemanticError):
                await semantic.recognize("题意", b"image", settings)
        asyncio.run(exercise(False))
        asyncio.run(exercise(True))


class SemanticDotenvTests(unittest.TestCase):
    def test_project_env_loads_only_model_fields_without_mutating_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, ".env").write_text(
                'DOUYIN_CAPTCHA_LLM_BASE_URL="https://vision.example/v1"\n'
                'DOUYIN_CAPTCHA_LLM_MODEL="vision model" # inline comment\n'
                "DOUYIN_CAPTCHA_LLM_API_KEY='secret#${LITERAL}'\n"
                'DOUYIN_CAPTCHA_LLM_TIMEOUT_SECONDS=1\n')
            with patch.dict(os.environ, {"CAPTCHA_PROJECT_ROOT": directory}, clear=True):
                before = dict(os.environ)
                result = semantic.settings_from_env()
                self.assertEqual(dict(os.environ), before)
        self.assertEqual(result['base_url'], 'https://vision.example/v1')
        self.assertEqual(result['model'], 'vision model')
        self.assertEqual(result['api_key'], 'secret#${LITERAL}')
        self.assertEqual(result['timeout_seconds'], 45)

    def test_process_environment_takes_precedence_over_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, '.env')
            path.write_text('DOUYIN_CAPTCHA_LLM_BASE_URL=https://local.example/v1\n'
                            'DOUYIN_CAPTCHA_LLM_MODEL=local-model\n'
                            'DOUYIN_CAPTCHA_LLM_API_KEY=local-secret\n')
            with patch.dict(os.environ, {**ENV, 'DOUYIN_CAPTCHA_LLM_API_KEY': ''}, clear=True):
                result = semantic.settings_from_env(path)
        self.assertEqual(result['base_url'], ENV['DOUYIN_CAPTCHA_LLM_BASE_URL'])
        self.assertEqual(result['model'], ENV['DOUYIN_CAPTCHA_LLM_MODEL'])
        self.assertEqual(result['api_key'], '')

    def test_missing_file_still_supports_existing_environment_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, ENV, clear=True):
                result = semantic.settings_from_env(Path(directory, 'missing.env'))
        self.assertEqual(result['model'], 'test-vision')


if __name__ == "__main__":
    unittest.main()
