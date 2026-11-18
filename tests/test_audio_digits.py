from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

import httpx

from audio_digits.asr import normalize_digits, transcribe
from cli import main


class AudioDigitsTests(unittest.TestCase):
    def test_normalization_preserves_spoken_digits_and_rejects_unrelated_words(self):
        cases = {"0,0,7,2.": "0072", "one one oh zero.": "1100", "０１８０３。": "01803",
                 "<|zh|>零幺两七。": "0127", "0ero.": None, "Phone.": None, "即系。": None, "": None}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_digits(raw), expected)

    def test_cli_sends_one_file_and_preserves_the_transcript(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("ASR_BASE_URL=https://speech.example/v1/\nASR_API_KEY=test-key\nASR_MODEL=file-model\n")
            audio = root / "private-recording.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", ""))
                wav.writeframes(bytes(320))
            requests = []

            def handle(request):
                requests.append(request)
                self.assertEqual((request.method, str(request.url)), ("POST", "https://speech.example/v1/audio/transcriptions"))
                self.assertEqual(request.headers["authorization"], "Bearer test-key")
                self.assertIn(b'name="model"\r\n\r\nprocess-model', request.content)
                self.assertIn(b'name="file"; filename="audio.wav"', request.content)
                self.assertNotIn(b"private-recording", request.content)
                self.assertIn(audio.read_bytes(), request.content)
                return httpx.Response(200, json={"text": "0,0,7,2."})

            client = httpx.Client(transport=httpx.MockTransport(handle))
            output = StringIO()
            with patch.dict(os.environ, {"CAPTCHA_PROJECT_ROOT": directory, "ASR_MODEL": "process-model"}, clear=True), \
                    patch("audio_digits.asr.httpx.Client", return_value=client), \
                    patch("sys.argv", ["captcha-local", "transcribe-digits", str(audio)]), redirect_stdout(output):
                main()
            self.assertEqual(len(requests), 1)
            self.assertEqual(json.loads(output.getvalue()), {"model": "process-model", "text": "0,0,7,2.", "digits": "0072"})

    def test_api_failure_and_invalid_response_are_separate_from_non_digit_transcripts(self):
        settings = {"ASR_BASE_URL": "https://speech.example/v1", "ASR_API_KEY": "test-key", "ASR_MODEL": "test-model"}
        with TemporaryDirectory() as directory, patch.dict(os.environ, settings, clear=True):
            audio = Path(directory, "clip.mp3")
            audio.touch()
            env = Path(directory, ".env")
            for status, body, error in ((429, {"message": "test-key"}, "HTTP 429"), (200, {"text": 4}, "text 字段")):
                client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body)))
                with self.subTest(status=status), patch("audio_digits.asr.httpx.Client", return_value=client):
                    with self.assertRaisesRegex(ValueError, error) as raised:
                        transcribe(audio, env)
                    self.assertNotIn("test-key", str(raised.exception))
            client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"text": "Phone."})))
            with patch("audio_digits.asr.httpx.Client", return_value=client):
                self.assertIsNone(transcribe(audio, env)["digits"])

    def test_missing_key_and_invalid_address_fail_before_upload(self):
        settings = {"ASR_BASE_URL": "https://speech.example/v1", "ASR_MODEL": "test-model"}
        with TemporaryDirectory() as directory, patch.dict(os.environ, settings, clear=True), \
                patch("audio_digits.asr.httpx.Client") as client:
            audio, env = Path(directory, "clip.wav"), Path(directory, ".env")
            with self.assertRaisesRegex(ValueError, "ASR_API_KEY"):
                transcribe(audio, env)
            for url in ("http://speech.example/v1", "https://speech.example:bad/v1"):
                with patch.dict(os.environ, {"ASR_BASE_URL": url, "ASR_API_KEY": "test-key"}):
                    with self.assertRaisesRegex(ValueError, "ASR_BASE_URL"):
                        transcribe(audio, env)
            client.assert_not_called()
