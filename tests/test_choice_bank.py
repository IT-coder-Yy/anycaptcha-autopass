import base64
from contextlib import redirect_stdout
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from cli import main
from color_letter.solver import centered_glyph, templates
from lab import choices
from lab.server import ChallengeStore
from photo_vqa.solver import PROMPT, photo


class ChoiceBankTests(unittest.TestCase):
    def test_card_bank_rebuilds_the_original_pixels_and_reuses_its_cache(self):
        with TemporaryDirectory() as root, patch.dict(os.environ, {"CAPTCHA_PROJECT_ROOT": root}):
            templates.cache_clear()
            self.addCleanup(templates.cache_clear)
            fingerprint = hashlib.sha256()
            for rows in choices.bank("color-letter").values():
                for png in choices.images("color-letter", rows):
                    with Image.open(BytesIO(png)) as image:
                        fingerprint.update(image.convert("RGB").tobytes())
            for _, mask in templates():
                fingerprint.update(mask.tobytes())
            # 原实验的 64 张 RGB 字卡和 16 张居中参考字形，按题库顺序拼接像素。
            self.assertEqual(fingerprint.hexdigest(), "dfd6b7abd0ba75974314d626ba86ee32cf1da5e6944079d7556159f05316b0d6")
            templates.cache_clear()
            with patch("color_letter.cards.ImageFont.truetype", side_effect=AssertionError("重复渲染字卡")):
                self.assertEqual(choices.evaluate("color-letter")["total"], 16)

    def test_color_letters_keep_pixel_height_and_case(self):
        mask = np.zeros((48, 60), np.uint8)
        mask[3:11, 10:15] = 255
        small = centered_glyph(mask)
        np.testing.assert_array_equal(small, centered_glyph(np.roll(mask, 9, axis=0)))
        mask[11:19, 10:15] = 255
        self.assertEqual((small.any(axis=1).sum(), centered_glyph(mask).any(axis=1).sum()), (8, 16))
        result = choices.evaluate("color-letter")
        self.assertEqual((result["correct"], result["total"], result["labels_correct"], result["options_correct"]),
                         (14, 16, 61, 62))
        red = next(q for q in result["samples"] if q["id"] == "letter-08")
        self.assertEqual(red["selected"], ["B", "C", "D"])
        self.assertEqual([v["value"] for v in red["items"].values()], ["红C", "红c", "红c", "红c"])

    def test_photo_cli_sends_original_prompt_and_marks_an_unanswered_option(self):
        frames = [bytes([i]) for i in range(4)]
        replies = dict(zip(frames, ("Yes.", "No", "Unsure", "No.")))
        def handle(request):
            self.assertEqual(str(request.url), "https://vision.invalid/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer local-test")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "qwen3.7-plus")
            message, = body["messages"]
            text, image = message["content"]
            self.assertEqual(text["text"], PROMPT + "Is there a bicycle in this picture?")
            jpeg = base64.b64decode(image["image_url"]["url"].removeprefix("data:image/jpeg;base64,"))
            return httpx.Response(200, json={"choices": [{"message": {"content": replies.pop(jpeg)}}]})

        client = httpx.Client
        config = {"DOUYIN_CAPTCHA_LLM_BASE_URL": "https://vision.invalid/v1", "DOUYIN_CAPTCHA_LLM_MODEL": "",
                  "DOUYIN_CAPTCHA_LLM_API_KEY": "local-test"}
        output = StringIO()
        with TemporaryDirectory() as root, patch.dict(os.environ, {**config, "CAPTCHA_PROJECT_ROOT": root}), \
                patch("photo_vqa.solver.httpx.Client", side_effect=lambda **kw: client(transport=httpx.MockTransport(handle), **kw)), \
                patch.object(choices, "images", return_value=frames), \
                patch("sys.argv", ["captcha-local", "recognize", "--kind", "photo-vqa", "object-01"]), redirect_stdout(output):
            main()
        result = json.loads(output.getvalue())
        self.assertFalse(replies)
        self.assertEqual((result["correct"], result["options_correct"], result["unanswered_options"]), (0, 3, 1))
        self.assertEqual(result["samples"][0]["unanswered"], ["C"])
        self.assertNotIn("local-test", output.getvalue())

    def test_photo_cache_only_accepts_the_manifest_hash(self):
        data = b"public sample"
        row = {"image": "000000056205.jpg", "sha256": hashlib.sha256(data).hexdigest()}
        with TemporaryDirectory() as root, patch.dict(os.environ, {"CAPTCHA_PROJECT_ROOT": root}):
            path = Path(root, ".cache", "photo-vqa", row["image"])
            with patch("photo_vqa.solver.urlopen", return_value=BytesIO(b"incorrect")):
                with self.assertRaisesRegex(ValueError, "校验失败"):
                    photo(row)
            self.assertFalse(path.exists())
            with patch("photo_vqa.solver.urlopen", return_value=BytesIO(data)) as fetch:
                self.assertEqual(photo(row), data)
                self.assertEqual(photo(row), data)
                fetch.assert_called_once()

    def test_unknown_case_never_opens_a_model_client(self):
        with patch("photo_vqa.solver.reader") as reader:
            with self.assertRaisesRegex(LookupError, "没有题号"):
                choices.evaluate("photo-vqa", "outside.jpg")
            reader.assert_not_called()

    def test_both_banks_use_unordered_exact_selection_in_the_local_form(self):
        for kind in choices.CHOICE_KINDS:
            bank = choices.bank(kind)
            self.assertEqual(len(bank), 16)
            rows = next(rows for rows in bank.values()
                        if sum(r["label"] == choices.target(kind, r) for r in rows) == 2)
            answer = "".join(r["option"] for r in rows if r["label"] == choices.target(kind, r))
            store = ChallengeStore()
            with patch.object(store.rng, "choice", return_value=rows):
                for submitted, correct in ((answer[::-1], True), (answer[:1], False), (answer + answer[0], False)):
                    public = store.issue(kind)
                    self.assertNotIn("text", public)
                    self.assertEqual(store.verify(public["key"], {"text": submitted})["accepted"], correct)
                    self.assertFalse(store.verify(public["key"], {"text": answer})["accepted"])
        with Image.open(BytesIO(choices.sheet("color-letter", choices.bank("color-letter")["letter-08"]))) as image:
            self.assertEqual(image.size, choices.SHEET_SIZE)
