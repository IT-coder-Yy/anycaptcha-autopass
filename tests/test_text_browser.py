import asyncio
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import text.douyin as text_click
from common.vision import Prediction

spec = importlib.util.spec_from_file_location("text_worker", ROOT / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@unittest.skipUnless(os.getenv("CAPTCHA_BROWSER_TEST") == "1", "需显式启用本地 Chromium 测试")
class TextBrowserTests(unittest.TestCase):
    async def exercise(self, *, change=False, usable=True, invalid=False, reject=False, replace=False):
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, channel="chromium")
            try:
                page = await browser.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=2)
                fixture = (ROOT / "tests/fixtures/text.html").read_text()
                async def serve(route):
                    body = fixture if "frame.test" in route.request.url else (
                        '<iframe src="https://frame.test/" style="position:absolute;left:127px;top:91px;'
                        'border:0;width:550px;height:650px"></iframe>')
                    await route.fulfill(content_type="text/html", body=body)
                await page.route("**/*", serve)
                await page.goto("https://main.test/")
                frame = page.frames[1]
                await frame.locator(text_click.BACKGROUND).wait_for()
                await frame.evaluate("""options => {
                    document.querySelector('#vc_captcha_box').style.transform = 'scale(0.8)';
                    window.rejectClick = options.reject; window.replaceAfterClick = options.replace;
                }""", {"reject": reject, "replace": replace})
                obs = SimpleNamespace(generation=1)
                def answer(bg, hint, model_dir):
                    if change:
                        obs.generation += 1
                    h, w = bg.shape[:2]
                    points = [[w * .8, h * .6], [w * .2, h * .3]]
                    if invalid:
                        points[0][0] = w + 1
                    return Prediction("text", usable, .95, points=points)
                record = {}
                with tempfile.TemporaryDirectory() as directory:
                    for name in ("text-detector.onnx", "text-matcher.onnx"):
                        (Path(directory) / name).touch()
                    with patch.dict(os.environ, {"DOUYIN_CAPTCHA_MODEL_DIR": directory, "CAPTCHA_ACTION_SCALE": ".25"}), patch.object(text_click, "recognize", side_effect=answer):
                        result = await text_click.solve(page, frame, obs, record, stable_boxes=worker.stable_boxes)
                state = await frame.evaluate("() => ({clicks, submitted})")
                return result, state, record
            finally:
                await browser.close()

    def test_click_order_and_coordinates_survive_iframe_css_scale_and_device_scale(self):
        result, state, record = asyncio.run(self.exercise())
        self.assertEqual(result, "submitted")
        self.assertTrue(state["submitted"])
        self.assertEqual(len(state["clicks"]), 2)
        self.assertGreater(state["clicks"][0][0], state["clicks"][1][0])
        for actual, geometry in zip(state["clicks"], record["geometry"]):
            self.assertAlmostEqual(actual[0] + 127, geometry["page_point"][0], delta=1)
            self.assertAlmostEqual(actual[1] + 91, geometry["page_point"][1], delta=1)

    def test_stale_low_confidence_and_invalid_points_do_not_click(self):
        for options, expected in [({"change": True}, "changed"), ({"usable": False}, "unanswered"),
                                  ({"invalid": True}, "unavailable")]:
            with self.subTest(options=options):
                result, state, _ = asyncio.run(self.exercise(**options))
                self.assertEqual(result, expected)
                self.assertEqual(state["clicks"], [])
                self.assertFalse(state["submitted"])

    def test_missing_marker_and_changed_image_stop_before_submission(self):
        for options, expected in [({"reject": True}, "unknown"), ({"replace": True}, "changed")]:
            with self.subTest(options=options):
                result, state, _ = asyncio.run(self.exercise(**options))
                self.assertEqual(result, expected)
                self.assertEqual(len(state["clicks"]), 1)
                self.assertFalse(state["submitted"])
