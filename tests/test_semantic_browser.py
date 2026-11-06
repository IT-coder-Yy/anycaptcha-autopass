"""本地 Chromium 验证语义题的 iframe 坐标、图片一致性与失败停止行为。"""

import asyncio
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import semantic.solver as semantic

spec = importlib.util.spec_from_file_location("semantic_worker", ROOT / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

ENV = {"DOUYIN_CAPTCHA_SEMANTIC_ENABLED": "true",
       "DOUYIN_CAPTCHA_LLM_BASE_URL": "https://vision.invalid/v1",
       "DOUYIN_CAPTCHA_LLM_MODEL": "test", "CAPTCHA_ACTION_SCALE": ".25"}


@unittest.skipUnless(os.getenv("CAPTCHA_BROWSER_TEST") == "1", "需显式启用本地 Chromium 测试")
class SemanticBrowserTests(unittest.TestCase):
    async def exercise(self, *, scale=1, change_during_inference=False, confidence=.95,
                       reject_drag=False, replace_image=False, cancel_drag=False):
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, channel="chromium")
            try:
                page = await browser.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=2)
                fixture = (ROOT / "tests/fixtures/semantic.html").read_text()
                async def serve(route):
                    body = fixture if "frame.test" in route.request.url else (
                        '<iframe src="https://frame.test/" style="position:absolute;left:123px;top:87px;'
                        'border:0;width:550px;height:650px"></iframe>')
                    await route.fulfill(content_type="text/html", body=body)
                await page.route("**/*", serve)
                await page.goto("https://main.test/")
                frame = page.frames[1]
                await frame.locator(semantic.TILES).nth(8).wait_for()
                await frame.evaluate("""options => {
                    document.querySelector('#vc_captcha_box').style.transform = `scale(${options.scale})`;
                    window.rejectDrag = options.reject; window.replaceImageAfterDrag = options.replace;
                }""", {"scale": scale, "reject": reject_drag, "replace": replace_image})
                obs = SimpleNamespace(generation=1, challenge={"mode": "semantic_reasoning", "question": {
                    "title": "选择森林植被", "version": "2", "url": "private-image-url"}})
                async def answer(title, image, settings):
                    self.assertEqual(title, "选择森林植被")
                    self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
                    if change_during_inference:
                        obs.generation += 1
                    return {"criterion": "森林植被", "selected_indices": [2, 9], "confidence": confidence}
                original_move = semantic.Interaction.move
                async def move(interaction, *args, **kwargs):
                    if cancel_drag and kwargs.get("drag"):
                        raise asyncio.CancelledError()
                    await original_move(interaction, *args, **kwargs)
                record = {}
                with patch.dict(os.environ, ENV), patch.object(semantic, "recognize", AsyncMock(side_effect=answer)), patch.object(semantic.Interaction, "move", move):
                    if cancel_drag:
                        with self.assertRaises(asyncio.CancelledError):
                            await semantic.solve(page, frame, obs, record, stable_boxes=worker.stable_boxes)
                        result = "cancelled"
                    else:
                        result = await semantic.solve(page, frame, obs, record, stable_boxes=worker.stable_boxes)
                state = await frame.evaluate("() => ({drags, submitted, pressed})")
                return result, state, record
            finally:
                await browser.close()

    def test_measured_coordinates_work_with_cross_site_iframe_and_css_scaling(self):
        for scale in (1, .75):
            with self.subTest(scale=scale):
                result, state, record = asyncio.run(self.exercise(scale=scale))
                self.assertEqual(result, "submitted")
                self.assertTrue(state["submitted"])
                self.assertEqual([item["index"] for item in state["drags"]], [2, 9])
                self.assertTrue(all(item["trusted"] for item in state["drags"]))
                for actual, geometry in zip(state["drags"], record["geometry"]):
                    for key in ("source", "target"):
                        point = actual["start" if key == "source" else "end"]
                        self.assertAlmostEqual(point[0] + 123, geometry[key][0], delta=1)
                        self.assertAlmostEqual(point[1] + 87, geometry[key][1], delta=1)

    def test_changed_challenge_during_inference_is_not_touched(self):
        result, state, _ = asyncio.run(self.exercise(change_during_inference=True))
        self.assertEqual(result, "changed")
        self.assertEqual(state["drags"], [])
        self.assertFalse(state["submitted"])

    def test_low_confidence_does_not_drag_or_submit(self):
        result, state, _ = asyncio.run(self.exercise(confidence=.2))
        self.assertEqual(result, "unanswered")
        self.assertEqual(state["drags"], [])
        self.assertFalse(state["submitted"])

    def test_failed_drop_stops_without_submission(self):
        result, state, _ = asyncio.run(self.exercise(reject_drag=True))
        self.assertEqual(result, "unknown")
        self.assertEqual(len(state["drags"]), 1)
        self.assertFalse(state["submitted"])

    def test_replaced_canvas_stops_remaining_drags_without_submission(self):
        result, state, _ = asyncio.run(self.exercise(replace_image=True))
        self.assertEqual(result, "changed")
        self.assertEqual(len(state["drags"]), 1)
        self.assertFalse(state["submitted"])

    def test_cancellation_releases_mouse_without_submission(self):
        result, state, record = asyncio.run(self.exercise(cancel_drag=True))
        self.assertEqual(result, "cancelled")
        self.assertFalse(state["pressed"])
        self.assertFalse(state["submitted"])
        self.assertIn("actions", record)


if __name__ == "__main__":
    unittest.main()
