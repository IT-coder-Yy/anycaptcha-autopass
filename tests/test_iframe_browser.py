"""本地无网络的 Chromium 回归：跨 iframe 拖动不能跳回主页面的旧鼠标位置。"""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.interaction import Interaction
from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker import browser_identity, browser_options, rotation_angle


@unittest.skipUnless(os.getenv("CAPTCHA_BROWSER_TEST") == "1", "需显式启用本地 Chromium 测试")
class IframeTests(unittest.TestCase):
    def test_cross_site_iframe_uses_native_screen_containing_mouse_coordinates(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as profile:
                async with async_playwright() as pw:
                    options = browser_options({"user_agent": "test", "screen": {"width": 2560, "height": 1440}})
                    context = await pw.chromium.launch_persistent_context(profile, headless=True, **options)
                    try:
                        async def serve(route):
                            if "frame.test" in route.request.url:
                                body = '<body><button id="handle" style="position:absolute;left:20px;top:285px;width:64px;height:40px">drag</button><script>window.lastMove=null;addEventListener("mousemove",e=>lastMove=[e.screenX,e.screenY])</script></body>'
                            else:
                                body = '<iframe src="https://frame.test/" style="border:0;position:absolute;left:470px;top:258px;width:380px;height:384px"></iframe>'
                            await route.fulfill(status=200, content_type="text/html", body=body)
                        await context.route("**/*", serve)  # 全部本地响应，不访问外网。
                        page = context.pages[0]
                        await page.goto("https://main.test/")
                        frame = page.frames[1]
                        await frame.locator("#handle").hover()
                        await frame.wait_for_function("lastMove !== null")
                        actual = await frame.evaluate("() => ({screen: [screen.width,screen.height], point: lastMove})")
                        self.assertEqual(actual["screen"], [2560, 1440])
                        self.assertIsNotNone(actual["point"])
                        self.assertTrue(all(0 <= point < limit for point, limit in zip(actual["point"], actual["screen"])))
                    finally:
                        await context.close()
        asyncio.run(exercise())

    def test_identity_matches_headers_main_frame_cross_site_and_headed_handoff(self):
        async def exercise():
            screen = {"width": 2560, "height": 1440}
            identity = await browser_identity({"screen": screen})
            options = browser_options({"screen": screen, "user_agent": identity["user_agent"]})
            seen = []
            async with async_playwright() as pw:
                for headless in ((True, False) if os.getenv("DISPLAY") else (True,)):
                    with tempfile.TemporaryDirectory() as profile:
                        context = await pw.chromium.launch_persistent_context(profile, headless=headless, **options)
                        try:
                            async def serve(route):
                                seen.append(await route.request.all_headers())
                                body = '<body>frame</body>' if 'frame.test' in route.request.url else '<iframe src="https://frame.test/"></iframe>'
                                await route.fulfill(content_type="text/html", body=body)
                            await context.route("**/*", serve)
                            page = context.pages[0]
                            await page.goto("https://main.test/")
                            self.assertEqual(len(page.frames), 2)
                            for frame in page.frames:
                                state = await frame.evaluate("""async () => ({ua:navigator.userAgent,platform:navigator.platform,
                                    hints:await navigator.userAgentData.getHighEntropyValues(['fullVersionList','platformVersion']),
                                    screen:[screen.width,screen.height]})""")
                                self.assertEqual(state["ua"], identity["user_agent"])
                                self.assertEqual(state["platform"], identity["query"]["browser_platform"])
                                self.assertEqual(state["hints"]["platform"], identity["metadata"]["platform"])
                                self.assertEqual(state["hints"]["fullVersionList"], identity["metadata"]["fullVersionList"])
                                self.assertEqual(state["screen"], [2560,1440])
                        finally:
                            await context.close()
            for headers in seen:
                self.assertEqual(headers["user-agent"], identity["user_agent"])
                for name, value in identity["headers"].items():
                    self.assertEqual(headers.get(name), value)
        asyncio.run(exercise())

    def test_rotation_measures_relative_angle_of_oppositely_rotating_discs(self):
        async def exercise():
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True, channel="chromium")
                try:
                    page = await browser.new_page()
                    await page.set_content('<img id="captcha_verify_image"><img class="captcha-verify-image-slide">')
                    for angle in (0, 9, 90, 179, 181):
                        await page.evaluate("""angle => {
                            document.querySelector('#captcha_verify_image').style.transform = `translate(-50%, -50%) rotateZ(${-angle}deg)`;
                            document.querySelector('img.captcha-verify-image-slide').style.transform = `translate(-50%, -50%) rotateZ(${angle}deg)`;
                        }""", angle)
                        measured = await rotation_angle(page)
                        self.assertAlmostEqual((measured + 360) % 360, (2 * angle) % 360, delta=.001)
                finally:
                    await browser.close()
        asyncio.run(exercise())

    def test_session_cookie_handoff_survives_browser_restart(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as profile:
                async with async_playwright() as pw:
                    first = await pw.chromium.launch_persistent_context(profile, headless=True, channel="chromium")
                    await first.add_cookies([{"name": "session", "value": "updated", "domain": "example.invalid", "path": "/"}])
                    current = await first.cookies()
                    await first.close()
                    second = await pw.chromium.launch_persistent_context(profile, headless=True, channel="chromium")
                    try:
                        self.assertEqual(await second.cookies(), [])
                        await second.add_cookies(current)
                        self.assertEqual([(c["name"], c["value"]) for c in await second.cookies()], [("session", "updated")])
                    finally:
                        await second.close()
        asyncio.run(exercise())

    def test_drag_continues_from_previous_global_position(self):
        async def exercise():
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True, channel="chromium")
                try:
                    page = await browser.new_page()
                    await page.set_content('<iframe style="position:absolute;left:100px;top:100px;width:500px;height:400px;border:0" srcdoc="<body></body>"></iframe>')
                    interaction = Interaction(page, track_pointer=False, straight_drag=False, seed=479)
                    await interaction.install()
                    await interaction.move(200, 200)
                    self.assertIsNone(await page.evaluate("window.__captchaTestPointer ?? null"))
                    frame = page.frames[1]
                    await frame.evaluate("window.moves=[]; addEventListener('mousemove', e => moves.push([e.clientX,e.clientY]))")
                    await page.mouse.down()
                    await interaction.move(240, 200, drag=True)
                    await page.mouse.up()
                    moves = await frame.evaluate("window.moves")
                    self.assertAlmostEqual(moves[0][0], 100, delta=2)
                    self.assertAlmostEqual(moves[-1][0], 140, delta=1)
                    self.assertAlmostEqual(moves[-1][1], 100, delta=1)
                    self.assertGreater(max(point[1] for point in moves) - min(point[1] for point in moves), 0)
                finally:
                    await browser.close()
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
