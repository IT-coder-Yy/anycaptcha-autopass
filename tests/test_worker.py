import asyncio
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

spec = importlib.util.spec_from_file_location("captcha_worker", Path(__file__).resolve().parents[1] / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    def test_slow_cdp_does_not_replay_expired_mouse_points_in_a_burst(self):
        async def exercise():
            now = [0.0]
            emitted = []
            async def sleep(duration):
                now[0] += duration
            async def move(x, y):
                emitted.append((x,y,now[0]))
                if len(emitted) == 1:
                    now[0] += .080  # 一次 CDP 卡顿使后续数帧过期。
            page = SimpleNamespace(mouse=SimpleNamespace(move=AsyncMock(side_effect=move)))
            interaction = worker.Interaction(page, track_pointer=False, straight_drag=False, seed=479)
            with patch('common.interaction.time.monotonic', side_effect=lambda:now[0]), patch('common.interaction.asyncio.sleep', side_effect=sleep):
                await interaction.move(180,60,drag=True)
            self.assertEqual(emitted[-1][:2], (180,60))
            self.assertGreater(emitted[1][2]-emitted[0][2], .080)
            self.assertTrue(all(right[2]-left[2] > .001 for left,right in zip(emitted,emitted[1:])))
        asyncio.run(exercise())

    def test_three_total_attempts_and_refresh_only_after_rejection(self):
        async def exercise(outcomes):
            observation = SimpleNamespace(generation=1, searches=0, challenge={"mode": "slide", "id": "test"})

            async def refresh(**kwargs):
                observation.generation += 1

            frame = SimpleNamespace(locator=lambda _: SimpleNamespace(click=AsyncMock(side_effect=refresh)))
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(worker, "challenge_frame", AsyncMock(return_value=frame)), patch.object(worker, "solve_slide", AsyncMock(side_effect=outcomes)) as solve, patch.object(worker.asyncio, "sleep", AsyncMock()), patch.object(worker, "emit"):
                    result = await worker.automatic(None, None, observation, Path(directory))
                    return result, solve.await_count, len(list(Path(directory).glob("attempt-*.json")))

        self.assertEqual(asyncio.run(exercise(["rejected"] * 3)), (False, 3, 3))
        self.assertEqual(asyncio.run(exercise(["rejected", "accepted"])), ("automatic", 2, 2))
        self.assertEqual(asyncio.run(exercise(["unknown"])), (False, 1, 1))

    def test_second_challenge_after_acceptance_uses_remaining_attempts(self):
        async def exercise():
            obs = SimpleNamespace(generation=1, searches=0, challenge={"mode": "slide", "id": "test"})
            async def solve(*args):
                obs.generation += 1  # 通过后跳转，仍遇到真实新题。
                return "accepted"
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(worker, "challenge_frame", AsyncMock(return_value=object())), patch.object(worker, "solve_slide", AsyncMock(side_effect=solve)) as solver, patch.object(worker.asyncio, "sleep", AsyncMock()), patch.object(worker, "emit"):
                    self.assertFalse(await worker.automatic(None, None, obs, Path(directory)))
                self.assertEqual(solver.await_count, 3)
                self.assertEqual(len(list(Path(directory).glob("attempt-*.json"))), 3)
        asyncio.run(exercise())

    def test_passed_http_challenge_confirms_original_before_answering_page_followup(self):
        async def exercise():
            obs = SimpleNamespace(generation=1, searches=0, attempts=0, challenge={"mode":"slide","id":"original"})
            async def solve(*args):
                obs.generation += 1
                obs.challenge = {"mode":"semantic_reasoning","id":"page-followup"}
                return "accepted"
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(worker,"challenge_frame",AsyncMock(return_value=object())), patch.object(worker,"solve_slide",AsyncMock(side_effect=solve)) as solver, patch.object(worker,"emit"):
                    self.assertEqual(await worker.automatic(None,None,obs,Path(directory),confirm_original=True),"automatic")
                    solver.assert_awaited_once()
                    self.assertEqual(obs.attempts,1)
        asyncio.run(exercise())

    def test_challenge_response_before_iframe_mount_does_not_skip_automatic(self):
        async def exercise():
            obs = SimpleNamespace(generation=1, searches=0, challenge={"mode": "slide", "id": "test"})
            async def solve(*args):
                obs.searches += 1
                return "accepted"
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(worker, "challenge_frame", AsyncMock(side_effect=[None, None, object()])), patch.object(worker, "solve_slide", AsyncMock(side_effect=solve)) as solver, patch.object(worker.asyncio, "sleep", AsyncMock()), patch.object(worker, "emit"):
                    self.assertEqual(await worker.automatic(None, None, obs, Path(directory)), "automatic")
                    solver.assert_awaited_once()
        asyncio.run(exercise())

    def test_unknown_type_goes_to_manual_without_guessing(self):
        async def exercise():
            obs = SimpleNamespace(generation=1, searches=0, challenge={"mode": "unknown", "id": "test"})
            with patch.object(worker, "challenge_frame", AsyncMock(return_value=object())), patch.object(worker, "solve_slide", AsyncMock()) as solve, patch.object(worker, "emit"):
                self.assertFalse(await worker.automatic(None, None, obs, None))
                solve.assert_not_awaited()
        asyncio.run(exercise())

    def test_semantic_routes_to_solver_and_unavailable_does_not_refresh(self):
        async def exercise(outcome):
            obs = SimpleNamespace(generation=1, searches=0, attempts=0,
                                  challenge={"mode": "semantic_reasoning", "id": "test"})
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(worker, "challenge_frame", AsyncMock(return_value=object())), patch.object(worker, "solve_semantic", AsyncMock(return_value=outcome)) as solve, patch.object(worker, "solve_slide", AsyncMock()) as slide, patch.object(worker, "emit"):
                    result = await worker.automatic(None, None, obs, Path(directory), confirm_original=True)
                    solve.assert_awaited_once()
                    slide.assert_not_awaited()
                    self.assertEqual(obs.attempts, 1)
                    self.assertEqual(result, "automatic" if outcome == "accepted" else False)
        asyncio.run(exercise("accepted"))
        asyncio.run(exercise("unavailable"))

    def test_semantic_submission_requires_bound_receipt(self):
        async def exercise():
            obs = SimpleNamespace(generation=7)
            for outcome in ("accepted", "rejected", "unknown"):
                record = {}
                with patch("semantic.solver.solve", AsyncMock(return_value="submitted")), patch.object(worker, "wait_for_receipt", AsyncMock(return_value=outcome)) as receipt:
                    self.assertEqual(await worker.solve_semantic(None, None, None, obs, record), outcome)
                    receipt.assert_awaited_once_with(None, obs, 7, record)
            with patch("semantic.solver.solve", AsyncMock(return_value="changed")), patch.object(worker, "wait_for_receipt", AsyncMock()) as receipt:
                self.assertEqual(await worker.solve_semantic(None, None, None, obs, {}), "changed")
                receipt.assert_not_awaited()
        asyncio.run(exercise())

    def test_text_routes_to_local_solver_and_requires_receipt(self):
        async def exercise():
            obs = SimpleNamespace(generation=7, searches=0, attempts=0, challenge={"mode": "text", "id": "test"})
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(worker, "challenge_frame", AsyncMock(return_value=object())), patch("text.douyin.solve", AsyncMock(return_value="submitted")) as solve, patch.object(worker, "wait_for_receipt", AsyncMock(return_value="accepted")) as receipt, patch.object(worker, "emit"):
                    self.assertEqual(await worker.automatic(None, None, obs, Path(directory), confirm_original=True), "automatic")
                    solve.assert_awaited_once()
                    self.assertEqual(receipt.await_args.args[2], 7)
                    self.assertEqual(obs.attempts, 1)
        asyncio.run(exercise())

    def test_semantic_disabled_goes_to_manual_before_browser_or_network_access(self):
        async def exercise():
            with patch.dict("os.environ", {"DOUYIN_CAPTCHA_SEMANTIC_ENABLED": "false"}, clear=True), patch.object(worker, "emit"), patch.object(worker, "wait_for_receipt", AsyncMock()) as receipt:
                record = {}
                self.assertEqual(await worker.solve_semantic(None, None, None, SimpleNamespace(generation=1), record), "unavailable")
                self.assertIn("DOUYIN_CAPTCHA_SEMANTIC_ENABLED", record["reason"])
                receipt.assert_not_awaited()
        asyncio.run(exercise())

    def test_refresh_requires_new_search_and_disappearing_challenge(self):
        async def exercise(new_search):
            obs = SimpleNamespace(generation=1, searches=0)
            async def reload(**kwargs):
                if new_search:
                    obs.searches += 1
                else:
                    obs.generation += 1
            page = SimpleNamespace(reload=AsyncMock(side_effect=reload))
            with patch.object(worker, "challenge_frame", AsyncMock(side_effect=[object(), None])), patch.object(worker, "emit"):
                self.assertEqual(await worker.refresh_challenge(page, obs), new_search)
            page.reload.assert_awaited_once()
        asyncio.run(exercise(True))
        asyncio.run(exercise(False))

    def test_receipt_is_bound_to_request_generation(self):
        async def exercise():
            obs = worker.Observations(SimpleNamespace(on=lambda *_: None))
            request = type("Request", (), {"url": "https://verify.example/captcha/verify"})()
            obs.generation = 1
            obs.request(request)
            obs.generation = 2
            response = SimpleNamespace(request=request, json=AsyncMock(return_value={"code": 200}))
            await obs.capture(response, "/captcha/verify")
            self.assertEqual(obs.receipts[0]["generation"], 1)
            self.assertNotEqual(obs.receipts[0]["generation"], obs.generation)
        asyncio.run(exercise())

    def test_browser_search_records_only_verification_parameter_allowlist(self):
        obs = worker.Observations(SimpleNamespace(on=lambda *_: None))
        obs.request(SimpleNamespace(url="https://www.douyin.com/aweme/v1/web/general/search/single/?webid=browser&msToken=fresh%3D&keyword=private"))
        self.assertEqual(obs.verify_params, {"webid": "browser", "msToken": "fresh="})

    def test_cookie_values_and_proxy_credentials_are_preserved(self):
        self.assertEqual(worker.cookie_items("a=x=y; b=2")[0]["value"], "x=y")
        self.assertEqual(worker.browser_proxy("http://user:p%40ss@localhost:8080"), {"server": "http://localhost:8080", "username": "user", "password": "p@ss"})

    def test_layout_must_stabilize_again_after_element_disappears(self):
        async def exercise():
            box = {"x": 10, "y": 20, "width": 50, "height": 50}
            element = SimpleNamespace(bounding_box=AsyncMock(side_effect=[box, box, box, None, box, box, box, box]))
            with patch.object(worker.asyncio, "sleep", AsyncMock()):
                self.assertEqual(await worker.stable_boxes([element]), [box])
            self.assertEqual(element.bounding_box.await_count, 8)
        asyncio.run(exercise())

    def test_interrupted_attempt_is_recorded_without_sensitive_error_message(self):
        async def exercise():
            obs = SimpleNamespace(generation=1, searches=0, challenge={"mode": "slide", "id": "private-id"})
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(worker, "challenge_frame", AsyncMock(return_value=object())), patch.object(worker, "solve_slide", AsyncMock(side_effect=TimeoutError("private-value"))), patch.object(worker, "emit"):
                    with self.assertRaises(TimeoutError):
                        await worker.automatic(None, None, obs, Path(directory))
                record = (Path(directory) / "attempt-1.json").read_text()
                self.assertIn('"outcome": "interrupted"', record)
                self.assertNotIn("private", record)
        asyncio.run(exercise())

    def test_automatic_replay_refreshes_cookies_without_restarting_solver(self):
        async def exercise(replies):
            ctx = SimpleNamespace(pages=[SimpleNamespace(goto=AsyncMock())],
                add_cookies=AsyncMock(), clear_cookies=AsyncMock(), close=AsyncMock(),
                cookies=AsyncMock(side_effect=[[{"name": "session", "value": value}] for value in ("before-redirect", "after-redirect", "settled")]))
            launch = AsyncMock(return_value=ctx)
            manager = AsyncMock()
            manager.__aenter__.return_value = SimpleNamespace(chromium=SimpleNamespace(launch_persistent_context=launch))
            obs = SimpleNamespace(challenge={"mode": "slide"}, searches=0, generation=2, verify_params={},
                receipts=[{"code": 200, "generation": 1}], closed=False, finish=AsyncMock())
            commands = asyncio.Queue()
            for reply in replies:
                await commands.put(reply)
            with tempfile.TemporaryDirectory() as directory:
                payload = {"timeout": 5, "profile": str(Path(directory) / "profile"),
                           "keyword": "测试", "user_agent": "test", "cookies": "session=old",
                           "screen": {"width": 2560, "height": 1440}}
                with patch.object(worker, "async_playwright", return_value=manager), patch.object(worker, "Observations", return_value=obs), patch.object(worker, "refresh_challenge", AsyncMock(return_value=False)), patch.object(worker, "automatic", AsyncMock(return_value="automatic")) as solve, patch.object(worker.asyncio, "sleep", AsyncMock()) as sleep, patch.object(worker, "emit") as emit:
                    old_mask = worker.os.umask(0o077)
                    try:
                        await worker.run(payload, commands)
                    finally:
                        worker.os.umask(old_mask)
                    solve.assert_awaited_once()
                    launch.assert_awaited_once()
                    self.assertEqual(sleep.await_count, len(replies) - 1)
                sessions = [c.kwargs for c in emit.call_args_list if c.args == ("session",)]
                self.assertEqual([s["cookies"] for s in sessions], ["session=" + v for v in ("before-redirect", "after-redirect", "settled")][:len(replies)])
                self.assertTrue(all(s["mode"] == "automatic" for s in sessions))
                handoffs = json.loads(next(Path(directory).glob("runs/*/report.json")).read_text())["handoffs"]
                self.assertTrue(handoffs[-1]["recovered"])
        asyncio.run(exercise([{"recovered": False, "retry": True}, {"recovered": True}]))
        asyncio.run(exercise([{"recovered": False, "retry": True}] * 2 + [{"recovered": True}]))

    def test_manual_handoff_reuses_profile_without_reseeding_old_cookies(self):
        async def exercise(manual_code, render_error=False):
            def context():
                return SimpleNamespace(pages=[SimpleNamespace(goto=AsyncMock())],
                    add_cookies=AsyncMock(), clear_cookies=AsyncMock(), close=AsyncMock(),
                    cookies=AsyncMock(return_value=[{"name": "session", "value": "updated"}]))
            headless, headed = context(), context()
            launch = AsyncMock(side_effect=[headless, headed])
            pw = SimpleNamespace(chromium=SimpleNamespace(launch_persistent_context=launch))
            manager = AsyncMock()
            manager.__aenter__.return_value = pw
            observations = [SimpleNamespace(challenge={"mode": "slide"}, searches=int(code is None), generation=1, verify_params={},
                            receipts=[] if code is None else [{"code": code, "generation": 1}], closed=False, finish=AsyncMock())
                            for code in (502, manual_code)]
            commands = asyncio.Queue()
            await commands.put({"recovered": True})
            with tempfile.TemporaryDirectory() as directory:
                payload = {"timeout": 5, "profile": str(Path(directory) / "profile"),
                           "keyword": "测试", "user_agent": "test", "cookies": "session=old",
                           "screen": {"width": 2560, "height": 1440}}
                if render_error:
                    payload["verification"] = "original-ticket"
                with patch.object(worker, "async_playwright", return_value=manager), patch.object(worker, "Observations", side_effect=observations), patch.object(worker, "automatic", AsyncMock(return_value=False)), patch.object(worker, "challenge_frame", AsyncMock(return_value=None)), patch.object(worker, "present_http_challenge", AsyncMock(side_effect=[None, TimeoutError("private-ticket")])), patch.object(worker, "emit") as emit:
                    old_mask = worker.os.umask(0o077)
                    try:
                        await worker.run(payload, commands)
                    finally:
                        worker.os.umask(old_mask)
                self.assertEqual([call.kwargs["headless"] for call in launch.await_args_list], [True, False])
                self.assertEqual(launch.await_args_list[0].args, launch.await_args_list[1].args)
                headless.add_cookies.assert_awaited_once()
                headed.add_cookies.assert_awaited_once_with([{"name": "session", "value": "updated"}])
                headless.close.assert_awaited_once()
                headed.close.assert_awaited_once()
                sessions = [call for call in emit.call_args_list if call.args == ("session",)]
                expected_mode = "manual" if manual_code == 200 else "manual_session_refresh"
                self.assertEqual(sessions[0].kwargs, {"cookies": "session=updated", "mode": expected_mode, "verify_params": {}})
                report = next(Path(directory).glob("runs/*/report.json")).read_text()
                self.assertNotIn("session=", report)
                self.assertNotIn("private-ticket", report)
                if render_error:
                    self.assertEqual(json.loads(report)["manual_render_error"], "TimeoutError")
                self.assertEqual(json.loads(report)["handoffs"][0]["mode"], expected_mode)
        asyncio.run(exercise(200))
        asyncio.run(exercise(None))
        asyncio.run(exercise(200, render_error=True))


    def test_http_ticket_decodes_without_rendering_an_old_visible_challenge(self):
        async def exercise(encoded):
            obs = SimpleNamespace(generation=3)
            page = SimpleNamespace(wait_for_function=AsyncMock(), evaluate=AsyncMock())
            calls = 0
            async def wait(_):
                nonlocal calls
                calls += 1
                if calls == 2:
                    obs.generation += 1
            ticket = {"type": "verify", "subtype": "whirl", "detail": "private-ticket"}
            raw = json.dumps(ticket)
            if encoded:
                raw = worker.base64.b64encode(raw.encode()).decode()
            with patch.object(worker, "challenge_frame", AsyncMock(return_value=object())) as frame, patch.object(worker.asyncio, "sleep", AsyncMock(side_effect=wait)), patch.object(worker, "emit"):
                await worker.present_http_challenge(page, obs, raw)
                self.assertEqual(calls, 2)
                frame.assert_awaited_once()  # 旧题仍可见时，不提前返回。
            self.assertEqual(page.evaluate.await_args.args[1], ticket)
        asyncio.run(exercise(False))
        asyncio.run(exercise(True))

    def test_new_http_challenge_does_not_reset_automatic_attempt_budget(self):
        async def exercise():
            obs = SimpleNamespace(generation=1, searches=0, attempts=0, challenge={"mode": "slide", "id": "test"})
            frame = SimpleNamespace(locator=lambda _: SimpleNamespace(click=AsyncMock()))
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(worker, "challenge_frame", AsyncMock(return_value=frame)), patch.object(worker, "solve_slide", AsyncMock(side_effect=["accepted", "accepted", "rejected"])) as solve, patch.object(worker.asyncio, "sleep", AsyncMock()), patch.object(worker, "emit"):
                    self.assertEqual(await worker.automatic(None, None, obs, Path(directory)), "automatic")
                    obs.generation += 1
                    self.assertEqual(await worker.automatic(None, None, obs, Path(directory)), "automatic")
                    obs.generation += 1
                    self.assertFalse(await worker.automatic(None, None, obs, Path(directory)))
                    self.assertFalse(await worker.automatic(None, None, obs, Path(directory)))
                    self.assertEqual(solve.await_count, 3)
                    self.assertEqual(obs.attempts, 3)
                self.assertEqual(len(list(Path(directory).glob("attempt-*.json"))), 3)
        asyncio.run(exercise())

    def test_real_whirl_type_routes_to_rotation_solver(self):
        async def exercise():
            obs = SimpleNamespace(generation=1, searches=0, challenge={"mode": "whirl", "id": "test"})
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(worker, "challenge_frame", AsyncMock(return_value=object())), patch.object(worker, "solve_rotation", AsyncMock(return_value="accepted")) as rotate, patch.object(worker, "solve_slide", AsyncMock()) as slide, patch.object(worker.asyncio, "sleep", AsyncMock()), patch.object(worker, "emit"):
                    self.assertEqual(await worker.automatic(None, None, obs, Path(directory)), "automatic")
                    rotate.assert_awaited_once()
                    slide.assert_not_awaited()
                record = json.loads((Path(directory) / "attempt-1.json").read_text())
                self.assertEqual(record["kind"], "whirl")
        asyncio.run(exercise())

    def test_auto_only_failure_closes_video_without_opening_headed_browser(self):
        async def exercise():
            async def video_path():
                ctx.close.assert_awaited_once()
                return Path('/test/recording.webm')
            page = SimpleNamespace(goto=AsyncMock(), video=SimpleNamespace(path=video_path))
            ctx = SimpleNamespace(pages=[page], add_cookies=AsyncMock(), clear_cookies=AsyncMock(), close=AsyncMock())
            manager = AsyncMock()
            launch = AsyncMock(return_value=ctx)
            manager.__aenter__.return_value = SimpleNamespace(chromium=SimpleNamespace(launch_persistent_context=launch))
            obs = SimpleNamespace(challenge={'mode': 'text'}, searches=0, generation=1, attempts=0,
                                  verify_params={}, receipts=[], closed=False, finish=AsyncMock())
            with tempfile.TemporaryDirectory() as directory:
                payload = {'timeout': 5, 'profile': str(Path(directory) / 'profile'), 'verification': 'ticket',
                           'keyword': '测试', 'cookies': 'session=old', 'screen': {'width': 2560, 'height': 1440}}
                with patch.dict(os.environ, {'DOUYIN_CAPTCHA_AUTO_ONLY': 'true', 'DOUYIN_CAPTCHA_RECORD_VIDEO': 'true'}), patch.object(worker, 'async_playwright', return_value=manager), patch.object(worker, 'Observations', return_value=obs), patch.object(worker, 'challenge_frame', AsyncMock(return_value=None)), patch.object(worker, 'present_http_challenge', AsyncMock()), patch.object(worker, 'automatic', AsyncMock(return_value=False)), patch.object(worker, 'emit'):
                    old_mask = worker.os.umask(0o077)
                    try:
                        with self.assertRaisesRegex(RuntimeError, '未恢复'):
                            await worker.run(payload, asyncio.Queue())
                    finally:
                        worker.os.umask(old_mask)
                launch.assert_awaited_once()
                self.assertTrue(launch.await_args.kwargs['headless'])
                self.assertEqual(launch.await_args.kwargs['record_video_size'], {'width': 1320, 'height': 900})
                report = json.loads(next((Path(directory) / 'runs').glob('*/report.json')).read_text())
                self.assertTrue(report['auto_only_stopped'])
                self.assertTrue(report['videos'][0]['headless'])
                self.assertEqual(report['videos'][0]['path'], '/test/recording.webm')
        asyncio.run(exercise())

    def test_http_retry_renders_new_ticket_in_same_browser_and_same_budget(self):
        async def exercise():
            ctx = SimpleNamespace(pages=[SimpleNamespace(goto=AsyncMock())],
                add_cookies=AsyncMock(), clear_cookies=AsyncMock(), close=AsyncMock(),
                cookies=AsyncMock(return_value=[{"name": "session", "value": "updated"}]))
            manager = AsyncMock()
            launch = AsyncMock(return_value=ctx)
            manager.__aenter__.return_value = SimpleNamespace(chromium=SimpleNamespace(launch_persistent_context=launch))
            obs = SimpleNamespace(challenge={"mode": "whirl"}, searches=0, generation=1, attempts=0,
                verify_params={}, receipts=[], closed=False, finish=AsyncMock())
            async def solve(*args, **kwargs):
                self.assertTrue(kwargs["confirm_original"])
                obs.attempts += 1
                return "automatic"
            commands = asyncio.Queue()
            await commands.put({"recovered": False, "retry": True, "verification": "new-ticket"})
            await commands.put({"recovered": True})
            with tempfile.TemporaryDirectory() as directory:
                payload = {"timeout": 5, "profile": str(Path(directory) / "profile"), "verification": "first-ticket",
                           "keyword": "测试", "user_agent": "test", "cookies": "session=old",
                           "screen": {"width": 2560, "height": 1440}}
                with patch.object(worker, "async_playwright", return_value=manager), patch.object(worker, "Observations", return_value=obs), patch.object(worker, "challenge_frame", AsyncMock(return_value=None)), patch.object(worker, "present_http_challenge", AsyncMock()) as present, patch.object(worker, "automatic", AsyncMock(side_effect=solve)), patch.object(worker, "emit"):
                    old_mask = worker.os.umask(0o077)
                    try:
                        await worker.run(payload, commands)
                    finally:
                        worker.os.umask(old_mask)
                self.assertEqual([call.args[2] for call in present.await_args_list], ["first-ticket", "new-ticket"])
                self.assertEqual(obs.attempts, 2)
                launch.assert_awaited_once()
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
