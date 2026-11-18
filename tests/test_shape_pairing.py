from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
import unittest
from unittest.mock import patch

from cli import main
from lab.choices import bank
from lab.server import ChallengeStore, local_server
from shape_pairing.lab import masks
from shape_pairing.solver import match


class ShapePairingTests(unittest.TestCase):
    def test_cli_reproduces_fixed_boards_and_matches_the_entire_bank(self):
        fingerprint = hashlib.sha256()
        for case, rows in bank("shape-pairing").items():
            fingerprint.update((case + rows[0]["title"]).encode())
            fingerprint.update(bytes(int(row["slot"]) - 1 for row in rows))
            pieces, slots = masks(rows)
            for mask in pieces + slots:
                fingerprint.update(mask.tobytes())
        self.assertEqual(fingerprint.hexdigest(), "c23874a43cb50526cbf8404c8513d2b9957d6a81388358f374a299c09cb90bd3")

        out = StringIO()
        with patch("sys.argv", ["captcha-local", "recognize", "--kind", "shape-pairing"]), redirect_stdout(out):
            main()
        result = json.loads(out.getvalue())
        self.assertEqual((result["correct"], result["total"], result["pieces_correct"], result["pieces_total"]),
                         (16, 16, 48, 48))
        self.assertEqual(next(s["selected"] for s in result["samples"] if s["id"] == "G14"), [3, 1, 2])

    def test_identical_shapes_only_hide_ambiguous_positions(self):
        (a, b), _ = masks(bank("shape-pairing")["G01"])
        self.assertEqual(match([a, a, b], [a, b, a]), [None, None, 2])

    def test_local_form_checks_slot_order_and_uses_each_key_once(self):
        for answer, accepted in (("132", False), ("31", False), ("312", True)):
            store = ChallengeStore(seed=17)  # G14
            public = store.issue("shape-pairing")
            self.assertEqual(set(public), {"kind", "key", "width", "height", "instruction", "pieces", "slots"})
            self.assertEqual(len(public["pieces"]), len(public["slots"]))
            for src in public["pieces"] + public["slots"]:
                self.assertTrue(src.startswith("data:image/png;base64,"))
            key = public["key"]
            self.assertEqual(store.verify(key, {"text": answer}), {"key": key, "accepted": accepted})
            self.assertFalse(store.verify(key, {"text": "312"})["accepted"])

    @unittest.skipUnless(os.getenv("CAPTCHA_BROWSER_TEST") == "1", "需显式启用本地 Chromium 测试")
    def test_drag_click_keyboard_and_reassignment(self):
        from playwright.sync_api import expect, sync_playwright

        with local_server(seed=17) as url, sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(f"{url}?kind=shape-pairing")
            confirm = page.get_by_role("button", name="确认")
            a, b, c = [page.get_by_label(f"拼块 {letter}", exact=True) for letter in "ABC"]
            one, two, three = [page.get_by_label(f"槽位 {i}", exact=True) for i in (1, 2, 3)]
            expect(confirm).to_be_disabled()
            a.drag_to(one)
            expect(one).to_have_text("1 ← A")
            b.click()
            one.click()
            expect(one).to_have_text("1 ← B")
            c.press("Enter")
            two.press("Enter")
            expect(confirm).to_be_disabled()
            a.drag_to(three)
            expect(confirm).to_be_enabled()
            key = page.locator("#stage").get_attribute("data-key")
            with page.expect_response(url + "verify") as receipt:
                confirm.click()
            self.assertEqual(receipt.value.json(), {"key": key, "accepted": True})
            expect(page.get_by_role("status")).to_have_text("验证成功")
            expect(confirm).to_be_disabled()
            page.get_by_role("button", name="换一题").click()
            expect(one).to_have_text("1")
            expect(confirm).to_be_disabled()
            browser.close()
