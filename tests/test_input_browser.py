import os
import random
import unittest
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright
from lab.printed import samples
from lab.server import local_server


@unittest.skipUnless(os.getenv("CAPTCHA_BROWSER_TEST") == "1", "需显式启用 Chromium 测试")
class InputBrowserTests(unittest.TestCase):
    def test_mixed_case_submission_follows_the_hint(self):
        cases = (
            (31, "照图输入字符。区分大小写。", "0ak09", False),
            (25, "照图输入字符。", "aM42", True),
            (44, "照图输入字符。不区分大小写。", "70Bk", True),
        )
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for seed, hint, submitted, accepted in cases:
                with self.subTest(seed=seed), local_server(seed) as url:
                    page = browser.new_page()
                    page.goto(f"{url}?kind=normal")
                    expect(page.locator("#instruction")).to_have_text(hint)
                    key = page.locator("#stage").get_attribute("data-key")
                    page.get_by_label("答案", exact=True).fill(submitted)
                    with page.expect_response(lambda r: urlsplit(r.url).path == "/verify") as response:
                        page.get_by_role("button", name="提交").click()
                    self.assertEqual(response.value.json(), {"key": key, "accepted": accepted})
                    expect(page.get_by_role("status")).to_have_text("验证成功" if accepted else "验证失败")
                    page.close()
            browser.close()

    def test_rejection_refresh_and_acceptance_match_current_receipts(self):
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for kind in ("normal", "arithmetic"):
                with self.subTest(kind=kind), local_server(seed=1) as url:
                    draw = random.Random(1)
                    draw.choice(samples(kind))
                    second_answer = draw.choice(samples(kind)).answer
                    page = browser.new_page()
                    page.goto(f"{url}?kind={kind}")
                    field = page.get_by_label("答案", exact=True)
                    status = page.get_by_role("status")
                    stage = page.locator("#stage")
                    expect(field).to_be_visible()
                    first_key = stage.get_attribute("data-key")
                    field.fill("0")
                    with page.expect_response(lambda r: urlsplit(r.url).path == "/verify") as first:
                        field.press("Enter")
                    self.assertEqual(first.value.json(), {"key": first_key, "accepted": False})
                    expect(status).to_have_attribute("data-state", "failed")
                    page.get_by_role("button", name="换一题").click()
                    expect(field).to_have_value("")
                    second_key = stage.get_attribute("data-key")
                    self.assertNotEqual(first_key, second_key)
                    field.fill(second_answer)
                    with page.expect_response(lambda r: urlsplit(r.url).path == "/verify") as second:
                        page.get_by_role("button", name="提交").click()
                    self.assertEqual(second.value.json(), {"key": second_key, "accepted": True})
                    expect(status).to_have_attribute("data-key", second_key)
                    expect(status).to_have_attribute("data-state", "passed")
                    page.close()
            browser.close()

    def test_old_receipt_leaves_new_form_unconfirmed(self):
        with local_server(seed=1) as url, sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(f"{url}?kind=restore")
            expect(page.locator("#rail")).to_be_visible()
            expect(page.locator("#answer-form")).to_be_hidden()
            page.goto(f"{url}?kind=normal")
            page.route("**/verify", lambda route: route.fulfill(json={"accepted": True, "key": "old"}))
            page.get_by_label("答案", exact=True).fill("0073")
            page.get_by_role("button", name="提交").click()
            status = page.get_by_role("status")
            expect(status).to_have_text("回执不属于当前题目")
            expect(status).to_have_attribute("data-state", "")
            browser.close()
