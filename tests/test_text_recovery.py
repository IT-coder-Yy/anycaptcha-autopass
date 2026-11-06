import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from common.demo import refresh
from common.evaluation import summarize_reports
from common.interaction import can_retry
from common.text_ocr import choose_assignment, prepare


class TextSelectionTests(unittest.TestCase):
    def test_models_can_correct_an_unusable_baseline_but_disagreement_cannot(self):
        strong = {'assignment': [1, 0], 'score': .05, 'margin': 1.2}
        weak = {'assignment': [0, 1], 'score': .01, 'margin': .1}
        self.assertEqual(choose_assignment([strong, strong], [0, 1], False)[:2], ([1, 0], True))
        self.assertFalse(choose_assignment([strong, weak], [0, 1], False)[1])

    def test_agreement_on_numerical_noise_is_not_evidence(self):
        noise = {'assignment': [1, 0], 'score': 1e-10, 'margin': 4}
        self.assertFalse(choose_assignment([noise, noise], [0, 1], False)[1])

    def test_usable_baseline_needs_ocr_support(self):
        model = {'assignment': [1, 0], 'score': .05, 'margin': 1.2}
        other = {'assignment': [0, 1], 'score': .01, 'margin': .1}
        self.assertTrue(choose_assignment([model, other], [1, 0], True)[1])
        self.assertFalse(choose_assignment([model, other], [0, 1], True)[1])

    def test_bgr_channel_order_and_padding_match_model_contract(self):
        image = np.zeros((20, 10, 3), np.uint8)
        image[:, :, 0] = 255
        tensor = prepare(image)
        np.testing.assert_equal(tensor[:, 0, 0], [1, -1, -1])
        np.testing.assert_equal(tensor[:, :, 24:], 0)


class RecoveryTests(unittest.TestCase):
    def test_only_explicit_text_policy_allows_unanswered_recovery(self):
        entry = {'attempt_id': 'a', 'outcome': 'low_confidence'}
        self.assertFalse(can_retry(entry))
        self.assertTrue(can_retry(entry, allow_unanswered=True))
        for outcome in ('unknown', 'stale_challenge', 'passed', 'failed'):
            self.assertFalse(can_retry(entry | {'outcome': outcome}, allow_unanswered=True))

    def test_rates_keep_unanswered_and_failed_recovery_in_denominators(self):
        def case(outcomes, verdict):
            return {'attempts': [{'outcome': o} for o in outcomes], 'verdict': verdict, 'refreshes': []}
        summary = summarize_reports([case(['passed'], 'passed'), case(['low_confidence', 'passed'], 'passed'),
                                     case(['failed', 'low_confidence', 'failed'], 'failed'), case([], 'unknown')])
        self.assertEqual((summary['cases'], summary['first_attempt_passed'], summary['passed']), (4, 1, 2))
        self.assertEqual((summary['first_not_passed_cases'], summary['recovered_cases']), (3, 1))
        self.assertEqual((summary['retry_attempts'], summary['retry_passed']), (3, 1))
        self.assertEqual(summary['by_attempt']['2'], {'passed': 1, 'low_confidence': 1})
        self.assertEqual(summary['first_attempt_outcomes']['unknown'], 1)

    def test_old_reports_do_not_invent_zero_automatic_refreshes(self):
        summary = summarize_reports([{'attempts': [], 'verdict': 'unknown'}])
        self.assertIsNone(summary['refresh_sources'])


class RefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_site_refresh_is_reused_without_clicking_refresh(self):
        page = AsyncMock()
        with patch('common.demo.source_id', AsyncMock(side_effect=['old', 'old', 'new'])):
            event = await refresh(page, 'kg', 'text', 'old', wait_for_site=True)
        self.assertEqual(event['source'], 'site')
        self.assertEqual(event['challenge_id'], 'new')
        page.locator.assert_not_called()

    async def test_program_refresh_is_counted_after_unanswered_question(self):
        class Page:
            wait_for_timeout = AsyncMock()
            def locator(self, selector): return self.button
            button = AsyncMock()
        page = Page()
        with patch('common.demo.source_id', AsyncMock(side_effect=['old', 'new', 'new', 'new'])):
            event = await refresh(page, 'kg', 'text', 'old')
        self.assertEqual(event['source'], 'program')
        page.button.click.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
