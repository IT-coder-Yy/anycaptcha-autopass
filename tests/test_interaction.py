import asyncio
import math
import random
import time
import unittest

from common.interaction import Interaction, can_retry, timed_path


class RetryTests(unittest.TestCase):
    def test_low_score_unknown_and_stale_never_select_another_challenge(self):
        for outcome in ('low_confidence', 'unknown', 'stale_challenge', 'passed'):
            self.assertFalse(can_retry({'attempt_id': 'a', 'outcome': outcome,
                                       'receipts': [{'attempt_id': 'a', 'accepted': False}]}))

    def test_only_current_server_rejection_allows_recovery(self):
        entry = {'attempt_id': 'a', 'outcome': 'failed'}
        self.assertFalse(can_retry(entry))
        self.assertFalse(can_retry(entry | {'receipts': [{'attempt_id': 'old', 'accepted': False}]}))
        self.assertTrue(can_retry(entry | {'receipts': [{'attempt_id': 'a', 'accepted': False}]}))


class TrajectoryTests(unittest.TestCase):
    def test_curve_stays_inside_target_surface_and_ends_exactly(self):
        for seed in range(20):
            points = timed_path((1, 1), (99, 79), .6, random.Random(seed), (0, 0, 100, 80))
            self.assertEqual(points[-1], (99, 79, .6))
            self.assertTrue(all(0 <= x <= 100 and 0 <= y <= 80 for x, y, _ in points))
            self.assertTrue(all(a[2] < b[2] for a, b in zip(points, points[1:])))

    def test_drag_has_no_sideways_overshoot_and_varies_speed(self):
        points = [(0, 10, 0), *timed_path((0, 10), (200, 10), .8, random.Random(1), straight=True)]
        self.assertTrue(all(math.isclose(y, 10) for _, y, _ in points))
        self.assertTrue(all(a[0] <= b[0] for a, b in zip(points, points[1:])))
        speeds = [(b[0]-a[0])/(b[2]-a[2]) for a, b in zip(points, points[1:])]
        self.assertGreater(speeds[len(speeds)//2], speeds[0]*5)
        self.assertGreater(speeds[len(speeds)//2], speeds[-1]*5)


class FakeMouse:
    def __init__(self):
        self.points = []
        self.held = False
    async def move(self, x, y):
        self.points.append((x, y, time.monotonic()))
    async def down(self):
        self.held = True
    async def up(self):
        self.held = False


class FakePage:
    def __init__(self): self.mouse = FakeMouse()
    async def evaluate(self, script): return None


class TimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_events_are_really_spread_over_time(self):
        page = FakePage()
        interaction = Interaction(page, scale=1, seed=4)
        start = time.monotonic()
        await interaction.move(240, 60)
        record = interaction.actions[-1]
        self.assertGreaterEqual(time.monotonic()-start, record['planned_ms']/1000-.003)
        self.assertGreater(page.mouse.points[-1][2]-page.mouse.points[0][2], .4)
        self.assertEqual(page.mouse.points[-1][:2], (240, 60))

    async def test_cancelled_click_releases_mouse(self):
        page = FakePage()
        interaction = Interaction(page, scale=1, seed=4)
        task = asyncio.create_task(interaction.click())
        await asyncio.sleep(.01)
        self.assertTrue(page.mouse.held)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertFalse(page.mouse.held)


if __name__ == '__main__': unittest.main()
