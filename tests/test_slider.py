import unittest
from unittest.mock import patch

import cv2
import numpy as np

from slider.solver import outline_candidate, settings


def fixture(origins=(140,), vertical=0):
    """独立构造已知位移的拼图片；缺口填色与碎片内容不同。"""
    mask = np.zeros((100, 54), np.uint8)
    cv2.rectangle(mask, (12, 32), (40, 62), 255, -1)
    cv2.circle(mask, (26, 32), 7, 255, -1)
    cv2.circle(mask, (40, 48), 7, 255, -1)
    cv2.circle(mask, (26, 62), 6, 0, -1)
    piece = np.zeros((100, 54, 4), np.uint8)
    piece[:, :, :3] = np.random.default_rng(13).integers(30, 190, (100,54,3), dtype=np.uint8)
    piece[:, :, 3] = mask
    bg = np.full((120, 260, 3), 70, np.uint8)
    for x in origins:
        canvas=bg[vertical:vertical+100, x:x+54]
        canvas[mask > 0]=210
    return bg, piece


class OutlineTests(unittest.TestCase):
    def test_erased_texture_still_locates_known_gap(self):
        bg, piece = fixture()
        result = outline_candidate(bg, piece)
        self.assertTrue(result.usable)
        self.assertLessEqual(abs(result.value-140), 1)

    def test_geometry_origin_changes_displacement_not_target(self):
        bg, piece = fixture(vertical=8)
        result = outline_candidate(bg, piece, cur_x=18, cur_y=8)
        self.assertTrue(result.usable)
        self.assertLessEqual(abs(result.value-122), 1)

    def test_two_identical_gaps_are_ambiguous(self):
        bg, piece = fixture((60, 160))
        self.assertFalse(outline_candidate(bg, piece).usable)

    def test_no_gap_and_no_visible_piece_are_rejected(self):
        bg, piece = fixture(())
        self.assertFalse(outline_candidate(bg, piece).usable)
        piece[:, :, 3] = 0
        self.assertFalse(outline_candidate(bg, piece).usable)

    def test_cannot_reach_gap_on_other_row_or_left_of_origin(self):
        bg, piece = fixture(vertical=16)
        self.assertFalse(outline_candidate(bg, piece).usable)
        bg, piece = fixture((40,))
        self.assertFalse(outline_candidate(bg, piece, cur_x=100).usable)

    def test_invalid_settings_do_not_silently_change_engine(self):
        for values in ({'CAPTCHA_SLIDER_ENGINE':'typo'}, {'CAPTCHA_SLIDER_OUTLINE_MIN':'nan'},
                       {'CAPTCHA_SLIDER_OUTLINE_MARGIN':'0'}):
            with patch.dict('os.environ', values):
                with self.assertRaises(ValueError):settings()


if __name__ == '__main__': unittest.main()
