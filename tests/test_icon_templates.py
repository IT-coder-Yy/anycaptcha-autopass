"""独立渲染的旋转、缩放与遮挡图；断言坐标及弃权，不用在线题目答案。"""
import unittest

import cv2
import numpy as np
from PIL import Image

from icon.templates import ASSETS, click_gocaptcha_icons, template_info


def stamp(canvas, shape, center, size, angle, color):
    source = Image.open(ASSETS / f'shape-{shape}.png').getchannel('A')
    source = source.crop(source.getbbox()).rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    source = source.crop(source.getbbox())
    source.thumbnail((size, size), Image.Resampling.LANCZOS)
    alpha = np.asarray(source).astype(float)/255
    h, w = alpha.shape
    x, y = round(center[0]-w/2), round(center[1]-h/2)
    patch = canvas[y:y+h, x:x+w]
    patch[:, :, :3] = np.rint(patch[:, :, :3]*(1-alpha[:, :, None])+np.array(color)*alpha[:, :, None])
    if canvas.shape[2] == 4:
        patch[:, :, 3] = np.rint(alpha*255)
        # 透明图片保留实色通道，模拟 PNG 的直通 alpha。
        patch[:, :, :3] = color


def scene(shapes=(5, 9, 6), occluded=True):
    rng = np.random.default_rng(905)
    background = rng.integers(160, 205, (180, 300, 3), dtype=np.uint8)
    hint = np.zeros((40, 160, 4), np.uint8)
    centers = [(55, 55), (150, 100), (245, 65)]
    for index, shape in enumerate(shapes):
        stamp(background, shape, centers[index], 42, 17+index*48, (35, 70, 125))
        stamp(hint, shape, (25+index*53, 20), 28, 113-index*39, (180, 60, 30))
    if occluded:
        cv2.line(hint, (0, 16), (159, 25), (30, 160, 100, 255), 1)
    return background, hint, centers


class IconTemplateTests(unittest.TestCase):
    def test_assets_are_pinned_and_verified(self):
        info = template_info()
        self.assertEqual(info['revision'], '4da532768f57f0116cd6161d8f1bca93584dedd1')
        self.assertEqual(len(info['templates_sha256']), 64)

    def test_rotation_scale_disconnected_petals_and_crossing_line(self):
        for occluded in (False, True):
            background, hint, centers = scene(occluded=occluded)
            prediction = click_gocaptcha_icons(background, hint)
            self.assertTrue(prediction.usable, prediction.to_dict())
            self.assertEqual(prediction.detail['classes'], [5, 9, 6])
            for actual, expected in zip(prediction.points, centers):
                self.assertLess(np.linalg.norm(np.array(actual)-expected), 4)

    def test_duplicate_targets_abstain_instead_of_guessing_order(self):
        background, hint, _ = scene((5, 5, 6), occluded=False)
        prediction = click_gocaptcha_icons(background, hint)
        self.assertFalse(prediction.usable)

    def test_missing_target_and_empty_hint_abstain(self):
        background, hint, _ = scene()
        background[30:85, 30:85] = 190
        self.assertFalse(click_gocaptcha_icons(background, hint).usable)
        self.assertFalse(click_gocaptcha_icons(background, np.zeros_like(hint)).usable)

    def test_partial_hint_cannot_become_short_answer(self):
        background, hint, _ = scene()
        hint[:, :50] = 0
        hint[10:25, 10:25] = (80, 120, 50, 255)  # 未知方块不应强行映射到已有类别。
        self.assertFalse(click_gocaptcha_icons(background, hint).usable)

    def test_wrong_profile_inputs_abstain(self):
        background, hint, _ = scene()
        for invalid in (None, hint[:, :, :3], np.zeros((10, 10, 4), np.uint8)):
            self.assertFalse(click_gocaptcha_icons(background, invalid).usable)


if __name__ == '__main__':
    unittest.main()
