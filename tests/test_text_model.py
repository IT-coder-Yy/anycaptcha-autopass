import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from text.douyin import recognize


@unittest.skipUnless(os.getenv('CAPTCHA_MODEL_TEST') == '1', '需安装点字权重并显式启用模型回归')
class TextModelTests(unittest.TestCase):
    def test_small_hint_image_produces_correct_character_order(self):
        """真实模型回归：小提示图的留白不能使字符失检或改变匹配顺序。"""
        fixtures = ROOT / 'tests/fixtures'
        background = cv2.imread(str(fixtures / 'text-model-background.png'))
        hint = cv2.imread(str(fixtures / 'text-model-hint.png'))
        with patch.dict(os.environ, {'CAPTCHA_TEXT_THRESHOLD': '.8', 'CAPTCHA_TEXT_AMBIGUITY': '.08',
                                     'CAPTCHA_TEXT_ENGINE': 'siamese'}):
            result = recognize(background, hint, ROOT / 'models')
        self.assertTrue(result.usable, result.reason)
        self.assertEqual(len(result.points), 2)
        for (x, y), (left, top, right, bottom) in zip(result.points, [(30, 20, 85, 85), (200, 10, 255, 75)]):
            self.assertTrue(left < x < right and top < y < bottom, result.points)
