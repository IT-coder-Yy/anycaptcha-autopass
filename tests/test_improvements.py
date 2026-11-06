import asyncio
import unittest
import cv2
import numpy as np
from common.text_support import choose_anchor, constrain_scores
from common.vision import best_assignment
from rotate_overlap.solver import align_overlap
from rotate_overlap.demo import rotation_verdict, RotationReceipts
from text.demo import normalize_prompt


class OCRTests(unittest.TestCase):
    def test_hidden_prompt_whitespace_is_not_new_challenge(self):
        self.assertEqual(normalize_prompt(' "扩" "类" "库"'),normalize_prompt('"扩" "类" "库"'))
        self.assertNotEqual(normalize_prompt('"扩" "类" "库"'),normalize_prompt('"库" "类" "扩"'))
    def test_one_angle_or_conflicting_character_does_not_force_match(self):
        def reading(char,score):return {'text':char,'score':score,'angle':0}
        self.assertIsNone(choose_anchor([reading('特',.99)])['trusted'])
        self.assertIsNone(choose_anchor([reading('特',.99),reading('特',.90),reading('体',.97)])['trusted'])
        self.assertEqual(choose_anchor([reading('特',.99),reading('特',.90)])['trusted'],'特')

    def test_positive_anchor_constrains_row_and_column(self):
        constrained=constrain_scores([[.8,.99],[.7,.95]],'特体',[{'trusted':'特','score':.98},{'trusted':None}])
        self.assertEqual(best_assignment(constrained)[0],[0,1])

    def test_duplicate_anchors_remain_ambiguous(self):
        constrained=constrain_scores([[.8,.9]],'特',[{'trusted':'特','score':.98},{'trusted':'特','score':.98}])
        self.assertAlmostEqual(best_assignment(constrained)[2],0)


class RotationTests(unittest.TestCase):
    def test_clockwise_mapping_with_shared_crop(self):
        rng=np.random.default_rng(12)
        image=cv2.GaussianBlur(rng.integers(0,256,(220,220,3),dtype=np.uint8),(9,9),0)
        crop=image[40:180,40:180]
        matrix=cv2.getRotationMatrix2D((69.5,69.5),73,1)
        piece=cv2.warpAffine(crop,matrix,(140,140))
        result=align_overlap(image,piece)
        self.assertTrue(result.usable)
        self.assertLess(abs(result.value-73),1)

    def test_uniform_image_has_no_valid_angle(self):
        image=np.ones((220,220,3),np.uint8)*100
        self.assertFalse(align_overlap(image,image[40:180,40:180]).usable)

    def test_server_and_current_ui_both_required(self):
        good={'accepted':True,'challenge_matched':True}
        self.assertEqual(rotation_verdict(good,True),'passed')
        self.assertEqual(rotation_verdict(good,False),'unknown')
        self.assertEqual(rotation_verdict({'accepted':False},True),'failed')
        self.assertEqual(rotation_verdict({'accepted':True,'challenge_matched':False},True),'unknown')


class RotationReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_response_keeps_attempt_and_rejects_wrong_or_missing_key(self):
        class Page:
            def on(self,*args):pass
        class Request:
            url='http://47.104.180.148:8081/api/go-captcha-check-data/rotate-basic'
            post_data='angle=40&key=fixture'
        class Response:
            url=Request.url
            status=200
            async def json(self):return {'code':0}
        for key,expected in [('fixture',True),('other',None),(None,None)]:
            receipts=RotationReceipts(Page());receipts.active='old'
            receipts.challenge={'key':key,'host':'47.104.180.148:8081'}
            request=Request();receipts.request(request);receipts.active='new'
            response=Response();response.request=request
            await receipts.response(response)
            self.assertEqual(receipts.results[0]['attempt_id'],'old')
            self.assertIs(receipts.results[0]['accepted'],expected)


if __name__=='__main__':unittest.main()
