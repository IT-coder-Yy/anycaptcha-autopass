"""独立题库留出样本、判定边界与一次性服务端校验。"""
import base64
import random
import unittest

import cv2
import numpy as np

from restore.solver import restore_band
from drag2d.solver import drag_piece
from icon.solver import click_icons
from common.geometry_demo import verdict, Receipts
from lab.server import ChallengeStore, generate


def decode(source):
    return cv2.imdecode(np.frombuffer(base64.b64decode(source.split(',')[1]),np.uint8),-1)


class GeometryTests(unittest.TestCase):
    def test_heldout_images_have_correct_coordinates(self):
        for kind in ('restore','drag2d','icon'):
            usable=correct=0
            for seed in range(101,121):
                p,answer=generate(kind,random.Random(seed))
                bg=decode(p['background'])
                prediction=(restore_band(bg,p['band']) if kind=='restore' else
                            drag_piece(bg,decode(p['piece'])) if kind=='drag2d' else click_icons(bg,decode(p['hint'])))
                if not prediction.usable:continue
                usable+=1
                if kind=='restore':match=abs(prediction.value-answer['shift'])<=3
                elif kind=='drag2d':match=np.linalg.norm(np.array(prediction.points[0])-answer['point'])<=4
                else:match=len(prediction.points)==len(answer['points']) and all(
                    np.linalg.norm(np.array(a)-b)<=12 for a,b in zip(prediction.points,answer['points']))
                correct+=bool(match)
            self.assertGreaterEqual(usable,16,(kind,usable))
            self.assertEqual(correct,usable,(kind,usable,correct))

    def test_flat_restore_and_missing_seam_abstain(self):
        image=np.full((100,180,3),128,np.uint8)
        self.assertFalse(restore_band(image,[0,25,180,50]).usable)
        self.assertFalse(restore_band(image,[0,0,180,100]).usable)
        self.assertFalse(restore_band(image,[0,0,200,20]).usable)

    def test_repeated_stripes_restore_abstains(self):
        image=np.tile(np.arange(180)%10,(100,1)).astype(np.uint8)*25
        image=np.repeat(image[:,:,None],3,axis=2)
        image[20:60]=np.roll(image[20:60],3,axis=1)
        self.assertFalse(restore_band(image,[0,20,180,40]).usable)

    def test_drag_rejects_missing_alpha_or_empty_piece(self):
        self.assertFalse(drag_piece(np.zeros((100,160,3),np.uint8),np.zeros((25,25,3),np.uint8)).usable)
        self.assertFalse(drag_piece(np.zeros((100,160,3),np.uint8),np.zeros((25,25,4),np.uint8)).usable)
        piece=np.zeros((110,170,4),np.uint8);piece[10:40,10:40,3]=255
        self.assertFalse(drag_piece(np.zeros((100,160,3),np.uint8),piece).usable)

    def test_duplicate_icons_do_not_get_arbitrary_order(self):
        background=np.full((160,300,3),255,np.uint8)
        for x in (40,150):cv2.rectangle(background,(x,40),(x+40,80),(40,90,180),-1)
        hint=np.full((60,100,3),255,np.uint8);cv2.rectangle(hint,(20,10),(60,50),(180,80,30),-1)
        self.assertFalse(click_icons(background,hint).usable)

    def test_empty_icon_abstains(self):
        self.assertFalse(click_icons(np.full((160,300,3),255,np.uint8),np.full((40,150,3),255,np.uint8)).usable)

    def test_partial_hint_segmentation_does_not_submit_short_answer(self):
        background=np.full((160,300,3),255,np.uint8)
        cv2.circle(background,(80,70),20,(30,60,150),-1)
        hint=np.full((60,150,3),255,np.uint8)
        cv2.circle(hint,(5,30),14,(180,70,30),-1)
        cv2.circle(hint,(105,30),14,(90,140,60),-1)
        result=click_icons(background,hint)
        self.assertFalse(result.usable)
        self.assertEqual(result.detail['foreground_hint_count'],2)


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self.receipt={'attempt_id':'a','challenge_key':'k','challenge_matched':True,'accepted':True}

    def test_success_requires_current_receipt_and_ui(self):
        self.assertEqual(verdict(self.receipt,'a','k','passed'),'passed')
        for receipt,attempt,key,ui,current in [(None,'a','k','passed',True),
                (self.receipt,'b','k','passed',True),(self.receipt,'a','old','passed',True),
                (self.receipt,'a','k','',True),(self.receipt,'a','k','passed',False),
                (dict(self.receipt,challenge_matched=False),'a','k','passed',True)]:
            self.assertEqual(verdict(receipt,attempt,key,ui,current),'unknown')

    def test_rejection_overrides_success_text(self):
        self.assertEqual(verdict(dict(self.receipt,accepted=False),'a','k','passed'),'failed')

    def test_http_200_unknown_payload_is_not_accepted(self):
        import asyncio
        class Page:
            def on(self,*args):pass
        class Request:
            url='http://127.0.0.1/verify';post_data='{"key":"k","answer":{}}'
        class Response:
            url=Request.url;status=200
            async def json(self):return {'key':'k','accepted':1}
        book=Receipts(Page(),'http://127.0.0.1','/challenge','/verify',True)
        book.challenge={'key':'k'};book.active=('a','k');request=Request();book.request(request)
        response=Response();response.request=request
        asyncio.run(book.response(response))
        self.assertIsNone(book.results[0]['accepted'])
        self.assertEqual(verdict(book.results[0],'a','k','passed'),'unknown')

    def test_wrong_origin_or_unarmed_request_cannot_bind(self):
        class Page:
            def on(self,*args):pass
        class Request:
            url='http://other.test/verify';post_data='{"key":"k"}'
        book=Receipts(Page(),'http://127.0.0.1','/challenge','/verify',True)
        book.challenge={'key':'k'};book.active=('a','k');book.request(Request())
        self.assertFalse(book.requests)
        book.active=None;request=Request();request.url='http://127.0.0.1/verify';book.request(request)
        self.assertFalse(book.requests)

    def test_remote_service_error_is_not_an_answer_rejection(self):
        import asyncio
        class Page:
            def on(self,*args):pass
        class Request:
            url='http://demo.test/check';post_data='key=k&point=10,20'
        class Response:
            url=Request.url;status=200
            async def json(self):return {'code':403,'message':'服务不可用'}
        book=Receipts(Page(),'http://demo.test','/data','/check')
        book.challenge={'key':'k'};book.active=('a','k');request=Request();book.request(request)
        response=Response();response.request=request
        asyncio.run(book.response(response))
        self.assertIsNone(book.results[0]['accepted'])


class StoreTests(unittest.TestCase):
    def test_public_payload_has_no_answers_and_tokens_are_one_use(self):
        store=ChallengeStore(5)
        for kind in ('restore','drag2d','icon'):
            p=store.issue(kind)
            self.assertFalse(set(p)&{'answer','points','point','shift','seed'})
            answer=store.answers[p['key']][2]
            submission={'shift':answer['shift']} if kind=='restore' else answer
            self.assertTrue(store.verify(p['key'],submission)['accepted'])
            self.assertFalse(store.verify(p['key'],submission)['accepted'])

    def test_wrong_answers_and_malformed_values_fail(self):
        store=ChallengeStore(50)
        for answer in ({'shift':float('nan')},{'shift':float('inf')},{'shift':[]},{}):
            p=store.issue('restore');self.assertFalse(store.verify(p['key'],answer)['accepted'])
        p=store.issue('icon');answer=store.answers[p['key']][2]
        self.assertFalse(store.verify(p['key'],{'points':list(reversed(answer['points']))})['accepted'])
        self.assertFalse(store.verify('unknown',{})['accepted'])


if __name__=='__main__':unittest.main()
