"""第二批的独立裁判、方向约束、词序歧义和 CLI 分发边界。"""
import math
import random
import unittest
from unittest.mock import patch
import numpy as np
from lab.server import ChallengeStore, generate
from rotate_single.solver import upright
from common.vision import Prediction
from order.solver import order_words, rank_words


class SecondBatchTests(unittest.TestCase):
    def test_private_answers_do_not_appear_in_public_challenge(self):
        for kind in ('rotate-single','font','order'):
            public,answer=generate(kind,random.Random(1234))
            self.assertTrue(answer)
            self.assertFalse(set(public)&{'angle','source','points','word','font','odd','seed'})
            self.assertTrue(public['background'].startswith('data:image/'))

    def test_rotation_wrap_tolerance_and_single_use(self):
        store=ChallengeStore()
        for value,expected in [(359,True),(0,True),(7,True),(9,False),(180,False),(math.nan,False),(math.inf,False)]:
            public=store.issue('rotate-single');key=public['key']
            expires,kind,answer=store.answers[key];store.answers[key]=(expires,kind,{'angle':0})
            self.assertEqual(store.verify(key,{'angle':value})['accepted'],expected)
            self.assertFalse(store.verify(key,{'angle':0})['accepted'])

    def test_click_judge_rejects_wrong_order_count_and_nonfinite(self):
        for kind in ('font','order'):
            for invalid in ('count','wrong','nan'):
                store=ChallengeStore(22);public=store.issue(kind);answer=store.answers[public['key']][2]
                points=[p[:] for p in answer['points']]
                if invalid=='count':points=[]
                elif invalid=='wrong':points[0]=[0,0]
                else:points[0][0]=math.nan
                self.assertFalse(store.verify(public['key'],{'points':points})['accepted'])
            store=ChallengeStore(22);public=store.issue(kind)
            self.assertTrue(store.verify(public['key'],store.answers[public['key']][2])['accepted'])

    def test_rotation_consistency_and_mass_are_both_required(self):
        image=np.zeros((40,40,3),np.uint8)
        for angle,mass,accepted in [(120,.9,True),(300,.9,False),(120,.4,False)]:
            class Model:
                def __init__(self):self.calls=0
                def rotate_single(self,im):
                    self.calls+=1
                    return Prediction('rotate_single',False,value=30 if self.calls==1 else angle,
                                      detail={'mass_within_8_degrees':mass})
            self.assertEqual(upright(Model(),image).usable,accepted)

    def test_complete_unique_word_and_positions_required(self):
        boxes=[[0,0,10,10],[20,0,30,10],[40,0,50,10],[60,0,70,10]]
        evidence=[{'花':.99},{'春':.99},{'开':.99},{'暖':.99}]
        image=np.zeros((80,80,3),np.uint8)
        with patch('order.solver.detect_characters',return_value=boxes),patch('order.solver.readings',return_value=[evidence,evidence]),patch('order.solver.vocabulary',return_value=('春暖花开',)):
            p=order_words(None,image)
            self.assertTrue(p.usable)
            self.assertEqual(p.points,[[25,5],[65,5],[5,5],[45,5]])
        self.assertFalse(rank_words([{'春':.99},{'暖':.99},{'花':.99},{'开':.99},{'余':.99}],['春暖花开']))

    def test_anagram_and_model_disagreement_abstain(self):
        boxes=[[x,0,x+8,8] for x in (0,10,20,30)];image=np.zeros((50,50,3),np.uint8)
        a=[{c:.99} for c in '望梅止渴']
        with patch('order.solver.detect_characters',return_value=boxes),patch('order.solver.readings',return_value=[a,a]),patch('order.solver.vocabulary',return_value=('望梅止渴','止渴望梅')):
            self.assertFalse(order_words(None,image).usable)
        b=[{c:.99} for c in '梅望止渴']
        with patch('order.solver.detect_characters',return_value=boxes),patch('order.solver.readings',return_value=[a,b]),patch('order.solver.vocabulary',return_value=('望梅止渴',)):
            self.assertFalse(order_words(None,image).usable)
