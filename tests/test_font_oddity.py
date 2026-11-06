import unittest
from unittest.mock import patch
import cv2
import numpy as np
from font.solver import color_mask, ink_mask, odd_font, profile_vote

class FontOddityTests(unittest.TestCase):
    def test_faint_glyph_survives_bright_crossing_line(self):
        im=np.full((60,60,3),210,np.uint8)
        cv2.rectangle(im,(15,10),(21,49),(180,180,180),-1)
        cv2.rectangle(im,(21,20),(43,26),(180,180,180),-1)
        cv2.line(im,(0,5),(59,55),(0,0,255),1)
        mask=color_mask(im)
        self.assertGreater(mask[12:19,16:20].mean(),.9)
        self.assertLess(ink_mask(im)[12:19,16:20].mean(),.1)

    def test_single_quantile_and_conflicting_profiles_abstain(self):
        a=np.ones((6,4));a[0,3]=3
        self.assertEqual(profile_vote(a)[0],-1)
        a[0,:2]=3;a[0,3]=1;a[1,2:]=3
        self.assertEqual(profile_vote(a)[0],-1)
        a=np.ones((6,4));a[2]=3
        self.assertEqual(profile_vote(a)[0],2)

    def test_old_ratio_and_gap_are_not_relaxed(self):
        a=np.ones((6,4));a[0]=1.7
        self.assertEqual(profile_vote(a)[0],-1)
        a[0]=3;a[1]=2.9
        self.assertEqual(profile_vote(a)[0],-1)

    def test_edge_strip_is_filtered_but_internal_one_is_retained(self):
        image=np.full((100,220,3),240,np.uint8)
        boxes=[[5,0,65,5],[10,20,55,26],[70,20,95,50],[110,20,135,50],[150,20,175,50]]
        for x1,y1,x2,y2 in boxes:cv2.rectangle(image,(x1,y1),(x2,y2),(20,20,20),-1)
        with patch('font.solver.detect_characters',return_value=boxes):
            p=odd_font(None,image,engine='profile')
        self.assertEqual(p.detail['discarded_indices'],[0])
        self.assertIn(1,p.detail['kept_indices'])

    def test_multiple_targets_and_unknown_engine_are_rejected(self):
        with patch('font.solver.detect_characters',return_value=[]):
            self.assertFalse(odd_font(None,np.zeros((10,10,3),np.uint8),2).usable)
        with self.assertRaises(ValueError):odd_font(None,None,engine='typo')

if __name__=='__main__':unittest.main()
