"""自建可见图形的留出旋转/极性测试，不使用官方题目作为源码夹具。"""
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw

from icon.yidun import click_yidun_icons, hint_masks, _photometric, _variant
from icon.demo import icon_verdict


def glyph(kind):
    image=Image.new('L',(64,64));draw=ImageDraw.Draw(image)
    if kind==0:
        draw.ellipse((5,5,59,59),fill=255);draw.ellipse((15,15,49,49),fill=0)
        draw.line((10,32,54,32),fill=255,width=5);draw.line((32,10,32,54),fill=255,width=5)
    elif kind==1:
        draw.rectangle((7,16,57,55),outline=255,width=6)
        draw.line((17,4,33,17,47,3),fill=255,width=5)
    else:
        draw.polygon([(5,20),(31,20),(31,5),(60,32),(31,59),(31,44),(5,44)],fill=255)
        draw.rectangle((16,26,30,37),fill=0)
    return image.crop(image.getbbox())


def paste(canvas, source, center, size, angle, color):
    source=source.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True)
    source=source.crop(source.getbbox());source.thumbnail((size,size),Image.Resampling.LANCZOS)
    alpha=np.asarray(source).astype(float)/255;h,w=alpha.shape
    x,y=round(center[0]-w/2),round(center[1]-h/2)
    patch=canvas[y:y+h,x:x+w]
    patch[:]=np.rint(patch*(1-alpha[:,:,None])+np.array(color)*alpha[:,:,None])


def scene(duplicate=False):
    rng=np.random.default_rng(12203)
    texture=rng.integers(90,160,(12,24,3),dtype=np.uint8)
    background=cv2.resize(texture,(320,160),interpolation=cv2.INTER_CUBIC)
    hint=np.full((20,80,3),245,np.uint8)
    points=[(45,48),(155,105),(270,60)]
    for index,point in enumerate(points):
        source=glyph(0 if duplicate and index==1 else index)
        paste(hint,source,(10+28*index,10),19,0,(65,65,65))
        paste(background,source,point,32,27+index*43,(40,48,47) if index==1 else (232,236,228))
    return background,hint,points


class YidunIconTests(unittest.TestCase):
    def test_flat_sky_is_not_evidence_despite_zero_foreground_variance(self):
        _,hint,_=scene();mask=hint_masks(hint)[0][1]
        candidate={'point':[60,60],'score':.8,'angle':0,'scale':1.5}
        result=_photometric(np.full((120,120,3),220,np.uint8),mask,candidate)
        self.assertEqual(result['spread'],0)
        self.assertEqual(result['correlation'],0)
        self.assertEqual(result['evidence'],0)

    def test_texture_penalizes_equal_silhouette_candidate(self):
        _,hint,_=scene();mask=hint_masks(hint)[0][1]
        kernel=_variant(mask,0,1.5);h,w=kernel.shape
        plain=np.full((120,120,3),120,np.uint8)
        noisy=plain.copy();box=np.s_[60-h//2:60-h//2+h,60-w//2:60-w//2+w]
        plain[box][kernel>.5]=230
        rng=np.random.default_rng(109)
        noisy[box][kernel>.5]=rng.integers(160,256,(int((kernel>.5).sum()),1))
        candidate={'point':[60,60],'score':.8,'angle':0,'scale':1.5}
        a=_photometric(plain,mask,dict(candidate));b=_photometric(noisy,mask,dict(candidate))
        self.assertEqual(a['score'],b['score'])
        self.assertGreater(a['correlation'],b['correlation'])
        self.assertGreater(a['evidence'],b['evidence'])

    def test_small_border_touching_hints_preserve_all_three(self):
        _,hint,_=scene()
        parts=hint_masks(hint)
        self.assertEqual(len(parts),3)
        self.assertTrue(any(box[0]==0 or box[1]==0 for box,mask in parts))

    def test_rotated_light_and_dark_icons_on_texture(self):
        background,hint,expected=scene()
        prediction=click_yidun_icons(background,hint)
        self.assertTrue(prediction.usable,prediction.to_dict())
        for actual,point in zip(prediction.points,expected):
            self.assertLess(np.linalg.norm(np.array(actual)-point),5)

    def test_duplicate_icons_have_no_arbitrary_order(self):
        background,hint,_=scene(duplicate=True)
        self.assertFalse(click_yidun_icons(background,hint).usable)

    def test_missing_hint_abstains_before_search(self):
        background,hint,_=scene();hint[:,:24]=245
        self.assertFalse(click_yidun_icons(background,hint).usable)

    def test_toolbar_target_is_excluded(self):
        background,hint,_=scene()
        prediction=click_yidun_icons(background,hint,[(245,30,65,65)])
        self.assertFalse(prediction.usable,prediction.to_dict())

    def test_empty_background_and_invalid_inputs_abstain(self):
        background,hint,_=scene()
        self.assertFalse(click_yidun_icons(np.full_like(background,140),hint).usable)
        self.assertFalse(click_yidun_icons(None,hint).usable)
        self.assertFalse(click_yidun_icons(background,np.zeros((5,80,3),np.uint8)).usable)


class IconVerdictTests(unittest.TestCase):
    def test_current_server_and_ui_are_both_required(self):
        receipt={'attempt_id':'a','challenge_matched':True,'accepted':True}
        self.assertEqual(icon_verdict(receipt,'a',True,True),'passed')
        for r,attempt,current,ui in [(None,'a',True,True),(receipt,'old',True,True),
                (receipt,'a',False,True),(receipt,'a',True,False),
                (dict(receipt,challenge_matched=False),'a',True,True)]:
            self.assertEqual(icon_verdict(r,attempt,current,ui),'unknown')
        self.assertEqual(icon_verdict(dict(receipt,accepted=False),'a',True,True),'failed')


if __name__=='__main__':unittest.main()
