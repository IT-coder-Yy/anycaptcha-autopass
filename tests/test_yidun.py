import unittest
from common.yidun import ready, same_geometry


class ReadinessTests(unittest.TestCase):
    def snapshot(self):
        return {'classes':'yidun yidun--jigsaw','decoded':True,'image_visible':True,
                'piece_visible':True,'handle_visible':True,'hit_handle':True,
                **{k:{'x':50,'y':300,'width':40,'height':40}
                   for k in ('background','piece','handle','control','panel')}}

    def test_visible_image_during_failure_or_loading_is_not_ready(self):
        base=self.snapshot()
        self.assertTrue(ready(base))
        for state in ('loading','error','verifying','success'):
            self.assertFalse(ready(base|{'classes':base['classes']+' yidun--'+state}))

    def test_hidden_or_covered_handle_is_not_ready(self):
        for key in ('decoded','image_visible','piece_visible','handle_visible','hit_handle'):
            self.assertFalse(ready(self.snapshot()|{key:False}))

    def test_layout_drift_is_detected(self):
        first=self.snapshot();next_=self.snapshot()
        self.assertTrue(same_geometry(first,next_))
        next_['panel']['y']+=2
        self.assertFalse(same_geometry(first,next_))


if __name__=='__main__':unittest.main()
