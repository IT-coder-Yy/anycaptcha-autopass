import json
import unittest
import numpy as np

from common.evidence import Budget, Evidence, Verdict
from common.vision import Prediction, best_assignment
from common.demo import parse_jsonp


class EvidenceTests(unittest.TestCase):
    def good(self, **changes):
        fields=dict(attempt_id="new",challenge_id="image-A",challenge_visible=False,
                    server_accepted=True,response_attempt_id="new",ui_success=True)
        fields.update(changes)
        return Evidence(**fields)

    def test_joint_success(self):
        self.assertEqual(self.good().verdict(),Verdict.PASSED)

    def test_popup_disappearance_is_not_success(self):
        self.assertEqual(self.good(server_accepted=None).verdict(),Verdict.UNKNOWN)

    def test_stale_response_is_not_success(self):
        self.assertEqual(self.good(response_attempt_id="old").verdict(),Verdict.UNKNOWN)

    def test_next_challenge_is_not_success(self):
        self.assertEqual(self.good(challenge_visible=True).verdict(),Verdict.UNKNOWN)

    def test_completed_embedded_image_can_remain_visible(self):
        self.assertEqual(self.good(challenge_visible=True,challenge_completed=True).verdict(),Verdict.PASSED)
        self.assertEqual(self.good(challenge_visible=True,challenge_completed=True,server_accepted=None).verdict(),Verdict.UNKNOWN)
        self.assertEqual(self.good(challenge_visible=True,challenge_completed=True,response_attempt_id='old').verdict(),Verdict.UNKNOWN)

    def test_server_reject_overrides_success_label(self):
        self.assertEqual(self.good(server_accepted=False).verdict(),Verdict.FAILED)

    def test_ui_success_required(self):
        self.assertEqual(self.good(ui_success=False).verdict(),Verdict.UNKNOWN)

    def test_search_requires_new_current_keyword_response(self):
        fields=dict(requires_search=True,keyword="风景",response_keyword="风景",search_accepted=True,search_started_after_attempt=True)
        self.assertEqual(self.good(**fields).verdict(),Verdict.PASSED)
        for extra in ({"response_keyword":"旧关键词"},{"search_started_after_attempt":False},{"search_accepted":False}):
            self.assertEqual(self.good(**(fields|extra)).verdict(),Verdict.UNKNOWN)


class BudgetTests(unittest.TestCase):
    def test_time_and_rounds_share_one_budget(self):
        now=[0.0]
        budget=Budget(3,120,lambda:now[0])
        self.assertTrue(budget.take())
        self.assertTrue(budget.take())
        self.assertTrue(budget.take())
        self.assertFalse(budget.take())
        timed=Budget(3,120,lambda:now[0])
        now[0]=121
        self.assertFalse(timed.take())


class MatchingTests(unittest.TestCase):
    def test_assignment_is_global_not_greedy(self):
        chosen,_,_=best_assignment([[.90,.85],[.88,.10]])
        self.assertEqual(chosen,[1,0])

    def test_duplicate_target_not_clicked_twice(self):
        chosen,_,_=best_assignment([[.99,.95],[.99,.95]])
        self.assertEqual(len(set(chosen)),2)

    def test_ambiguity_is_detectable(self):
        _,_,gap=best_assignment([[.9,.9],[.9,.9]])
        self.assertEqual(gap,0)

    def test_invalid_detection_and_nonfinite_scores(self):
        for scores in ([],[[.9],[.9]],[[float("nan")]]):
            with self.assertRaises(ValueError): best_assignment(scores)

    def test_numpy_prediction_serializes(self):
        result=Prediction("slider",False,detail={"score":np.float32(.9),"raw":np.array([float('nan'),.5])})
        value=json.loads(json.dumps(result.to_dict(),allow_nan=False))
        self.assertIsNone(value['detail']['raw'][0])

    def test_jsonp_only_parses_data(self):
        self.assertEqual(parse_jsonp('cb({"data":{"result":false},"error":0});')['data']['result'],False)
        with self.assertRaises((ValueError,TypeError)): parse_jsonp('alert("not data")')


if __name__=="__main__": unittest.main()
