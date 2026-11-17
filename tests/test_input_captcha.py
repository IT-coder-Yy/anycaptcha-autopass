import base64
import unittest
from contextlib import redirect_stderr
from io import StringIO
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from arithmetic.solver import answer_from_text as arithmetic_answer
from cli import main
from common.ocr import evaluate_samples
from lab.printed import samples
from lab.server import ChallengeStore
from normal.solver import answer_from_text as normal_answer


class InputCaptchaTests(unittest.TestCase):
    def test_normal_preserves_case_and_zeroes(self):
        self.assertEqual(normal_answer("00Ao9"), ("00Ao9", ""))
        for text in ("", "Aa 1", "0A!", " 0073", "１２", "A\nB"):
            with self.subTest(text=text):
                self.assertIsNone(normal_answer(text)[0])

    def test_arithmetic_supported_expressions(self):
        examples = {"7 + 5 = ?": "12", "7 + 5 = ？": "12", "2-7": "-5", "12 ÷ 3 =": "4",
                    "0/8": "0", "3*7": "21", "4×6": "24", "2x9": "18", "9X3": "27"}
        for text, answer in examples.items():
            with self.subTest(text=text):
                self.assertEqual(arithmetic_answer(text), (answer, ""))

    def test_arithmetic_rejects_incomplete_and_unsupported_expressions(self):
        for text in ("", "5/0", "7/2", "-2+1", "2+-1", "1+2*3", "(4+1)",
                     "3.0+2", "1 2+3", "7十5", "1+2\n", "７+５", "1+2？", "1+2=？？",
                     "1？+2", "1+2=？3", "len('a')"):
            with self.subTest(text=text):
                answer, reason = arithmetic_answer(text)
                self.assertIsNone(answer)
                self.assertTrue(reason)

    def test_real_ocr_answers_match_local_grading(self):
        for kind, name, text, answer, engine in (
            ("normal", "normal-03.png", "0073", "0073", "ppocr-v5-mobile"),
            ("normal", "normal-13.png", "Am42", "Am42", "ddddocr"),
            ("normal", "normal-14.png", "70bk", "70bk", "ddddocr"),
            ("arithmetic", "arithmetic-02.png", "2-7", "-5", "ppocr-v5-mobile"),
            ("arithmetic", "arithmetic-08.png", "7+5=？", "12", "ppocr-v5-mobile"),
        ):
            with self.subTest(kind=kind, image=name):
                case = next(s for s in samples(kind) if s.image.name == name)
                result = evaluate_samples(kind, case.image)
                self.assertEqual((result["total"], result["correct"]), (1, 1))
                row, = result["samples"]
                self.assertEqual((row["raw_text"], row["answer"]), (text, answer))
                self.assertEqual(row["engine"], engine)
                self.assertEqual(set(result["engines"]), {engine})
                store = ChallengeStore()
                with patch.object(store.rng, "choice", return_value=case):
                    key = store.issue(kind)["key"]
                self.assertEqual(store.verify(key, {"text": row["answer"]}), {"key": key, "accepted": True})

    def test_local_judge_checks_the_displayed_rule_and_consumes_key(self):
        for kind, index, variant, matches in (
            ("normal", 0, "0ak09", False), ("normal", 2, "73", False),
            ("normal", 12, "aM42", True), ("normal", 13, "70Bk", True),
            ("normal", 13, "70bK", False), ("normal", 14, "0g7", False),
            ("arithmetic", 3, "9", False),
        ):
            case = samples(kind)[index]
            for submitted, accepted in ((case.answer, True), (variant, matches), (0, False)):
                with self.subTest(kind=kind, submitted=submitted):
                    store = ChallengeStore()
                    store.rng.choice = lambda cases: cases[index]
                    challenge = store.issue(kind)
                    self.assertEqual(set(challenge), {"key", "kind", "instruction", "background", "width", "height"})
                    if kind == "normal":
                        hint = {True: "区分大小写。", False: "不区分大小写。", None: ""}[case.case_sensitive]
                        self.assertEqual(challenge["instruction"], "照图输入字符。" + hint)
                    png = base64.b64decode(challenge["background"].split(",", 1)[1])
                    self.assertEqual(png, case.image.read_bytes())
                    key = challenge["key"]
                    self.assertEqual(store.verify(key, {"text": submitted, "ignore_case": True}),
                                     {"key": key, "accepted": accepted})
                    self.assertFalse(store.verify(key, {"text": case.answer})["accepted"])

    def test_batch_keeps_wrong_answers_abstentions_and_errors(self):
        cases = samples("normal")
        readings = [case.text for case in cases]
        readings[:3] = ["wrong", "", OSError("样本读取错误")]
        readings[12] = cases[12].text.swapcase()
        with patch("common.ocr._load_engine", return_value=(Mock(side_effect=readings), {"version": "test"})) as load:
            result = evaluate_samples("normal")
        self.assertEqual((result["total"], result["correct"]), (len(cases), len(cases) - 3))
        self.assertEqual((result["wrong"], result["abstained"], result["error"]), (1, 1, 1))
        self.assertEqual([call.args[0] for call in load.call_args_list], ["ppocr-v5-mobile", "ddddocr"])
        loose = result["samples"][12]
        self.assertEqual((loose["raw_text"], loose["text_exact"], loose["outcome"]), ("aM42", False, "correct"))

    def test_missing_ppocr_model_only_affects_samples_that_use_it(self):
        with TemporaryDirectory() as root, patch.dict("os.environ", {"CAPTCHA_PROJECT_ROOT": root}):
            with self.assertRaisesRegex(FileNotFoundError, "ppocr-v5-mobile"):
                evaluate_samples("arithmetic")
            loose = samples("normal")[12]
            self.assertEqual(evaluate_samples("normal", loose.image)["samples"][0]["engine"], "ddddocr")

    def test_cli_rejects_unavailable_inputs_and_site_combinations(self):
        calls = (["recognize", "--kind", "text"],
                 ["recognize", "--kind", "normal", "src/arithmetic/data/arithmetic-02.png"],
                 ["demo", "--site", "kg", "--kind", "arithmetic"])
        for args in calls:
            with self.subTest(args=args), patch("sys.argv", ["captcha-local", *args]), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
