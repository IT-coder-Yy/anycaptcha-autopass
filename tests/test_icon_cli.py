import contextlib
import io
import unittest
from unittest.mock import AsyncMock, patch

from cli import main


class IconCliTests(unittest.TestCase):
    def test_retries_are_additional_to_first_attempt(self):
        result={'cases':1,'passed':1,'failed':0,'unknown':0}
        command=['captcha-local','demo','--site','gocaptcha','--kind','icon','--retries','3','--retry-low-confidence']
        with patch('sys.argv',command), patch('common.geometry_demo.run_geometry',AsyncMock(return_value=result)) as run:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
                main()
        self.assertEqual(exit.exception.code,0)
        self.assertEqual(run.call_args.args[4],4)
        self.assertIs(run.call_args.args[-1],True)

    def test_retry_budget_is_bounded_and_unambiguous(self):
        for extra in (['--retries','4'],['--retries','3','--attempts','3']):
            with patch('sys.argv',['captcha-local','demo','--site','gocaptcha','--kind','icon',*extra]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit:
                    main()
            self.assertEqual(exit.exception.code,2)

    def test_yidun_icon_routes_to_local_solver_with_four_attempts(self):
        result={'cases':1,'passed':1,'failed':0,'unknown':0}
        command=['captcha-local','demo','--site','yidun','--kind','icon','--retries','3','--retry-low-confidence']
        with patch('sys.argv',command), patch('icon.demo.run_yidun_icons',AsyncMock(return_value=result)) as run:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
                main()
        self.assertEqual(exit.exception.code,0)
        self.assertEqual(run.call_args.args[2],4)
        self.assertIs(run.call_args.args[-1],True)
