"""用私有裁判答案检查动作映射；这组控制测试不计入算法通过率。"""
import os
import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch,AsyncMock
from playwright.async_api import async_playwright
import lab.server as lab
from common.geometry_demo import Adapter, decode, verdict, run_geometry
from common.interaction import Interaction
from common.vision import Prediction
from common.language_demo import KGAdapter, DemoUnavailable


@unittest.skipUnless(os.getenv('CAPTCHA_BROWSER_TEST')=='1','需显式启用 Chromium 测试')
class SecondBatchBrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.answers={};original=lab.generate
        def issue(kind,rng):
            public,answer=original(kind,rng);self.answers[public['background']]=answer
            return public,answer
        self.patcher=patch('lab.server.generate',side_effect=issue);self.patcher.start()
        self.server=lab.local_server(4321);self.url=self.server.__enter__()
        self.pw=await async_playwright().start();self.browser=await self.pw.chromium.launch(headless=True)
        self.page=await self.browser.new_page(viewport={'width':1320,'height':960},device_scale_factor=2)
        self.interaction=Interaction(self.page,scale=.25,seed=5);await self.interaction.install()

    async def asyncTearDown(self):
        await self.browser.close();await self.pw.stop();self.server.__exit__(None,None,None);self.patcher.stop()

    async def prepare(self,kind,scale=1):
        a=Adapter(self.page,'lab',kind,self.url,scale);await a.open();c=a.receipts.challenge
        image,_=decode(c['background']);answer=self.answers[c['background']]
        p=Prediction(kind,True,value=answer.get('angle'),points=answer.get('points',[]))
        a.receipts.active=('control',c['key'])
        return a,c,{'background':image},p

    async def test_all_new_actions_at_two_scales_and_high_dpi(self):
        for kind in ('rotate-single','font','order'):
            for scale in (.8,1.25):
                a,c,images,p=await self.prepare(kind,scale)
                await a.act(c,images,p,self.interaction)
                await self.page.locator('#status[data-state="passed"]').wait_for()
                for _ in range(100):
                    if a.receipts.results:break
                    await asyncio.sleep(.02)
                self.assertTrue(a.receipts.results)
                self.assertEqual(verdict(a.receipts.results[-1],'control',c['key'],await a.ui(c['key']),await a.current(c)),'passed')

    async def test_new_question_invalidates_old_clicks(self):
        a,c,images,p=await self.prepare('order');await a.refresh(c['key'])
        with self.assertRaisesRegex(RuntimeError,'题面已变化'):await a.act(c,images,p,self.interaction)
        self.assertFalse(a.receipts.results)

    async def test_refresh_mid_sequence_never_submits_old_answer(self):
        a,c,images,p=await self.prepare('order');pause=self.interaction.pause
        async def changed(stage,lo,hi):
            if stage=='between_clicks':await a.refresh(c['key'])
            else:await pause(stage,lo,hi)
        with patch.object(self.interaction,'pause',side_effect=changed):
            with self.assertRaisesRegex(RuntimeError,'题面变化'):await a.act(c,images,p,self.interaction)
        self.assertFalse(a.receipts.results)

    async def test_kg_error_message_does_not_count_as_new_image(self):
        await self.page.set_content('<div id="KgControl">请点击1个字体不一样的文字</div><div id="KgDisplay"><img src="/strbasemap/one.png"></div>')
        adapter=KGAdapter(self.page,'font',Path('.'))
        before=await adapter.snapshot()
        await self.page.locator('#KgControl').evaluate("e=>e.textContent='验证失败，请重试'")
        self.assertEqual(before['key'],(await adapter.snapshot())['key'])
        await self.page.locator('#KgDisplay img').evaluate("e=>e.src='/strbasemap/two.png'")
        self.assertNotEqual(before['key'],(await adapter.snapshot())['key'])

    async def test_site_quota_stops_entire_batch_without_more_open_requests(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(KGAdapter,'open',new=AsyncMock(side_effect=DemoUnavailable('每小时请求次数超出限制'))) as opening:
                summary=await run_geometry(Path(directory),'kg','order',cases=3,max_attempts=4,headless=True,retry_low_confidence=True)
                self.assertEqual(opening.await_count,1)
                self.assertEqual(summary['cases'],3)
                self.assertEqual(summary['unknown'],3)
                self.assertEqual(summary['attempts'],1)
                self.assertEqual(summary['first_attempt_passed'],0)

    async def test_explicit_quota_is_detected_before_image_decode(self):
        await self.page.set_content('<div id="KgControl">每小时请求次数超出限制</div>')
        adapter=KGAdapter(self.page,'order',Path('.'))
        with self.assertRaisesRegex(DemoUnavailable,'每小时'):
            await adapter.ready()
