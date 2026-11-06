"""真实 Chromium + 本机 HTTP；故障注入只用于判定边界，不计入算法通过率。"""
import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from playwright.async_api import async_playwright
from common.geometry_demo import Adapter, run_geometry, verdict
from common.interaction import Interaction
from lab.server import ChallengeStore, local_server
from common.vision import Prediction


@unittest.skipUnless(os.getenv('CAPTCHA_BROWSER_TEST')=='1','需显式启用 Chromium 测试')
class GeometryBrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.server=local_server(101);self.url=self.server.__enter__()
        self.pw=await async_playwright().start()
        self.browser=await self.pw.chromium.launch(headless=True)
        self.page=await self.browser.new_page(viewport={'width':1320,'height':960},device_scale_factor=2)
        self.interaction=Interaction(self.page,scale=.25,seed=10);await self.interaction.install()

    async def asyncTearDown(self):
        await self.browser.close();await self.pw.stop();self.server.__exit__(None,None,None);self.tmp.cleanup()

    async def prepare(self,kind='drag2d',scale=.8):
        adapter=Adapter(self.page,'lab',kind,self.url,scale);await adapter.open()
        folder=self.root/f'{kind}-{scale}';folder.mkdir(exist_ok=True)
        c,images,p=await adapter.read(folder)
        self.assertTrue(p.usable,p.to_dict())
        adapter.receipts.active=('test',c['key'])
        return adapter,c,images,p

    async def wait_receipt(self,adapter):
        for _ in range(100):
            if adapter.receipts.results:return adapter.receipts.results[-1]
            await asyncio.sleep(.02)
        self.fail('没有校验回执')

    async def test_css_scale_and_device_pixels_all_three_kinds(self):
        for kind in ('restore','drag2d','icon'):
            for scale in (.8,1.25):
                a,c,images,p=await self.prepare(kind,scale)
                await a.act(c,images,p,self.interaction)
                receipt=await self.wait_receipt(a)
                await self.page.locator('#status[data-state="passed"]').wait_for()
                self.assertEqual(verdict(receipt,'test',c['key'],await a.ui(c['key']),await a.current(c)),'passed')

    async def test_changed_challenge_stops_before_action(self):
        a,c,images,p=await self.prepare()
        await a.refresh(c['key'])
        with self.assertRaisesRegex(RuntimeError,'题面已变化'):await a.act(c,images,p,self.interaction)
        self.assertFalse(a.receipts.results)

    async def test_change_after_first_icon_click_does_not_submit(self):
        a,c,images,p=await self.prepare('icon')
        original=self.interaction.pause
        async def pause(stage,low,high):
            if stage=='between_clicks':await a.refresh(c['key'])
            else:await original(stage,low,high)
        with patch.object(self.interaction,'pause',side_effect=pause):
            with self.assertRaisesRegex(RuntimeError,'题面变化'):await a.act(c,images,p,self.interaction)
        self.assertFalse(a.receipts.results)

    async def test_ui_only_and_old_ui_do_not_pass(self):
        a,c,images,p=await self.prepare()
        await self.page.locator('#status').evaluate('(e,key)=>{e.dataset.key=key;e.dataset.state="passed"}',c['key'])
        self.assertEqual(verdict(None,'test',c['key'],await a.ui(c['key'])),'unknown')
        self.assertEqual(await a.ui('old-key'),'')

    async def test_server_success_without_ui_is_unknown(self):
        a,c,images,p=await self.prepare()
        await self.page.evaluate('''()=>{const e=document.querySelector('#status');new MutationObserver(()=>{
            if(e.dataset.state==='passed'){e.dataset.state='';e.textContent='';}
        }).observe(e,{attributes:true});}''')
        await a.act(c,images,p,self.interaction)
        receipt=await self.wait_receipt(a)
        self.assertIs(receipt['accepted'],True)
        self.assertEqual(verdict(receipt,'test',c['key'],await a.ui(c['key'])),'unknown')

    async def test_wrong_key_in_response_is_unknown(self):
        a,c,images,p=await self.prepare()
        async def wrong_key(route):
            response=await route.fetch();body=await response.json();body['key']='old-key'
            await route.fulfill(response=response,json=body)
        await self.page.route('**/verify',wrong_key)
        await a.act(c,images,p,self.interaction)
        receipt=await self.wait_receipt(a)
        self.assertIsNone(receipt['accepted'])
        self.assertEqual(verdict(receipt,'test',c['key'],await a.ui(c['key'])),'unknown')

    async def test_duplicate_toasts_are_read_without_strict_locator_error(self):
        a,c,images,p=await self.prepare()
        a.site='gocaptcha'
        await self.page.evaluate('''()=>{for(let i=0;i<2;i++){
            const e=document.createElement('p');e.textContent='check data failed';document.body.append(e);
        }}''')
        self.assertTrue(await a.toast_visible('check data failed'))
        self.assertEqual(await a.ui(c['key']),'failed')

    async def test_ready_waits_for_matching_auxiliary_image(self):
        a,c,images,p=await self.prepare()
        await self.page.locator('#piece').evaluate('''e=>{const src=e.src;e.src='';
            setTimeout(()=>e.src=src,250);}''')
        await a.ready()
        self.assertEqual(await a.auxiliary().get_attribute('src'),c['piece'])

    async def test_decode_interrupted_by_refresh_is_retried_without_new_challenge(self):
        a,c,images,p=await self.prepare()
        await self.page.locator('#background').evaluate('''e=>{
            const decode=e.decode.bind(e);let calls=0;
            e.decode=()=>++calls===1?Promise.reject(new DOMException('换题','EncodingError')):decode();
        }''')
        await a.ready()
        self.assertEqual(a.receipts.challenge['key'],c['key'])

    async def test_drag_exception_always_releases_mouse(self):
        a,c,images,p=await self.prepare()
        original=self.interaction.pause
        async def pause(stage,low,high):
            if stage=='before_release':raise RuntimeError('控制实验：拖动中断')
            await original(stage,low,high)
        await self.page.evaluate('()=>document.addEventListener("pointerup",()=>window.released=true)')
        with patch.object(self.interaction,'pause',side_effect=pause):
            with self.assertRaisesRegex(RuntimeError,'控制实验'):await a.act(c,images,p,self.interaction)
        self.assertTrue(await self.page.evaluate('()=>window.released'))

    async def test_runner_budget_and_low_confidence_stop(self):
        # 真 HTTP 拒绝，限定三次；然后模拟识别弃权，必须一题结束，不刷新挑题。
        with patch.dict(os.environ,{'CAPTCHA_ACTION_SCALE':'.25'}), patch.object(
                ChallengeStore,'verify',side_effect=lambda key,answer:{'key':key,'accepted':False}):
            result=await run_geometry(self.root,'lab','drag2d',cases=1,max_attempts=3,headless=True,seed=101)
        self.assertEqual((result['attempts'],result['failed'],result['server_submissions']),(3,1,3))
        async def abstain(adapter,folder):
            return dict(adapter.receipts.challenge),{},Prediction('drag2d',False,0,reason='控制实验：弃权')
        with patch.object(Adapter,'read',abstain):
            result=await run_geometry(self.root,'lab','drag2d',cases=1,max_attempts=3,headless=True)
        self.assertEqual((result['attempts'],result['unknown'],result['server_submissions']),(1,1,0))
        self.assertEqual(result['refresh_sources'],{})
        async def timeout(adapter):await asyncio.sleep(3)
        with patch.object(Adapter,'open',timeout):
            result=await run_geometry(self.root,'lab','drag2d',cases=1,max_attempts=3,seconds=.1,headless=True)
        self.assertEqual((result['attempts'],result['unknown'],result['exceptions']),(1,1,1))

    async def test_three_retries_mean_four_attempts_and_then_stop(self):
        with patch.dict(os.environ,{'CAPTCHA_ACTION_SCALE':'.25'}), patch.object(
                ChallengeStore,'verify',side_effect=lambda key,answer:{'key':key,'accepted':False}):
            result=await run_geometry(self.root,'lab','drag2d',cases=1,max_attempts=4,headless=True,seed=101)
        self.assertEqual((result['attempts'],result['retry_attempts'],result['server_submissions']),(4,3,4))
        self.assertEqual(result['failed'],1)

    async def test_opt_in_abstention_retries_have_shared_budget(self):
        async def abstain(adapter,folder):
            return dict(adapter.receipts.challenge),{},Prediction('icon',False,reason='控制实验：弃权')
        with patch.object(Adapter,'read',abstain):
            result=await run_geometry(self.root,'lab','icon',cases=1,max_attempts=4,headless=True,retry_low_confidence=True)
        self.assertEqual((result['attempts'],result['retry_attempts'],result['unknown'],result['server_submissions']),(4,3,1,0))
        self.assertEqual(result['refresh_sources'],{'explicit_refresh':3})
        import json
        report=json.loads((Path(result['directory'])/'report.json').read_text())[0]
        self.assertEqual(len({a['challenge_id'] for a in report['attempts']}),4)
        self.assertTrue(all(r['after_outcome']=='low_confidence' for r in report['refreshes']))

    async def test_abstention_then_success_counts_as_recovered_not_first_pass(self):
        original=Adapter.read
        calls=0
        async def first_abstain(adapter,folder):
            nonlocal calls
            calls+=1
            if calls==1:return dict(adapter.receipts.challenge),{},Prediction('icon',False,reason='控制实验：弃权')
            return await original(adapter,folder)
        with patch.dict(os.environ,{'CAPTCHA_ACTION_SCALE':'.25'}), patch.object(Adapter,'read',first_abstain):
            result=await run_geometry(self.root,'lab','icon',cases=1,max_attempts=4,headless=True,seed=101,retry_low_confidence=True)
        self.assertEqual((result['attempts'],result['first_attempt_passed'],result['recovered_cases'],result['passed']),(2,0,1,1))


if __name__=='__main__':unittest.main()
