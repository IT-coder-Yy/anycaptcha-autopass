"""本机拦截全部网络的易盾协议夹具；检查 sprite 裁剪、缩放、回执和换题。"""
import asyncio
import base64
import json
import os
from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
from playwright.async_api import async_playwright

from common.interaction import Interaction
from common.vision import Prediction
from icon.demo import IconAdapter


HTML = '''<style>
.yidun{position:absolute;left:100px;top:250px;width:320px;transform:scale(.8);transform-origin:0 0}
.yidun_bgimg{width:320px;height:160px;overflow:hidden}.yidun_bg-img{width:320px;height:240px}
.yidun_control{height:40px}.yidun_tips__answer{position:relative;width:80px;height:20px;overflow:hidden}
.yidun_tips__img{position:absolute;top:-200px;width:320px;height:240px}
.yidun_top__right{position:absolute;right:0;top:0;width:80px;height:25px;background:#ddd}
</style><div class="yidun"><div class="yidun_panel"><div class="yidun_bgimg"><img class="yidun_bg-img"></div>
<div class="yidun_top__right"><button class="yidun_refresh">刷新</button></div></div>
<div class="yidun_control"><div class="yidun_tips__answer"><img class="yidun_tips__img"></div></div></div>
<script>
const root=document.querySelector('.yidun');let count=0,token;window.clicks=[];window.change=false;window.ui=true;
async function issue(){const data=(await (await fetch('https://c.dun.163.com/api/v3/get')).json()).data;
 token=data.token;count=0;root.className='yidun';document.querySelectorAll('img').forEach(e=>e.src=data.bg[0]);}
document.querySelector('.yidun_refresh').onclick=issue;
document.querySelector('.yidun_bgimg').onclick=async e=>{
 const b=e.currentTarget.getBoundingClientRect();window.clicks.push([(e.clientX-b.x)/b.width*320,(e.clientY-b.y)/b.height*160]);
 count++;if(count===1&&window.change){await issue();return;}
 if(count===3){const data=(await (await fetch('https://c.dun.163.com/api/v3/check?token='+token)).json()).data;
 if(window.ui)root.className=data.result?'yidun yidun--success':'yidun yidun--error';}}
issue();</script>'''


@unittest.skipUnless(os.getenv('CAPTCHA_BROWSER_TEST')=='1','需显式启用 Chromium 测试')
class YidunIconBrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)
        self.pw=await async_playwright().start();self.browser=await self.pw.chromium.launch(headless=True)
        self.page=await self.browser.new_page(viewport={'width':1360,'height':1000},device_scale_factor=2)
        self.generation=0;self.accept=True;self.wrong_token=False
        image=np.full((240,320,3),180,np.uint8)
        image[200:220,:80]=245
        for x in (5,33,61):cv2.circle(image,(x+4,210),5,(50,50,50),-1)
        self.source='data:image/png;base64,'+base64.b64encode(cv2.imencode('.png',image)[1]).decode()
        async def route(request):
            url=urlsplit(request.request.url)
            if url.path=='/trial/icon-click':
                await request.fulfill(content_type='text/html',body=HTML)
            elif url.path=='/api/v3/get':
                self.generation+=1
                body={'error':0,'data':{'token':f't{self.generation}','bg':[self.source]}}
                await request.fulfill(json=body,headers={'Access-Control-Allow-Origin':'*'})
            elif url.path=='/api/v3/check':
                token=parse_qs(url.query)['token'][0]
                body={'error':0,'data':{'token':'old' if self.wrong_token else token,'result':self.accept,
                                      'validate':'fixture-ticket' if self.accept else ''}}
                await request.fulfill(json=body,headers={'Access-Control-Allow-Origin':'*'})
            else:await request.fulfill(status=404,body='')
        await self.page.route('**/*',route)
        self.adapter=IconAdapter(self.page,'baseline')
        self.interaction=Interaction(self.page,scale=.25,seed=8);await self.interaction.install()
        await self.adapter.open()
        self.challenge,self.images,_=await self.adapter.read(self.folder)
        scale=self.images['background'].shape[1]/320
        self.prediction=Prediction('icon',True,1,points=[[x*scale,y*scale] for x,y in [(45,48),(155,105),(270,60)]])
        self.adapter.book.active='attempt';self.adapter.book.expected_token=self.challenge['token']

    async def asyncTearDown(self):
        await self.browser.close();await self.pw.stop();self.tmp.cleanup()

    async def submit(self):
        await self.adapter.act(self.challenge,self.images,self.prediction,self.interaction)
        return await self.adapter.result(self.challenge,'attempt',.6)

    async def test_sprite_crop_device_pixels_and_css_scale(self):
        self.assertEqual(self.images['background'].shape[:2],(256,512))
        self.assertLess(self.images['hint'].shape[0],50)
        outcome,receipts,ui=await self.submit()
        self.assertEqual(outcome,'passed');self.assertTrue(receipts[-1]['challenge_matched'])
        for actual,expected in zip(await self.page.evaluate('clicks'),[(45,48),(155,105),(270,60)]):
            self.assertLess(np.linalg.norm(np.array(actual)-expected),2)

    async def test_wrong_token_and_ui_success_do_not_pass(self):
        self.wrong_token=True
        outcome,receipts,ui=await self.submit()
        self.assertTrue(ui);self.assertEqual(outcome,'unknown');self.assertIsNone(receipts[-1]['accepted'])

    async def test_wrong_answer_control_has_three_separated_scaled_points(self):
        self.accept=False
        await self.adapter.act(self.challenge,self.images,self.prediction,self.interaction,wrong=True)
        outcome,receipts,_=await self.adapter.result(self.challenge,'attempt',.6)
        self.assertEqual(outcome,'failed');self.assertIs(receipts[-1]['accepted'],False)
        points=await self.page.evaluate('clicks')
        self.assertEqual(len(points),3)
        for actual,expected in zip(points,[(38.4,80),(160,120),(272,80)]):
            self.assertLess(np.linalg.norm(np.array(actual)-expected),2)

    async def test_server_accept_without_ui_is_unknown(self):
        await self.page.evaluate('window.ui=false')
        outcome,receipts,ui=await self.submit()
        self.assertIs(receipts[-1]['accepted'],True);self.assertEqual(outcome,'unknown')

    async def test_rejection_then_explicit_refresh_binds_new_generation(self):
        self.accept=False
        outcome,receipts,ui=await self.submit()
        self.assertEqual(outcome,'failed')
        source=await self.adapter.refresh(self.challenge,'failed')
        self.assertEqual(source,'explicit_refresh')
        self.assertFalse(await self.adapter.current(self.challenge))
        self.assertEqual(self.adapter.book.generation,self.challenge['generation']+1)

    async def test_changed_challenge_interrupts_click_sequence(self):
        await self.page.evaluate('window.change=true')
        with self.assertRaisesRegex(RuntimeError,'题面变化'):
            await self.adapter.act(self.challenge,self.images,self.prediction,self.interaction)
        self.assertFalse(self.adapter.book.results)


if __name__=='__main__':unittest.main()
