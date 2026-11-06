"""凯格公开字体/语序演示适配；仅读取可见题面，通过页面鼠标事件提交。"""
import asyncio
import hashlib
import os
import re
from pathlib import Path
import cv2
import numpy as np
from common.demo import Receipts, image_bytes
from common.vision import Vision
from font.solver import odd_font
from order.solver import order_words


class DemoUnavailable(RuntimeError):
    """演示站明确拒绝继续出题；终止整批，不能以换题绕过。"""


class KGBook(Receipts):
    def __init__(self,page):
        super().__init__(page,'kg');self.challenge=None

    async def response(self,response):
        start=len(self.results)
        await super().response(response)
        for item in self.results[start:]:
            binding=item.get('attempt_id')
            if isinstance(binding,tuple):
                attempt,key=binding
                item.update(attempt_id=attempt,challenge_key=key,
                            challenge_matched=bool(self.challenge and self.challenge['key']==key))


class KGAdapter:
    def __init__(self,page,kind,root):
        self.page,self.kind=page,kind;self.vision=Vision(root/'models')
        self.receipts=KGBook(page);self.icon_engine='baseline'
        # profile 在自建题库有效，但当前凯格对照未改善，默认保留 baseline。
        self.font_engine=os.environ.get('CAPTCHA_FONT_ENGINE','baseline')
        self.widget=page.locator('#KgStrCaptchaBox')
        self.background=page.locator("#KgDisplay img[src*='strbasemap']")

    async def snapshot(self):
        source=await self.background.get_attribute('src')
        prompt=await self.page.locator('#KgControl').inner_text()
        return {'background':source,'prompt':prompt,
                'key':hashlib.sha256(str(source).encode()).hexdigest()}

    async def ready(self,old_key=None):
        await self.page.locator('#KgControl').wait_for(state='visible')
        for _ in range(60):
            prompt=await self.page.locator('#KgControl').inner_text()
            if '请求次数超出限制' in prompt:
                raise DemoUnavailable(prompt.strip())
            if not await self.background.count() or not await self.background.is_visible():
                await self.page.wait_for_timeout(100);continue
            await self.background.evaluate('(e)=>e.decode()')
            c=await self.snapshot()
            if c['key']!=old_key and ('请点击' in c['prompt'] or '请按语序' in c['prompt']):
                self.receipts.challenge=c
                await self.widget.scroll_into_view_if_needed();await self.page.wait_for_timeout(150)
                return
            await self.page.wait_for_timeout(100)
        raise RuntimeError('没有取得新的凯格题面')

    async def open(self):
        self.receipts.active=None;self.receipts.challenge=None
        await self.page.goto('https://www.kgcaptcha.com/demo/content?t='+('10' if self.kind=='font' else '7'),wait_until='domcontentloaded')
        await self.ready()

    async def refresh(self,old_key):
        self.receipts.active=None
        for _ in range(15):
            if await self.background.count() and (await self.snapshot())['key']!=old_key:
                await self.ready(old_key);return 'automatic'
            await self.page.wait_for_timeout(100)
        await self.page.locator("#captchaBox img[src$='/refresh.png']").click()
        await self.ready(old_key);return 'explicit_refresh'

    async def current(self,c):
        return (self.receipts.challenge is not None and self.receipts.challenge['key']==c['key']
                and await self.background.count()==1 and await self.background.get_attribute('src')==c['background'])

    async def read(self,folder):
        c=dict(self.receipts.challenge)
        data=await image_bytes(self.page,c['background']);(folder/'background.png').write_bytes(data)
        image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('题图无法解码')
        if self.kind=='font':
            found=re.search(r'请点击\s*(\d+)\s*个',c['prompt'])
            count=int(found.group(1)) if found else 0
            prediction=await asyncio.to_thread(odd_font,self.vision,image,count,engine=self.font_engine)
        else:prediction=await asyncio.to_thread(order_words,self.vision,image)
        return c,{'background':image},prediction

    async def act(self,c,images,prediction,interaction,wrong=False):
        if not await self.current(c):raise RuntimeError('题面已变化，丢弃旧答案')
        box=await self.background.bounding_box()
        if not box:raise RuntimeError('题图不可见')
        await interaction.pause('observe',.75,1.25)
        points=prediction.points
        if wrong:
            count=1 if self.kind=='font' else len(prediction.detail.get('boxes',[]))
            if not 1<=count<=8:raise RuntimeError('无法确定错误对照的点击次数')
            points=[[15+i*18,80] for i in range(count)]
        for x,y in points:
            if not await self.current(c):raise RuntimeError('点击途中题面变化')
            px=box['x']+x*box['width']/images['background'].shape[1]
            py=box['y']+y*box['height']/images['background'].shape[0]
            await interaction.move(px,py,box=box);await interaction.click()
            await interaction.pause('between_clicks',.25,.55)

    async def ui(self,key):
        if not self.receipts.challenge or self.receipts.challenge['key']!=key:return ''
        text=await self.page.locator('#KgControl').inner_text()
        if '验证成功' in text or '验证通过' in text:return 'passed'
        if any(x in text for x in ('验证失败','请重试','验证不通过')):return 'failed'
        return ''
