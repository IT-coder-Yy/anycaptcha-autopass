"""新题型的有限评测：共享预算、当前题目回执、首题与控制实验分别记录。"""
import asyncio
import base64
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import parse_qs, urlsplit
import uuid

import cv2
import numpy as np
from playwright.async_api import async_playwright

from common.demo import record_run_config, write_json
from common.evaluation import summarize_reports
from common.evidence import Budget
from restore.solver import restore_band
from drag2d.solver import drag_piece
from icon.solver import click_icons
from icon.templates import click_gocaptcha_icons, template_info
from common.interaction import Interaction, can_retry
from lab.server import local_server


def verdict(receipt, attempt_id, challenge_key, ui_state, current=True):
    """无当前回执、旧题或只有 UI 文案都不能通过，也不能据此自动重试。"""
    if not current or not receipt or receipt.get('attempt_id')!=attempt_id:
        return 'unknown'
    if receipt.get('challenge_key')!=challenge_key or not receipt.get('challenge_matched'):
        return 'unknown'
    if receipt.get('accepted') is False: return 'failed'
    if receipt.get('accepted') is True and ui_state=='passed': return 'passed'
    return 'unknown'


class Receipts:
    def __init__(self,page,origin,data_path,check_path,local=False):
        self.origin=origin.rstrip('/');self.data_path=data_path;self.check_path=check_path;self.local=local
        self.challenge=None;self.active=None;self.requests={};self.results=[];self.pending=set()
        page.on('request',self.request);page.on('response',self.schedule)

    def matches(self,url,path):
        p=urlsplit(url)
        return f'{p.scheme}://{p.netloc}'==self.origin and p.path==path

    def request(self,request):
        if not self.active or not self.matches(request.url,self.check_path):return
        try:
            data=json.loads(request.post_data or '{}') if self.local else parse_qs(request.post_data or '')
            key=data.get('key') if self.local else data.get('key',[None])[0]
            attempt,expected=self.active
            self.requests[request]=(attempt,expected,key==expected and self.challenge is not None and self.challenge['key']==expected)
        except (TypeError,ValueError):pass

    def schedule(self,response):
        if not (self.matches(response.url,self.data_path) or self.matches(response.url,self.check_path)):return
        task=asyncio.create_task(self.response(response));self.pending.add(task);task.add_done_callback(self.pending.discard)

    async def response(self,response):
        bound=self.requests.pop(response.request,None)
        try:
            body=await response.json()
            if self.matches(response.url,self.data_path):
                if response.status!=200:return
                if self.local:
                    if isinstance(body.get('key'),str):self.challenge=body
                elif type(body.get('code')) is int and body['code']==0 and body.get('captcha_key'):
                    # 白名单只接收可见题图、起点与尺寸；不把其他字段传入识别器。
                    self.challenge={'key':body['captcha_key'],'background':body.get('image_base64'),
                                    'piece':body.get('tile_base64'),'hint':body.get('thumb_base64')}
                return
            if not bound:return
            attempt,key,matched=bound
            accepted=None
            if matched and response.status==200:
                if self.local and body.get('key')==key and type(body.get('accepted')) is bool:
                    accepted=body['accepted']
                elif not self.local and type(body.get('code')) is int and body['code'] in (0,1):
                    accepted=body['code']==0
            self.results.append({'attempt_id':attempt,'challenge_key':key,'challenge_matched':matched,
                                 'accepted':accepted,'http_status':response.status})
        except Exception as exc:
            if bound:self.results.append({'attempt_id':bound[0],'challenge_key':bound[1],
                                          'accepted':None,'error':type(exc).__name__})


def public_receipt(receipt):
    data = {k:v for k,v in receipt.items() if k!='challenge_key'}
    data['challenge_id'] = hashlib.sha256(receipt['challenge_key'].encode()).hexdigest()
    return data


def decode(source):
    if not isinstance(source,str) or not source.startswith('data:image/'):
        raise ValueError('需要页面实际显示的内嵌题图')
    data=base64.b64decode(source.split(',',1)[1],validate=True)
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
    if image is None:raise ValueError('题图解码失败')
    return image,data


class Adapter:
    def __init__(self,page,site,kind,base_url,scale=1):
        self.page,self.site,self.kind,self.base_url,self.scale=page,site,kind,base_url,scale
        self.icon_engine=os.environ.get('CAPTCHA_ICON_ENGINE','template') if site=='gocaptcha' and kind=='icon' else 'baseline'
        if self.icon_engine not in ('template','baseline'):
            raise ValueError('CAPTCHA_ICON_ENGINE 必须是 template 或 baseline')
        subtype='slide-region' if kind=='drag2d' else 'click-shape'
        origin=base_url.rstrip('/') if site=='lab' else os.environ.get('CAPTCHA_GOCAPTCHA_API_ORIGIN','http://47.104.180.148:8081')
        self.receipts=Receipts(page,origin,'/challenge' if site=='lab' else '/api/go-captcha-data/'+subtype,
                              '/verify' if site=='lab' else '/api/go-captcha-check-data/'+subtype,site=='lab')

    async def open(self):
        self.receipts.active=None;self.receipts.challenge=None
        url=self.base_url+(f'?kind={self.kind}&scale={self.scale}' if self.site=='lab' else '')
        await self.page.goto(url,wait_until='domcontentloaded')
        await self.ready()

    async def ready(self,old_key=None):
        for _ in range(100):
            c=self.receipts.challenge
            if c and c['key']!=old_key:
                if self.site=='lab':
                    self.widget=self.page.locator('main');self.background=self.page.locator('#background')
                else:
                    widgets=self.page.locator('.go-captcha')
                    for index in range(await widgets.count()):
                        w=widgets.nth(index);bg=w.locator('img.gc-picture')
                        if await bg.count()==1 and await bg.get_attribute('src')==c['background']:
                            self.widget,self.background=w,bg;break
                    else:
                        await self.page.wait_for_timeout(100);continue
                if await self.background.get_attribute('src')==c['background']:
                    if not await self.image_ready(self.background,c['background']):
                        await self.page.wait_for_timeout(100);continue
                    if self.kind in ('drag2d','icon'):
                        aux=self.auxiliary()
                        if await aux.count()!=1 or not await self.image_ready(aux,c['piece' if self.kind=='drag2d' else 'hint']):
                            await self.page.wait_for_timeout(100);continue
                    await self.widget.scroll_into_view_if_needed()
                    await self.page.wait_for_timeout(150)
                    if await self.current(c):return
            await self.page.wait_for_timeout(100)
        raise RuntimeError('未取得与页面题图一致的新题目')

    async def refresh(self,old_key):
        self.receipts.active=None
        if self.site=='lab':await self.page.locator('#refresh').click();source='explicit_refresh'
        else:
            # 上游通常 1 秒后自动换题；先等自动刷新，防止又主动挑掉一道题。
            for _ in range(20):
                if self.receipts.challenge and self.receipts.challenge['key']!=old_key:break
                await self.page.wait_for_timeout(100)
            else:await self.widget.locator('.gc-icon-block svg').nth(1).click()
            source='automatic' if self.receipts.challenge and self.receipts.challenge['key']!=old_key else 'explicit_refresh'
        await self.ready(old_key)
        if self.site!='lab':
            # 旧提示必须消失，下一题的全局 toast 才能作为新的 UI 证据。
            for _ in range(80):
                if not await self.toast_visible('check data failed'):break
                await self.page.wait_for_timeout(100)
            else:raise RuntimeError('旧失败提示未消失')
        return source

    def auxiliary(self):
        selector=('#piece' if self.kind=='drag2d' else '#hint') if self.site=='lab' else ('.gc-tile img' if self.kind=='drag2d' else '.gc-header img')
        return self.widget.locator(selector)

    async def image_ready(self,locator,source):
        # 自动换题时 decode() 可能被新的 src 中断；只等待当前图，不再发出刷新请求。
        return await locator.evaluate('''async(e,src)=>{
            if(e.getAttribute('src')!==src)return false;
            try{await e.decode();return e.getAttribute('src')===src && e.naturalWidth>0;}
            catch{return false;}
        }''',source)

    async def toast_visible(self,text):
        return await self.page.get_by_text(text,exact=True).evaluate_all(
            "els=>els.some(e=>e.getClientRects().length && getComputedStyle(e).visibility!=='hidden')")

    async def current(self,c):
        return (self.receipts.challenge is not None and self.receipts.challenge['key']==c['key']
                and await self.background.get_attribute('src')==c['background'])

    async def read(self,folder):
        c=dict(self.receipts.challenge)
        images={}
        for name in ('background','piece','hint'):
            if c.get(name):
                images[name],data=decode(c[name]);(folder/f'{name}.png').write_bytes(data)
        if self.kind in ('drag2d','icon'):
            aux=self.auxiliary()
            if not await self.image_ready(aux,c['piece' if self.kind=='drag2d' else 'hint']):
                raise RuntimeError('提示或碎片与当前题目不一致')
        if self.kind in ('rotate-single','font','order'):
            from common.vision import Vision
            from rotate_single.solver import upright
            if not hasattr(self,'vision'):
                root=Path(os.environ.get('CAPTCHA_PROJECT_ROOT',Path(__file__).resolve().parents[2]))
                self.vision=Vision(root/'models')
            from font.solver import odd_font
            from order.solver import order_words
            solver={'rotate-single':upright,'font':odd_font,'order':order_words}[self.kind]
            prediction=await asyncio.to_thread(solver,self.vision,images['background'])
        elif self.kind=='restore':prediction=await asyncio.to_thread(restore_band,images['background'],c['band'])
        elif self.kind=='drag2d':prediction=await asyncio.to_thread(drag_piece,images['background'],images['piece'])
        else:
            solver=click_gocaptcha_icons if self.icon_engine=='template' else click_icons
            prediction=await asyncio.to_thread(solver,images['background'],images['hint'])
        return c,images,prediction

    async def act(self,c,images,prediction,interaction,wrong=False):
        if not await self.current(c):raise RuntimeError('题面已变化，丢弃旧答案')
        box=await self.background.bounding_box()
        if not box:raise RuntimeError('题图不可见')
        sx,sy=box['width']/images['background'].shape[1],box['height']/images['background'].shape[0]
        await interaction.pause('observe',.3,.5)
        if self.kind in ('icon','font','order'):
            points=prediction.points if not wrong else [[2,2]]
            for x,y in points:
                if not await self.current(c):raise RuntimeError('点选途中题面变化')
                await interaction.move(box['x']+x*sx,box['y']+y*sy);await interaction.click()
                await interaction.pause('between_clicks',.12,.22)
            if not await self.current(c):raise RuntimeError('提交前题面变化')
            await self.widget.locator('#confirm' if self.site=='lab' else '.gc-button-block button').click()
        else:
            handle=self.widget.locator('#handle' if self.kind in ('restore','rotate-single') else '#piece' if self.site=='lab' else '.gc-tile')
            hb=await handle.bounding_box()
            if not hb:raise RuntimeError('拖动对象不可见')
            x,y=hb['x']+hb['width']/2,hb['y']+hb['height']/2
            if self.kind=='rotate-single':
                angle=prediction.value if not wrong else ((prediction.value or 0)+180)%360
                tx,ty=x+angle/360*images['background'].shape[1]*sx,y
            elif self.kind=='restore':
                shift=prediction.value if not wrong else ((prediction.value or 0)+images['background'].shape[1]/2)%images['background'].shape[1]
                tx,ty=x+shift*sx,y
            else:
                px,py=prediction.points[0] if prediction.points and not wrong else (3,3)
                tx,ty=box['x']+px*sx+hb['width']/2,box['y']+py*sy+hb['height']/2
            await interaction.move(x,y)
            if not await self.current(c):raise RuntimeError('按下前题面变化')
            await self.page.mouse.down()
            try:
                await interaction.pause('drag_hold',.08,.16)
                await interaction.move(tx,ty,drag=True)
                if not await self.current(c):raise RuntimeError('拖动途中题面变化')
                await interaction.pause('before_release',.10,.18)
            finally:await self.page.mouse.up()

    async def ui(self,key):
        if self.site=='lab':
            return await self.page.locator('#status').evaluate('(e,key)=>e.dataset.key===key?e.dataset.state:""',key)
        if await self.toast_visible('check data success'):return 'passed'
        if await self.toast_visible('check data failed'):return 'failed'
        return ''


async def run_geometry(root,site,kind,cases=3,max_attempts=1,seconds=120,headless=False,
                       negative=False,recovery_control=False,seed=0,scale=1,retry_low_confidence=False):
    if kind not in ('restore','drag2d','icon','rotate-single','font','order') or site not in ('lab','gocaptcha','kg'):
        raise ValueError('未知新题型或演示站')
    if site=='kg' and kind not in ('font','order'):raise ValueError('本适配仅支持凯格字体与语序')
    if site=='gocaptcha' and kind not in ('drag2d','icon'):raise ValueError('GoCaptcha 不提供横带滑动还原；使用 --site lab')
    if not 1<=cases<=30 or not 1<=max_attempts<=4 or not 0<seconds<=600:
        raise ValueError('单批限定 1–30 轮，每轮 1–4 题，时间不超过 600 秒')
    if not .6<=scale<=1.4:
        raise ValueError('本地题库显示比例必须在 0.6–1.4 之间')
    if negative and recovery_control or recovery_control and max_attempts<2:
        raise ValueError('错误答案对照与恢复对照不能同时启用；恢复对照至少需要 2 次机会')
    folder=root/'runtime'/f'{site}-{kind}-{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:6]}'
    folder.mkdir(parents=True,mode=0o700);reports=[]
    with local_server(seed) if site=='lab' else nullcontext(os.environ.get('CAPTCHA_GOCAPTCHA_URL','http://gocaptcha.wencodes.com/demo/')) as url:
        async with async_playwright() as pw:
            browser=await pw.chromium.launch(headless=headless)
            context=await browser.new_context(viewport={'width':1320,'height':960},locale='zh-CN')
            page=await context.new_page();page.set_default_timeout(15000)
            interaction=Interaction(page);await interaction.install()
            if site=='kg':
                from common.language_demo import KGAdapter
                adapter=KGAdapter(page,kind,root)
            else:adapter=Adapter(page,site,kind,url,scale)
            book=adapter.receipts
            record_run_config(folder,root,site,kind,max_attempts,seconds,headless,interaction.config)
            write_json(folder/'experiment.json',{'seed':seed,'scale':scale,'recovery_control':recovery_control,
                                                'negative_control':negative,'source':site,
                                                'retry_low_confidence':retry_low_confidence,'max_retries':max_attempts-1,
                                                'font_engine':getattr(adapter,'font_engine',os.environ.get('CAPTCHA_FONT_ENGINE','profile')) if kind=='font' else None,
                                                'icon_solver':template_info() if adapter.icon_engine=='template' else {'engine':'baseline'},
                                                'lab_html_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'lab'/'lab.html').read_bytes()).hexdigest()})
            stop_reason=None
            try:
                for case in range(1,cases+1):
                    report={'case':case,'site':site,'kind':kind,'attempts':[],'refreshes':[],
                            'negative_control':negative,'recovery_control':recovery_control,'verdict':'unknown',
                            'retry_low_confidence':retry_low_confidence,'max_attempts':max_attempts}
                    if stop_reason:
                        report.update(error=stop_reason,skipped_due_to_site_limit=True,rounds_used=0,first_outcome='unknown')
                        reports.append(report);write_json(folder/'report.json',reports)
                        continue
                    budget=Budget(max_attempts,seconds);part=None
                    try:
                        async with asyncio.timeout(budget.remaining):
                            while budget.take():
                                interaction.actions=[];book.active=None
                                part=folder/f'case-{case}-attempt-{budget.used}';part.mkdir()
                                entry={'attempt_id':uuid.uuid4().hex,'outcome':'unknown','receipts':[]}
                                report['attempts'].append(entry)
                                if budget.used==1:await adapter.open()
                                else:
                                    source=await adapter.refresh(previous_key)
                                    report['refreshes'].append({'source':source,'attempt':budget.used,
                                                               'after_outcome':report['attempts'][-2]['outcome']})
                                c,images,prediction=await adapter.read(part);previous_key=c['key']
                                entry.update(challenge_id=hashlib.sha256(c['key'].encode()).hexdigest(),prediction=prediction.to_dict())
                                await adapter.widget.screenshot(path=str(part/'before.png'))
                                wrong=negative or recovery_control and budget.used==1
                                entry['injected_wrong_answer']=wrong
                                if not prediction.usable and not wrong:entry['outcome']='low_confidence'
                                else:
                                    book.active=(entry['attempt_id'],c['key'])
                                    await adapter.act(c,images,prediction,interaction,wrong)
                                    deadline=time.monotonic()+min(8,budget.remaining)
                                    while time.monotonic()<deadline:
                                        own=[r for r in book.results if r['attempt_id']==entry['attempt_id']]
                                        ui=await adapter.ui(c['key'])
                                        outcome=verdict(own[-1] if own else None,entry['attempt_id'],c['key'],ui,await adapter.current(c))
                                        entry.update(outcome=outcome,receipts=[public_receipt(r) for r in own],ui_state=ui)
                                        if outcome!='unknown':break
                                        await page.wait_for_timeout(50)
                                    await adapter.widget.screenshot(path=str(part/'after.png'))
                                book.active=None;entry['interaction']=interaction.actions
                                write_json(part/'result.json',entry)
                                report['verdict']=entry['outcome'] if entry['outcome'] in ('passed','failed') else 'unknown'
                                if negative or not can_retry(entry,allow_unanswered=retry_low_confidence):break
                    except Exception as exc:
                        from common.language_demo import DemoUnavailable
                        if isinstance(exc,DemoUnavailable):stop_reason=f'DemoUnavailable: {exc}'
                        report.update(verdict='unknown',error=f'{type(exc).__name__}: {str(exc).splitlines()[0][:250] if str(exc) else "操作超时"}')
                        if report['attempts']:
                            report['attempts'][-1].update(outcome='unknown',error=report['error'],interaction=interaction.actions)
                            if part:write_json(part/'result.json',report['attempts'][-1])
                    report['rounds_used']=budget.used
                    report['first_outcome']=report['attempts'][0]['outcome'] if report['attempts'] else 'unknown'
                    reports.append(report);write_json(folder/'report.json',reports)
                    print(json.dumps({'case':case,'site':site,'kind':kind,'first':report['first_outcome'],
                                      'verdict':report['verdict'],'attempts':budget.used},ensure_ascii=False),flush=True)
            finally:
                if book.pending:await asyncio.gather(*book.pending,return_exceptions=True)
                await browser.close()
    summary=dict(summarize_reports(reports),site=site,kind=kind,directory=str(folder),recovery_control=recovery_control,
                 metric_scope='control' if negative or recovery_control else 'normal',
                 max_attempts=max_attempts,max_retries=max_attempts-1,retry_low_confidence=retry_low_confidence)
    write_json(folder/'summary.json',summary)
    return summary
