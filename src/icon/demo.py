"""易盾图标演示：可见区域截图、本地识别、绑定当前题目的回执与有限重试。"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid

import cv2
import numpy as np
from playwright.async_api import async_playwright

import common.yidun as yidun
from common.demo import Receipts, parse_jsonp, record_run_config, write_json
from common.evaluation import summarize_reports
from common.evidence import Budget
from icon.solver import click_icons
from common.interaction import Interaction, can_retry


class IconReceipts(Receipts):
    def __init__(self, page):
        super().__init__(page, 'yidun')
        self.challenge = None

    async def response(self, response):
        await super().response(response)
        url = urlsplit(response.url)
        if url.hostname not in {'c.dun.163.com', 'c.dun.163yun.com'} or url.path != '/api/v3/get':
            return
        try:
            body = parse_jsonp(await response.text())
            data = body.get('data') or {}
            if response.status == 200 and body.get('error') == 0 and data.get('token') == self.token:
                sources = data.get('bg')
                if isinstance(sources, list) and all(isinstance(s, str) for s in sources):
                    self.challenge = {'token': self.token, 'generation': self.generation, 'sources': sources}
        except (TypeError, ValueError):
            pass


def icon_verdict(receipt, attempt, current, ui_success):
    if not current or not receipt or receipt.get('attempt_id') != attempt or not receipt.get('challenge_matched'):
        return 'unknown'
    if receipt.get('accepted') is False:
        return 'failed'
    if receipt.get('accepted') is True and ui_success:
        return 'passed'
    return 'unknown'


class IconAdapter:
    def __init__(self, page, engine):
        self.page, self.engine = page, engine
        self.book = IconReceipts(page)
        self.background = page.locator('.yidun_bgimg')
        self.hint = page.locator('.yidun_tips__answer')
        self.panel = page.locator('.yidun_panel')

    async def open(self):
        self.book.active = None
        self.book.challenge = None
        old_generation = self.book.generation
        await self.page.goto('https://dun.163.com/trial/icon-click', wait_until='domcontentloaded')
        control = self.page.locator('.yidun_control')
        await control.wait_for(state='visible')
        await control.evaluate("e=>e.scrollIntoView({block:'center',behavior:'instant'})")
        await control.hover()
        await self.page.wait_for_timeout(1500)
        await control.click()
        await self.ready(old_generation)

    async def ready(self, old_generation):
        control = self.page.locator('.yidun_control')
        previous = None
        stable = 0
        for index in range(100):
            if index % 10 == 0:
                box = await control.bounding_box()
                if box:
                    await self.page.mouse.move(max(2, box['x']-20), box['y']+box['height']/2)
                    await control.hover()
            c = self.book.challenge
            if c and c['generation'] > old_generation and c['generation'] == self.book.generation:
                snapshot = await self.page.locator('.yidun').evaluate('''e => {
                    const bg=e.querySelector('.yidun_bg-img'), hint=e.querySelector('.yidun_tips__img');
                    const visible=n=>{for(let p=n;p;p=p.parentElement){const s=getComputedStyle(p);
                        if(s.display==='none'||s.visibility==='hidden'||Number(s.opacity)<.99)return false;}
                        return !!n?.getBoundingClientRect().width;};
                    return {source:bg?.src,hint:hint?.src,decoded:!!(bg?.complete&&bg.naturalWidth&&hint?.complete&&hint.naturalWidth),
                        visible:visible(bg)&&visible(hint),classes:e.className};
                }''')
                blocked = {'yidun--loading', 'yidun--error', 'yidun--verifying', 'yidun--success'}
                valid = (snapshot['source'] in c['sources'] and snapshot['source'] == snapshot['hint']
                         and snapshot['decoded'] and snapshot['visible'] and not blocked.intersection(snapshot['classes'].split()))
                stable = stable+1 if valid and snapshot == previous else 0
                previous = snapshot
                if stable >= 2:
                    return
            await self.page.wait_for_timeout(100)
        raise RuntimeError('易盾图标题图、提示与当前题目凭据未就绪')

    async def current(self, challenge):
        return (self.book.token == challenge['token'] and self.book.generation == challenge['generation']
                and await self.page.locator('.yidun_bg-img').get_attribute('src') == challenge['source']
                and await self.page.locator('.yidun_tips__img').get_attribute('src') == challenge['source'])

    async def refresh(self, challenge, after_outcome):
        self.book.active = None
        if after_outcome == 'failed':
            for _ in range(20):
                if self.book.generation > challenge['generation']:
                    break
                await self.page.wait_for_timeout(100)
        if self.book.generation > challenge['generation']:
            source = 'automatic'
        else:
            await self.page.locator('.yidun_control').hover()
            await self.page.locator('.yidun_refresh').click()
            source = 'explicit_refresh'
        await self.ready(challenge['generation'])
        return source

    async def read(self, folder):
        challenge = dict(self.book.challenge)
        challenge['source'] = await self.page.locator('.yidun_bg-img').get_attribute('src')
        if challenge['source'] not in challenge['sources']:
            raise RuntimeError('图片不属于当前题目')
        images, boxes = {}, {}
        for name, element in [('background', self.background), ('hint', self.hint)]:
            boxes[name] = await element.bounding_box()
            data = await element.screenshot(path=str(folder/f'{name}.png'))
            images[name] = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if not await self.current(challenge):
            raise RuntimeError('截图期间题目变化')
        ignored = []
        # 工具按钮是页面覆盖层，不是题图内的目标；根据可见 DOM 布局记录其坐标。
        toolbar = self.page.locator('.yidun_top__right')
        if await toolbar.count() == 1 and await toolbar.is_visible():
            box = await toolbar.bounding_box()
            b = boxes['background']
            sx, sy = images['background'].shape[1]/b['width'], images['background'].shape[0]/b['height']
            ignored.append([(box['x']-b['x'])*sx, (box['y']-b['y'])*sy, box['width']*sx, box['height']*sy])
        if self.engine == 'baseline':
            prediction = await asyncio.to_thread(click_icons, images['background'], images['hint'])
        elif self.engine == 'contour-v1':
            from icon.yidun_v1 import click_yidun_icons
            prediction = await asyncio.to_thread(click_yidun_icons, images['background'], images['hint'], ignored)
        else:
            from icon.yidun import click_yidun_icons
            prediction = await asyncio.to_thread(click_yidun_icons, images['background'], images['hint'], ignored)
        write_json(folder/'input.json', {'boxes': boxes, 'ignored_rects': ignored, 'engine': self.engine})
        return challenge, images, prediction

    async def act(self, challenge, images, prediction, interaction, wrong=False):
        if not await self.current(challenge):
            raise RuntimeError('点选前题面变化')
        await yidun.start_trace(self.page)
        # 错误对照须避开已有点选标记，否则邻近点击会撤销标记，无法提交三点。
        height, width = images['background'].shape[:2]
        points = prediction.points if not wrong else [[width*x, height*y] for x, y in ((.12, .5), (.5, .75), (.85, .5))]
        await interaction.pause('observe', .4, .7)
        for x, y in points:
            if not await self.current(challenge):
                raise RuntimeError('点选途中题面变化')
            box = await self.background.bounding_box()
            if not box:
                raise RuntimeError('题面不可见')
            await interaction.move(box['x']+x*box['width']/images['background'].shape[1],
                                   box['y']+y*box['height']/images['background'].shape[0])
            await interaction.click()
            await interaction.pause('between_clicks', .25, .45)

    async def result(self, challenge, attempt, seconds):
        deadline = time.monotonic()+seconds
        outcome, own, ui_success = 'unknown', [], False
        while time.monotonic() < deadline:
            own = [r for r in self.book.results if r.get('attempt_id') == attempt]
            trace = await yidun.trace(self.page)
            classes = (await self.page.locator('.yidun').get_attribute('class')).split()
            terminal = trace and trace['terminal'] and trace['terminal']['same_image']
            ui_success = bool('yidun--success' in classes or terminal)
            outcome = icon_verdict(own[-1] if own else None, attempt, await self.current(challenge), ui_success)
            if outcome != 'unknown':
                break
            await self.page.wait_for_timeout(100)
        return outcome, own, ui_success


async def run_yidun_icons(root, cases=3, max_attempts=1, seconds=120, headless=False,
                          negative=False, recovery_control=False, retry_low_confidence=False):
    if not 1 <= cases <= 30 or not 1 <= max_attempts <= 4 or not 0 < seconds <= 600:
        raise ValueError('每批 1–30 轮，每轮最多 4 道、600 秒')
    if negative and recovery_control or recovery_control and max_attempts < 2:
        raise ValueError('错误答案与恢复控制互斥，恢复控制至少两次机会')
    engine = os.environ.get('CAPTCHA_YIDUN_ICON_ENGINE', 'contour')
    if engine not in ('baseline', 'contour-v1', 'contour'):
        raise ValueError('CAPTCHA_YIDUN_ICON_ENGINE 必须是 baseline、contour-v1 或 contour')
    folder = root/'runtime'/f'yidun-icon-{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:6]}'
    folder.mkdir(parents=True, mode=0o700)
    reports = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        page = await browser.new_page(viewport={'width':1360, 'height':1000})
        page.set_default_timeout(15000)
        interaction = Interaction(page)
        await interaction.install()
        adapter = IconAdapter(page, engine)
        book = adapter.book
        record_run_config(folder, root, 'yidun', 'icon', max_attempts, seconds, headless, interaction.config)
        config = json.loads((folder/'config.json').read_text())
        config.update(icon_engine=engine, retry_policy='rejection-or-low-confidence' if retry_low_confidence else 'server-rejection-only')
        write_json(folder/'config.json', config)
        write_json(folder/'experiment.json', {'engine':engine, 'max_retries':max_attempts-1,
                   'retry_low_confidence':retry_low_confidence, 'negative_control':negative, 'recovery_control':recovery_control})
        try:
            for case in range(1, cases+1):
                report = {'case':case, 'site':'yidun', 'kind':'icon', 'attempts':[], 'refreshes':[], 'verdict':'unknown',
                          'negative_control':negative, 'recovery_control':recovery_control}
                budget = Budget(max_attempts, seconds)
                part = None
                try:
                    async with asyncio.timeout(budget.remaining):
                        while budget.take():
                            book.active = None
                            interaction.actions = []
                            part = folder/f'case-{case}-attempt-{budget.used}'
                            part.mkdir()
                            entry = {'attempt_id':uuid.uuid4().hex, 'outcome':'unknown', 'receipts':[]}
                            report['attempts'].append(entry)
                            if budget.used == 1:
                                await adapter.open()
                            else:
                                previous = report['attempts'][-2]['outcome']
                                source = await adapter.refresh(challenge, previous)
                                report['refreshes'].append({'source':source, 'attempt':budget.used, 'after_outcome':previous})
                            challenge, images, prediction = await adapter.read(part)
                            entry.update(challenge_id=hashlib.sha256(challenge['token'].encode()).hexdigest(),
                                         generation=challenge['generation'], prediction=prediction.to_dict())
                            await adapter.panel.screenshot(path=str(part/'before.png'))
                            wrong = negative or recovery_control and budget.used == 1
                            entry['injected_wrong_answer'] = wrong
                            if not await adapter.current(challenge):
                                raise RuntimeError('识别后题面变化')
                            if not prediction.usable and not wrong:
                                entry['outcome'] = 'low_confidence'
                            else:
                                book.active, book.expected_token = entry['attempt_id'], challenge['token']
                                await adapter.act(challenge, images, prediction, interaction, wrong)
                                entry['outcome'], entry['receipts'], entry['ui_success'] = await adapter.result(
                                    challenge, entry['attempt_id'], min(8, budget.remaining))
                                await page.screenshot(path=str(part/'after.png'))
                            book.active = None
                            entry['interaction'] = interaction.actions
                            write_json(part/'result.json', entry)
                            report['verdict'] = entry['outcome'] if entry['outcome'] in ('passed', 'failed') else 'unknown'
                            if negative or not can_retry(entry, allow_unanswered=retry_low_confidence):
                                break
                except Exception as exc:
                    report.update(verdict='unknown', error=f'{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else "操作超时"}')
                    if report['attempts']:
                        report['attempts'][-1].update(outcome='unknown', error=report['error'], interaction=interaction.actions)
                        if part:
                            write_json(part/'result.json', report['attempts'][-1])
                finally:
                    book.active = None
                report.update(rounds_used=budget.used, first_outcome=report['attempts'][0]['outcome'] if report['attempts'] else 'unknown')
                reports.append(report)
                write_json(folder/'report.json', reports)
                print(f"{engine} 第 {case} 轮：首题 {report['first_outcome']}，最终 {report['verdict']}，共 {budget.used} 次", flush=True)
        finally:
            if book.pending:
                await asyncio.gather(*book.pending, return_exceptions=True)
            await browser.close()
    summary = dict(summarize_reports(reports), site='yidun', kind='icon', engine=engine, directory=str(folder),
                   max_attempts=max_attempts, max_retries=max_attempts-1, retry_low_confidence=retry_low_confidence,
                   metric_scope='control' if negative or recovery_control else 'normal')
    write_json(folder/'summary.json', summary)
    return summary
