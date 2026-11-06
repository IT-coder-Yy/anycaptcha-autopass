"""GoCaptcha 官方双图旋转演示；正常页面拖动，识别完全在本地。"""
import asyncio
from dataclasses import asdict
import hashlib
import json
import re
import time
from urllib.parse import parse_qs, urlsplit
import uuid

import cv2
import numpy as np
from playwright.async_api import async_playwright

from common.demo import image_bytes, write_json, record_run_config
from common.evidence import Budget
from common.interaction import Interaction, can_retry
from rotate_overlap.solver import align_overlap


def rotation_verdict(receipt, ui_success, ui_failure=False):
    if receipt.get("accepted") is False:
        return "failed"
    if receipt.get("accepted") is True and receipt.get("challenge_matched") and ui_success:
        return "passed"
    if ui_failure:
        return "failed"
    return "unknown"


class RotationReceipts:
    DATA = "/api/go-captcha-data/rotate-basic"
    CHECK = "/api/go-captcha-check-data/rotate-basic"

    def __init__(self, page):
        self.active = None
        self.challenge = None
        self.requests, self.results, self.pending = {}, [], set()
        page.on("request", self.request)
        page.on("response", self.schedule)

    def request(self, request):
        parts = urlsplit(request.url)
        if self.active and self.challenge and parts.path == self.CHECK and parts.netloc == self.challenge['host']:
            fields = parse_qs(request.post_data or "")
            matched = bool(self.challenge['key']) and fields.get("key", [None])[0] == self.challenge['key']
            self.requests[request] = (self.active, matched)

    def schedule(self, response):
        parts = urlsplit(response.url)
        # 当前官方演示的 API 来源；不接纳页面中其他服务的同名路径。
        if parts.netloc != "47.104.180.148:8081" or parts.path not in {self.DATA, self.CHECK}:
            return
        task = asyncio.create_task(self.response(response))
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)

    async def response(self, response):
        parts = urlsplit(response.url)
        bound = self.requests.pop(response.request, (None, False))
        try:
            body = await response.json()
            if parts.path == self.DATA:
                if response.status == 200 and type(body.get('code')) is int and body['code'] == 0:
                    self.challenge = {'key': body.get('captcha_key'), 'host': parts.netloc,
                                      'image': body.get('image_base64'), 'piece': body.get('thumb_base64')}
                return
            attempt, matched = bound
            if not attempt:
                return
            code = body.get('code')
            accepted = (code == 0) if matched and response.status == 200 and type(code) is int else None
            self.results.append({'attempt_id': attempt, 'path': parts.path, 'http_status': response.status,
                                 'challenge_matched': matched, 'code': code, 'accepted': accepted})
        except Exception as exc:
            if bound[0]:
                self.results.append({'attempt_id': bound[0], 'accepted': None, 'error': type(exc).__name__})


async def run_rotation(root, cases=3, max_attempts=1, seconds=120, headless=False, negative=False):
    if not 1 <= cases <= 10:
        raise ValueError("单批评测限定 1–10 轮")
    folder = root/'runtime'/f"gocaptcha-rotate-{time.strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True, mode=0o700)
    reports = []
    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(str(root/'runtime/gocaptcha-test-profile'),
                  headless=headless, viewport={'width': 1320, 'height': 900}, locale='zh-CN')
        page = context.pages[0]
        interaction=Interaction(page)
        await interaction.install()
        record_run_config(folder,root,'gocaptcha','rotate',max_attempts,seconds,headless,interaction.config)
        page.set_default_timeout(15000)
        receipts = RotationReceipts(page)
        try:
            for case in range(1, cases + 1):
                report = {'case': case, 'site': 'gocaptcha', 'kind': 'rotate', 'attempts': [],
                          'negative_control': negative, 'verdict': 'unknown'}
                budget = Budget(max_attempts, seconds)
                try:
                    async with asyncio.timeout(budget.remaining):
                        while budget.take():
                            interaction.actions=[]
                            receipts.active = None
                            receipts.challenge = None
                            await page.goto('http://gocaptcha.wencodes.com/demo/', wait_until='domcontentloaded')
                            body = page.locator('.gc-rotate-body')
                            await body.scroll_into_view_if_needed()
                            await page.wait_for_function("()=>document.querySelectorAll('.gc-rotate-body img').length===2&&[...document.querySelectorAll('.gc-rotate-body img')].every(e=>e.complete&&e.naturalWidth>0)")
                            image_src = [await el.get_attribute('src') for el in await body.locator('img').all()]
                            for _ in range(30):
                                if receipts.challenge and receipts.challenge['image'] == image_src[0] and receipts.challenge['piece'] == image_src[1]:
                                    break
                                await page.wait_for_timeout(100)
                            else:
                                raise RuntimeError("题图与本轮服务端题目不匹配")
                            attempt = uuid.uuid4().hex
                            part = folder/f'case-{case}-attempt-{budget.used}'
                            part.mkdir()
                            decoded = []
                            for name, src in zip(['background.png', 'piece.png'], image_src):
                                data = await image_bytes(page, src)
                                (part/name).write_bytes(data)
                                decoded.append(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED))
                            prediction = await asyncio.to_thread(align_overlap, *decoded)
                            entry = {'attempt_id': attempt, 'challenge_id': hashlib.sha256(''.join(image_src).encode()).hexdigest(),
                                     'prediction': prediction.to_dict(), 'outcome': 'low_confidence'}
                            report['attempts'].append(entry)
                            await body.screenshot(path=str(part/'before.png'))
                            if prediction.usable or negative:
                                await interaction.pause('observe',.75,1.25)
                                angle = ((prediction.value + 90) % 360) if negative else prediction.value
                                parent = body.locator('..')
                                handle, bar = parent.locator('.gc-drag-block'), parent.locator('.gc-drag-slide-bar')
                                h=await handle.bounding_box()
                                if not h:raise RuntimeError('旋转把手不可见')
                                await interaction.move(h['x']+h['width']/2,h['y']+h['height']/2)
                                await page.wait_for_timeout(200)
                                h = await handle.bounding_box()
                                travel = await bar.evaluate('e=>e.offsetWidth') - await handle.evaluate('e=>e.offsetWidth')
                                x, y = h['x'] + h['width']/2, h['y'] + h['height']/2
                                receipts.active = attempt
                                await interaction.move(x,y)
                                await interaction.pause('before_drag',.12,.25)
                                await page.mouse.down()
                                try:
                                    await interaction.pause('drag_hold',.12,.22)
                                    mouse_x = x + angle*travel/360
                                    await interaction.move(mouse_x,y,drag=True)
                                    entry['drag_feedback'] = []
                                    for _ in range(5):
                                        await page.wait_for_timeout(100)
                                        transform = await parent.locator('.gc-rotate-thumb-block').evaluate('e=>e.style.transform')
                                        match = re.fullmatch(r'rotate\(([-\d.]+)deg\)', transform)
                                        if not match:
                                            raise RuntimeError('无法读取实际旋转角度')
                                        actual = float(match.group(1))
                                        entry['drag_feedback'].append({'desired': angle, 'actual': actual})
                                        if abs(actual-angle) <= 1:
                                            break
                                        mouse_x += (angle-actual)*travel/360
                                        await interaction.move(mouse_x,y,drag=True)
                                    else:
                                        raise RuntimeError('页面旋转角度未达到目标')
                                    entry['observed_transform'] = await parent.locator('.gc-rotate-thumb-block').evaluate('e=>getComputedStyle(e).transform')
                                    await interaction.pause('before_release',.14,.28)
                                finally:
                                    await page.mouse.up()
                                deadline = time.monotonic() + min(10, budget.remaining)
                                result,own,success,failure='unknown',[],False,False
                                while time.monotonic() < deadline:
                                    own = [r for r in receipts.results if r.get('attempt_id') == attempt]
                                    receipt = own[-1] if own else {}
                                    success = await page.get_by_text('check data success', exact=True).is_visible()
                                    failure = await page.get_by_text('check data failed', exact=True).is_visible()
                                    result = rotation_verdict(receipt, success, failure)
                                    if result != 'unknown':
                                        break
                                    await page.wait_for_timeout(100)
                                entry.update({'outcome': result, 'receipts': own, 'ui_success': success,
                                              'ui_failure': failure, 'requested_angle': angle})
                                await page.screenshot(path=str(part/'after.png'))
                            receipts.active = None
                            entry['interaction']=interaction.actions
                            write_json(part/'result.json', entry)
                            report['verdict'] = entry['outcome'] if entry['outcome'] in {'passed','failed'} else 'unknown'
                            print(json.dumps({'case': case, 'attempt': budget.used, 'outcome': entry['outcome'],
                                              'score': prediction.score}, ensure_ascii=False), flush=True)
                            if negative or not can_retry(entry):
                                break
                except Exception as exc:
                    message = str(exc).splitlines()
                    report.update({'verdict': 'unknown', 'error': f"{type(exc).__name__}: {message[0][:200] if message else '操作超时'}"})
                report['rounds_used'] = budget.used
                if report['attempts']:
                    report['attempts'][-1].setdefault('interaction',interaction.actions)
                report['first_outcome']=report['attempts'][0]['outcome'] if report['attempts'] else 'unknown'
                reports.append(report)
                write_json(folder/'report.json', reports)
                print(json.dumps({'case': case, 'verdict': report['verdict']}, ensure_ascii=False), flush=True)
        finally:
            if receipts.pending:
                await asyncio.gather(*receipts.pending, return_exceptions=True)
            await context.close()
    summary = {'site': 'gocaptcha', 'kind': 'rotate', 'cases': cases, 'negative_control': negative,
               'passed': sum(r['verdict']=='passed' for r in reports),
               'failed': sum(r['verdict']=='failed' for r in reports),
               'unknown': sum(r['verdict']=='unknown' for r in reports), 'directory': str(folder)}
    summary['first_attempt_passed']=sum(bool(r['attempts']) and r['attempts'][0]['outcome']=='passed' for r in reports)
    write_json(folder/'summary.json', summary)
    return summary
