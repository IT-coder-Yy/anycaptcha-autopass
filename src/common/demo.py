"""两个官方演示站的浏览器适配；所有验证均经页面正常事件提交。"""
import asyncio
import base64
from dataclasses import asdict
import hashlib
import json
import os
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from playwright.async_api import async_playwright

from common.evidence import Budget, Evidence, Verdict
from common.evaluation import summarize_reports
from common.interaction import Interaction, can_retry
from common.vision import Vision, plain_json
from slider.solver import predict_slider, settings as slider_settings
import common.yidun as yidun


def write_json(path, value):
    path.write_text(json.dumps(plain_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def record_run_config(folder, root, site, kind, attempts, seconds, headless, interaction=None, mode='trigger'):
    from common.text_ocr import settings
    write_json(folder/'config.json',{'site':site,'kind':kind,'max_attempts':attempts,'seconds':seconds,
        'headless':headless,'mode':mode,'retry_policy':'bounded-text-recovery' if kind=='text' else 'server-rejection-only',
        'interaction':interaction,'text_engine':os.environ.get('CAPTCHA_TEXT_ENGINE','consensus'),
        'slider_settings':slider_settings(),
        'ocr_settings':settings(),
        'model_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'models').glob('*.onnx'))},
        'text_threshold':float(os.environ.get('CAPTCHA_TEXT_THRESHOLD','.80')),
        'text_ambiguity':float(os.environ.get('CAPTCHA_TEXT_AMBIGUITY','.08')),
        'cpu_threads':int(os.environ.get('CAPTCHA_CPU_THREADS','2')),
        'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((root/'src').rglob('*.py'))}})


def parse_jsonp(text):
    text = text.strip()
    if not text.startswith(("{", "[")):
        text = text[text.find("(") + 1:text.rfind(")")]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("回执必须为 JSON 对象")
    return value


class Receipts:
    def __init__(self, page, site):
        self.site, self.active, self.token = site, None, None
        self.expected_token = None
        self.requests, self.results, self.pending = {}, [], set()
        self.generation, self.generations = 0, []
        page.on("request", self.request)
        page.on("response", self.schedule)

    def request(self, request):
        parts = urlsplit(request.url)
        path = parts.path
        host = parts.hostname or ""
        allowed = host.endswith(".kgcaptcha.com") if self.site=="kg" else host in {"c.dun.163.com","c.dun.163yun.com"}
        if self.active and allowed and path in {"/requestFront", "/api/v3/check"}:
            self.requests[request] = (self.active, self.expected_token or self.token, time.time())

    def schedule(self, response):
        if urlsplit(response.url).path not in {"/requestFront", "/api/v3/check", "/api/v3/get"}:
            return
        task = asyncio.create_task(self.response(response))
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)

    async def response(self, response):
        parts = urlsplit(response.url)
        path,host = parts.path,parts.hostname or ''
        allowed = host.endswith('.kgcaptcha.com') if self.site=='kg' else host in {'c.dun.163.com','c.dun.163yun.com'}
        if not allowed:return
        attempt, token, requested = self.requests.pop(response.request, (None, None, None)) if path!='/api/v3/get' else (None,None,None)
        try:
            body = parse_jsonp(await response.text())
            data = body.get("data") or {}
            if path == "/api/v3/get":
                if response.status==200 and body.get('error')==0 and data.get('token'):
                    self.token = data['token']
                    self.generation+=1
                    self.generations.append({'generation':self.generation,'received_at':time.time(),
                        'challenge_token_sha256':hashlib.sha256(self.token.encode()).hexdigest()})
                return
            if not attempt:
                return
            accepted = None
            item = {"attempt_id": attempt, "path": path, "http_status": response.status,
                    "requested_at":requested,"received_at":time.time()}
            if path == "/api/v3/check":
                query_token = parse_qs(urlsplit(response.request.url).query).get("token", [None])[0]
                response_token = data.get("token")
                observed=[v for v in (query_token,response_token) if v is not None]
                matched=bool(token and observed and all(v==token for v in observed))
                item.update({"challenge_matched": matched, "error": body.get("error"),
                             "result": data.get("result"), "validate_present": bool(data.get("validate"))})
                if matched and response.status == 200 and body.get("error") == 0:
                    if data.get("result") is False:
                        accepted = False
                    elif data.get("result") is True and data.get("validate"):
                        accepted = True
            elif path == "/requestFront":
                item.update({"code": body.get("code"), "message": str(body.get("msg", ""))[:120],
                             "token_present": bool(body.get("token"))})
                if response.status == 200 and type(body.get("code")) is int:
                    accepted = False if body["code"] != 0 else (True if body.get("token") else None)
            item["accepted"] = accepted
            self.results.append(item)
        except Exception as exc:
            if attempt:
                self.results.append({"attempt_id": attempt, "path": path, "accepted": None,
                                     "error": type(exc).__name__})


async def image_bytes(page, source):
    if source.startswith("data:"):
        header, data = source.split(",", 1)
        if ";base64" not in header:
            raise ValueError("图片编码不是 base64")
        return base64.b64decode(data, validate=True)
    response = await page.request.get(source, timeout=15000)
    if not response.ok:
        raise RuntimeError(f"题图读取失败 HTTP {response.status}")
    data = await response.body()
    if len(data) > 5_000_000:
        raise ValueError("题图超出大小限制")
    return data


async def open_challenge(page, site, kind, mode='trigger'):
    opening={}
    if site == "kg":
        await page.goto("https://www.kgcaptcha.com/demo/", wait_until="domcontentloaded")
        await page.get_by_role("link").filter(has_text="文字点选" if kind == "text" else "滑动拼图").click()
        ready = "#KgControl b" if kind == "text" else "#KgBasemap > img[src^='data:']"
        await page.locator(ready).wait_for(state="visible")
    else:
        target = "picture-click" if kind == "text" else "jigsaw"
        await page.goto(f"https://dun.163.com/trial/{target}", wait_until="domcontentloaded")
        await page.locator(".yidun_control").wait_for(state="visible")
        if kind=='slider': opening=await yidun.select_mode(page,kind,mode)
        else: await show_yidun(page,kind)
    await page.wait_for_timeout(400)
    return opening


async def show_yidun(page, kind):
    if kind=='slider': return await yidun.show(page,kind)
    control = page.locator(".yidun_control")
    # 将触发条放在视口中央，给其上方的浮层留出完整空间。
    await control.evaluate("e=>e.scrollIntoView({block:'center',inline:'nearest',behavior:'instant'})")
    await page.wait_for_function("() => {const e=document.querySelector('.yidun_bg-img'); return e && e.complete && e.naturalWidth>0}",timeout=15000)
    bg=page.locator(".yidun_bg-img")
    ready = ".yidun_tips__point" if kind == "text" else ".yidun_jigsaw"
    opaque="""() => {let e=document.querySelector('.yidun_bg-img');
        if(!e || !e.complete || !e.naturalWidth || !e.getBoundingClientRect().width)return false;
        while(e){const s=getComputedStyle(e);if(s.display==='none'||s.visibility==='hidden'||Number(s.opacity)<.99)return false;e=e.parentElement;}return true;}"""
    for _ in range(3):
        if await bg.is_visible() and await page.locator(ready).is_visible():
            try:
                await page.wait_for_function(opaque,timeout=1500)
                return
            except Exception:
                pass
        # 页面控件可能早于事件处理器出现；跨过边界重新触发 mouseenter。
        box=await control.bounding_box()
        if not box: raise ValueError("验证条不可见")
        await page.mouse.move(max(2,box['x']-20),box['y']+box['height']/2)
        await control.hover()
        if kind=="text" and not await bg.is_visible():
            await control.click()
        try:
            await bg.wait_for(state="visible",timeout=2000)
            await page.locator(ready).wait_for(state="visible",timeout=2000)
            await page.wait_for_function(opaque,timeout=2000)
            await page.wait_for_timeout(100)
            return
        except Exception:
            continue
    raise RuntimeError("验证码图片已加载，但浮动面板未能展开")


async def source_id(page, site, kind):
    from text.demo import normalize_prompt
    if site == "kg":
        if kind == "text":
            source = await page.locator("#KgDisplay img[src*='strbasemap']").get_attribute("src")
            prompt = await page.locator("#KgControl b").inner_text()
        else:
            source = await page.locator("#KgBasemap").evaluate("e => getComputedStyle(e).backgroundImage")
            prompt = await page.locator("#KgBasemap > img[src^='data:']").get_attribute("src")
    else:
        source = await page.locator(".yidun_bg-img").get_attribute("src")
        prompt = await page.locator(".yidun_tips__point").inner_text() if kind == "text" else await page.locator(".yidun_jigsaw").get_attribute("src")
    # innerText 在浮层开合时可能增减前导空格；只有字符和顺序变化才是新提示。
    if kind=='text': prompt=normalize_prompt(prompt)
    return hashlib.sha256((str(source) + '\0' + str(prompt)).encode()).hexdigest()


async def wait_result(page,site,receipts,attempt,fingerprint,seconds):
    deadline=time.monotonic()+seconds
    last=None
    ui_classes=""
    text=""
    while time.monotonic()<deadline:
        loc=page.locator("#KgControl" if site=="kg" else ".yidun_control")
        # 凯格滑块的提示条与点字不同。
        if site=="kg" and not await loc.count(): loc=page.locator("#KgSlide")
        text=(await loc.inner_text()).strip() if site=='kg' and await loc.count() else ""
        completed=False
        if site=="yidun":
            ui=await page.evaluate("""()=>{const root=document.querySelector('.yidun');return {
              classes:root?.className||'',text:root?.querySelector('.yidun_control')?.innerText||''}}""")
            ui_classes,text=ui['classes'],ui['text'].strip()
            observed=await yidun.trace(page)
            terminal=observed and observed['terminal']
            completed=bool(terminal and terminal['same_image'])
            if completed:ui_classes=terminal['classes']
        success="验证成功" in text or "验证通过" in text
        failure=any(x in text for x in ("验证失败","请重试","验证不通过"))
        if site=="yidun":
            success="yidun--success" in ui_classes.split()
        own=[r for r in receipts.results if r.get("attempt_id")==attempt]
        response=own[-1] if own else {}
        active=await page.locator(".yidun_bg-img").is_visible() if site=="yidun" else not success
        last=Evidence(attempt,fingerprint,active,response.get("accepted"),response.get("attempt_id"),success,failure,
                      challenge_completed=completed)
        verdict=last.verdict()
        if verdict!=Verdict.UNKNOWN:
            return verdict,asdict(last)|{"observed_ui_classes":ui_classes,"observed_ui_text":text},own
        await page.wait_for_timeout(150)
    detail=asdict(last) if last else {}
    detail.update({"observed_ui_classes":ui_classes,"observed_ui_text":text})
    return Verdict.UNKNOWN,detail,[r for r in receipts.results if r.get("attempt_id")==attempt]


async def refresh(page,site,kind,previous,wait_for_site=False,receipts=None,previous_generation=None):
    started=time.monotonic()
    def new_generation():
        return receipts is not None and previous_generation is not None and receipts.generation>previous_generation
    # 答错后给站点自动换题留出时间，已经出现的新题不再额外刷新。
    for check in range(8 if wait_for_site else 1):
        if check: await page.wait_for_timeout(150)
        try:
            current=await source_id(page,site,kind)
            if current!=previous or new_generation():
                if site=="yidun": await show_yidun(page,kind)
                await page.wait_for_timeout(400)
                return {'source':'site','previous_challenge_id':previous,'challenge_id':current,
                        'elapsed_ms':(time.monotonic()-started)*1000}
        except Exception:
            pass
    if site=="kg":
        await page.locator("#captchaBox img[src$='/refresh.png']").click()
    else:
        for _ in range(3):
            await show_yidun(page,kind)
            try:
                await page.locator(".yidun_refresh").click(timeout=2500)
                break
            except Exception:
                continue
        else:
            raise RuntimeError("刷新控件在浮层动画后仍不可操作")
    for _ in range(40):
        await page.wait_for_timeout(150)
        try:
            if await source_id(page,site,kind)!=previous or new_generation(): break
        except Exception:
            continue
    if site=="yidun": await show_yidun(page,kind)
    if await source_id(page,site,kind)==previous and not new_generation():
        raise RuntimeError("刷新后题面未更新")
    return {'source':'program','previous_challenge_id':previous,
            'challenge_id':await source_id(page,site,kind),'elapsed_ms':(time.monotonic()-started)*1000}


async def run_demo(root,site,kind,cases=3,max_attempts=1,seconds=120,headless=False,negative=False,*,mode='trigger'):
    from text.demo import text_input, click_image_point
    from slider.demo import slider_input, drag_feedback
    if site not in {"kg","yidun"} or kind not in {"text","slider"}:
        raise ValueError("尚未验证的站点或题型")
    if not 1<=cases<=10:
        raise ValueError("单次小样本评测限定 1–10 轮")
    if not 1<=max_attempts<=3:
        raise ValueError("每轮题数限定 1–3 道，避免无界刷新")
    if mode not in yidun.MODES or (mode!='trigger' and (site!='yidun' or kind!='slider')):
        raise ValueError('展示模式仅支持易盾滑块')
    strategy=os.environ.get('CAPTCHA_YIDUN_DRAG','calibrated')
    if strategy not in {'segmented','calibrated'}:raise ValueError('未知易盾拖动策略')
    run_dir=root/"runtime"/f"{site}-{kind}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True,mode=0o700)
    vision=Vision(root/"models")
    reports=[]
    async with async_playwright() as pw:
        context=await pw.chromium.launch_persistent_context(str(root/"runtime"/f"{site}-test-profile"),
                 headless=headless,viewport={"width":1320,"height":900},locale="zh-CN")
        page=context.pages[0]
        interaction=Interaction(page)
        await interaction.install()
        record_run_config(run_dir,root,site,kind,max_attempts,seconds,headless,interaction.config|{'yidun_drag':strategy},mode)
        page.set_default_timeout(15000)
        receipts=Receipts(page,site)
        try:
            for case in range(cases):
                report={"case":case+1,"site":site,"kind":kind,"mode":mode,"negative_control":negative,
                        "attempts":[],"refreshes":[],"verdict":"unknown"}
                budget=Budget(max_attempts,seconds)
                try:
                    async with asyncio.timeout(budget.remaining):
                        report['opening']=await open_challenge(page,site,kind,mode)
                        while budget.take():
                            if site=="yidun": await show_yidun(page,kind)
                            attempt=uuid.uuid4().hex
                            interaction.actions=[]
                            folder=run_dir/f"case-{case+1}-attempt-{budget.used}"
                            folder.mkdir()
                            receipts.active=None
                            entry={"attempt_id":attempt,"challenge_id":None,"outcome":"unknown"}
                            report["attempts"].append(entry)
                            fingerprint=await source_id(page,site,kind)
                            entry['challenge_id']=fingerprint
                            if site=='yidun':
                                entry['generation']=receipts.generation
                                entry['challenge_token_sha256']=hashlib.sha256((receipts.token or '').encode()).hexdigest()
                                if kind=='slider':await yidun.start_trace(page)
                            if kind=="text":
                                image,hints,clip,b,hint_text=await text_input(page,site,folder)
                                region=[b['x']-clip['x'],b['y']-clip['y'],b['width'],b['height']]
                                prediction=await asyncio.to_thread(vision.text,image,None,None,hints,region,hint_text,
                                                                  "ddddocr" if site=="kg" else "default")
                            else:
                                bi,pi,b,p,handle,piece,scale=await slider_input(page,site,folder)
                                prediction=await asyncio.to_thread(predict_slider,bi,pi,site,
                                    (p['x']-b['x'])/scale,(p['y']-b['y'])/scale)
                                move=prediction.value
                                await page.screenshot(path=str(folder/"before.png"))
                            entry["prediction"]=prediction.to_dict()
                            write_json(folder/"prediction.json",entry["prediction"])
                            if await source_id(page,site,kind)!=fingerprint or (site=='yidun' and receipts.generation!=entry['generation']):
                                entry["outcome"]="stale_challenge"
                            elif not prediction.usable and not negative:
                                entry["outcome"]="low_confidence"
                            else:
                                receipts.active=attempt
                                receipts.expected_token=receipts.token
                                await interaction.pause('observe',.75,1.25)
                                if kind=="text":
                                    points=prediction.points if not negative else [[15+i*20,80] for i in range(len(hints))]
                                    for x,y in points:
                                        await click_image_point(page,site,clip['x']+x-b['x'],clip['y']+y-b['y'],fingerprint,interaction)
                                        await interaction.pause('between_clicks',.25,.55)
                                else:
                                    if site=="yidun":
                                        # 识别与观察停顿期间浮层可能收起，拖动前恢复当前题并重新读取位置。
                                        await show_yidun(page,'slider')
                                        if await source_id(page,site,kind)!=fingerprint or receipts.generation!=entry['generation']:
                                            raise RuntimeError('拖动前题面变化')
                                        p=await piece.bounding_box()
                                        if not p: raise RuntimeError('拖动前碎片仍不可见')
                                    target=p['x']+(move*scale if not negative else 35)
                                    diagnostic={} if site=='yidun' else None
                                    if diagnostic is not None:entry['drag_diagnostic']=diagnostic
                                    entry["drag"]=await drag_feedback(page,handle,piece,target,interaction,
                                        diagnostic=diagnostic,strategy=strategy if site=='yidun' else 'segmented')
                                verdict,evidence,result=await wait_result(page,site,receipts,attempt,fingerprint,min(12,budget.remaining))
                                entry.update({"outcome":verdict.value,"evidence":evidence,"receipts":result})
                                await page.screenshot(path=str(folder/"after.png"))
                                report["verdict"]=verdict.value
                            receipts.active=None
                            entry['interaction']=interaction.actions
                            if site=='yidun' and kind=='slider':entry['ui_trace']=await yidun.trace(page,stop=True)
                            report['verdict']=entry['outcome'] if entry['outcome'] in {'passed','failed'} else 'unknown'
                            write_json(folder/"result.json",entry)
                            print(json.dumps({"case":case+1,"attempt":budget.used,"outcome":entry['outcome'],"score":prediction.score},ensure_ascii=False),flush=True)
                            if (negative and kind!='text') or not can_retry(entry,allow_unanswered=kind=='text'): break
                            if budget.used<max_attempts and budget.remaining>1:
                                event=await refresh(page,site,kind,fingerprint,wait_for_site=entry['outcome']=='failed',
                                    receipts=receipts if site=='yidun' else None,previous_generation=entry.get('generation'))
                                report['refreshes'].append(event | {'after_attempt':budget.used,'reason':entry['outcome']})
                            else:
                                break
                except Exception as exc:
                    message=str(exc).splitlines()
                    report["error"]=f"{type(exc).__name__}: {message[0][:200] if message else '操作超时或被中止'}"
                    report["verdict"]="unknown"
                    if report['attempts']:
                        entry=report['attempts'][-1]
                        entry['error']=report['error']
                        entry['interaction']=interaction.actions
                        if receipts.pending:await asyncio.gather(*receipts.pending,return_exceptions=True)
                        entry['receipts']=[r for r in receipts.results if r.get('attempt_id')==entry['attempt_id']]
                        if site=='yidun' and kind=='slider':
                            try:entry['ui_trace']=await yidun.trace(page,stop=True)
                            except Exception:pass
                        write_json(folder/'result.json',entry)
                    receipts.active=None
                    try:
                        if site=='yidun':write_json(run_dir/f'case-{case+1}-error-state.json',await yidun.state(page))
                        await page.screenshot(path=str(run_dir/f"case-{case+1}-error.png"),timeout=3000)
                    except Exception:
                        pass
                report["rounds_used"]=budget.used
                if report['attempts']:
                    report['attempts'][-1].setdefault('interaction',interaction.actions)
                report['first_outcome']=report['attempts'][0]['outcome'] if report['attempts'] else 'unknown'
                reports.append(report)
                write_json(run_dir/"report.json",reports)
                if site=='yidun':write_json(run_dir/'generations.json',receipts.generations)
                print(json.dumps({"case_complete":case+1,"verdict":report['verdict'],"error":report.get('error')},ensure_ascii=False),flush=True)
        finally:
            if receipts.pending: await asyncio.gather(*receipts.pending,return_exceptions=True)
            await context.close()
    summary={"directory":str(run_dir),**summarize_reports(reports)}
    write_json(run_dir/"summary.json",summary)
    return summary
