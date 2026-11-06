"""独立 Python 环境中的抖音验证进程；stdin/stdout 仅传 JSON，不记录 Cookie。"""

import asyncio
import base64
import hashlib
import json
import math
import os
import re
import signal
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from playwright.async_api import async_playwright
from common.interaction import Interaction


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def browser_options(payload):
    # 跨站 iframe 不继承 Playwright 的屏幕模拟，需同时配置 Chromium 原生虚拟屏幕。
    screen = payload["screen"]
    return {"viewport": {"width": 1320, "height": 900}, "locale": "zh-CN",
            "channel": "chromium", "screen": screen,
            **({"user_agent": payload["user_agent"]} if payload.get("user_agent") else {}),
            "args": [f"--screen-info={{{screen['width']}x{screen['height']}}}"]}


async def browser_identity(payload):
    """只访问本地响应，读取当前 Chromium 的身份；不给浏览器拼造其它平台。"""
    async with async_playwright() as pw:
        with tempfile.TemporaryDirectory(prefix="douyin-identity-") as profile:
            context = await pw.chromium.launch_persistent_context(
                profile, headless=True, **browser_options({"screen": payload["screen"]}))
            try:
                page = context.pages[0]
                headers = {}
                async def serve(route):
                    headers.update({k: v for k, v in (await route.request.all_headers()).items()
                                    if k in {"sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform"}})
                    await route.fulfill(content_type="text/html", body="<body>identity</body>")
                await context.route("**/*", serve)
                await page.goto("https://identity.invalid/")
                native = await page.evaluate("""async () => ({
                    ua:navigator.userAgent, platform:navigator.platform, language:navigator.language,
                    cores:navigator.hardwareConcurrency, memory:navigator.deviceMemory,
                    screen:{width:screen.width,height:screen.height},
                    hints:await navigator.userAgentData.getHighEntropyValues(
                        ['architecture','bitness','model','platformVersion','uaFullVersion','fullVersionList','wow64'])
                })""")
                # 与同一二进制的可见模式使用相同 UA；保留真实版本、平台和原生 Client Hints。
                ua = native["ua"].replace("HeadlessChrome/", "Chrome/")
                version = re.search(r"Chrome/([\d.]+)", ua).group(1)
                return {"user_agent": ua, "headers": headers, "metadata": native["hints"],
                        "query": {"browser_platform": native["platform"], "browser_version": version,
                                  "engine_version": version, "browser_language": native["language"],
                                  "os_name": native["hints"]["platform"],
                                  "os_version": native["hints"]["platformVersion"],
                                  "screen_width": native["screen"]["width"], "screen_height": native["screen"]["height"],
                                  "cpu_core_num": native["cores"], "device_memory": native["memory"]}}
            finally:
                await context.close()


def cookie_items(header):
    return [
        {"name": name.strip(), "value": value, "domain": ".douyin.com", "path": "/", "secure": True}
        for item in header.split(";") if "=" in item
        for name, value in [item.strip().split("=", 1)] if name.strip()
    ]


def browser_proxy(value):
    if not value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https", "socks5"} or not parsed.hostname or not parsed.port:
        raise ValueError("代理地址格式无效")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    result = {"server": f"{parsed.scheme}://{host}:{parsed.port}"}
    if parsed.username:
        result.update(username=unquote(parsed.username), password=unquote(parsed.password or ""))
    return result


class Observations:
    """将回执绑定到发起验证请求时的题目，排除刷新后的旧成功回执。"""

    def __init__(self, context):
        self.challenge = None
        self.generation = 0
        self.receipts = []
        self.searches = 0
        self.requests = {}
        self.tasks = set()
        self.closed = False
        self.verify_params = {}
        self.attempts = 0
        self.challenge_errors = []
        context.on("request", self.request)
        context.on("response", self.response)
        context.on("close", lambda _: setattr(self, "closed", True))

    def request(self, request):
        parsed = urlsplit(request.url)
        if parsed.path == "/aweme/v1/web/general/search/single/":
            query = parse_qs(parsed.query)
            self.verify_params = {key: query[key][-1] for key in ("webid", "msToken") if query.get(key)}
        if parsed.path == "/captcha/verify":
            self.requests[request] = self.generation

    def response(self, response):
        path = urlsplit(response.url).path
        if path not in {"/captcha/get", "/captcha/verify", "/aweme/v1/web/general/search/single/"}:
            return
        task = asyncio.create_task(self.capture(response, path))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def capture(self, response, path):
        try:
            data = await response.json()
            if path == "/captcha/get" and data.get("code") != 200:
                self.challenge_errors.append({"code": data.get("code"), "http_status": response.status})
            if path == "/captcha/get" and data.get("code") == 200:
                challenge = data.get("data")
                if isinstance(challenge, dict) and challenge.get("id"):
                    self.challenge = challenge
                    self.generation += 1
            elif path == "/captcha/verify":
                reason = re.search(r"\[(\d+)\]", str(data.get("message", "")))
                self.receipts.append({
                    "generation": self.requests.pop(response.request, None),
                    "code": data.get("code"), "time": time.monotonic(),
                    "reason_code": reason.group(1) if reason else None,
                })
            elif data.get("status_code") == 0 and isinstance(data.get("data"), list) and data["data"]:
                self.searches += 1
        except Exception as exc:
            if path == "/captcha/verify":
                self.receipts.append({"generation": self.requests.pop(response.request, None),
                                      "code": None, "http_status": response.status,
                                      "error_type": type(exc).__name__})
            # 非 JSON 或页面关闭不产生成功证据。
            pass

    async def finish(self):
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


async def challenge_frame(page):
    for frame in page.frames:
        try:
            if await frame.locator("#vc_captcha_box").is_visible():
                return frame
        except Exception:
            continue
    return None


async def stable_boxes(elements):
    """换题后的复位动画结束再测量，避免把动画误差当成拖动倍率。"""
    previous = None
    stable = 0
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        boxes = [await element.bounding_box() for element in elements]
        if all(box and box["width"] > 0 and box["height"] > 0 for box in boxes):
            if previous and all(abs(a[key] - b[key]) < .1 for a, b in zip(boxes, previous) for key in ("x", "y", "width", "height")):
                stable += 1
                if stable >= 3:
                    return boxes
            else:
                stable = 0
            previous = boxes
        else:
            previous, stable = None, 0
        await asyncio.sleep(.1)
    raise TimeoutError("验证码布局未稳定")


async def solve_slide(page, frame, context, observation, record):
    """选择器来自真实抖音中间页；图像算法与动作复用来源程序。"""
    import cv2
    import numpy as np
    from slider.solver import outline_candidate

    generation = observation.generation
    background = frame.locator("#captcha_verify_image")
    piece = frame.locator("#captcha-verify_img_slide")
    handle = frame.locator(".captcha-slider .dragger-item")
    for element in (background, piece, handle):
        await element.wait_for(state="visible", timeout=10000)
    await frame.wait_for_function("() => [...document.querySelectorAll('#captcha_verify_image, #captcha-verify_img_slide')].length === 2 && [...document.querySelectorAll('#captcha_verify_image, #captcha-verify_img_slide')].every(e => e.complete && e.naturalWidth > 0)")
    urls = [await element.get_attribute("src") for element in (background, piece)]
    images = []
    for url, mode in zip(urls, (cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED)):
        response = await context.request.get(url, timeout=10000)
        if not response.ok:
            raise ValueError("题图下载失败")
        images.append(cv2.imdecode(np.frombuffer(await response.body(), np.uint8), mode))
    bg, fg = images
    if bg is None or fg is None:
        raise ValueError("题图解码失败")
    # 仅保留题图以核查识别与坐标；不保存带登录信息的整页或题图 URL。
    if record.get("image_prefix"):
        cv2.imwrite(record["image_prefix"] + "-background.png", bg)
        cv2.imwrite(record["image_prefix"] + "-piece.png", fg)
    bg_box, fg_box, _ = await stable_boxes((background, piece, handle))
    prediction = await asyncio.to_thread(
        outline_candidate, bg, fg,
        (fg_box["x"] - bg_box["x"]) * bg.shape[1] / bg_box["width"],
        (fg_box["y"] - bg_box["y"]) * bg.shape[0] / bg_box["height"],
    )
    record["prediction"] = prediction.to_dict()
    record["display"] = await frame.evaluate("() => ({screen: [screen.width, screen.height], viewport: [innerWidth, innerHeight]})")
    if not prediction.usable:
        return "unanswered"
    interaction = Interaction(page, track_pointer=False, straight_drag=False)
    await interaction.install()
    await interaction.pause("observe", .7, 1.2)
    if observation.generation != generation or urls != [await e.get_attribute("src") for e in (background, piece)]:
        return "changed"
    handle_box = await handle.bounding_box()
    x, y = handle_box["x"] + handle_box["width"] / 2, handle_box["y"] + handle_box["height"] / 2
    distance = prediction.value * bg_box["width"] / bg.shape[1]
    if not 12 < distance <= bg_box["width"] - fg_box["width"] + 1:
        return "unanswered"
    await interaction.move(x, y)
    await page.mouse.down()
    try:
        await interaction.pause("hold", .2, .35)
        await interaction.move(x + 12, y, drag=True)
        await asyncio.sleep(.15)
        partial = await piece.bounding_box()
        ratio = (partial["x"] - fg_box["x"]) / 12
        if not .5 < ratio < 2 or observation.generation != generation:
            return "changed"
        await interaction.move(x + distance / ratio, y, drag=True)
        await interaction.pause("release_hold", .25, .45)
        final = await piece.bounding_box()
        # 页面可能对 CSS 位移取整；按真实碎片位置做最多两次小幅修正。
        pointer_x = x + distance / ratio
        for _ in range(2):
            error = distance - (final["x"] - fg_box["x"])
            if abs(error) <= 1:
                break
            pointer_x += error / ratio
            await interaction.move(pointer_x, y, drag=True)
            await asyncio.sleep(.15)
            final = await piece.bounding_box()
        record["geometry"] = {"target": distance, "actual": final["x"] - fg_box["x"], "ratio": ratio}
        if abs(record["geometry"]["actual"] - distance) > 2:
            return "unknown"
    finally:
        await page.mouse.up()
        record["actions"] = interaction.actions
    return await wait_for_receipt(page, observation, generation, record)


async def wait_for_receipt(page, observation, generation, record):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        current = [r for r in observation.receipts if r["generation"] == generation]
        if current:
            record["receipt"] = current[-1]
            if current[-1]["code"] != 200:
                return "rejected"
            # 回执成功还须对应页面退出挑战，最终由父进程验证原请求。
            if await challenge_frame(page) is None:
                return "accepted"
        if observation.generation != generation:
            return "changed"
        await asyncio.sleep(.2)
    return "unknown"


async def rotation_angle(frame):
    return await frame.evaluate("""() => {
        const angle = e => {const m = new DOMMatrixReadOnly(getComputedStyle(e).transform);
                            return Math.atan2(m.b, m.a) * 180 / Math.PI};
        return angle(document.querySelector('img.captcha-verify-image-slide'))
             - angle(document.querySelector('#captcha_verify_image'));
    }""")


async def solve_rotation(page, frame, context, observation, record):
    """复用来源的双盘识别；实际测量内外盘反向转动的相对角度/位移。"""
    import cv2
    import numpy as np
    from common.vision import Vision

    generation = observation.generation
    background = frame.locator("#captcha_verify_image")
    piece = frame.locator("img.captcha-verify-image-slide")
    handle = frame.locator(".captcha-slider .dragger-item")
    for element in (background, piece, handle):
        await element.wait_for(state="visible", timeout=10000)
    await frame.wait_for_function("() => [...document.querySelectorAll('#captcha_verify_image, img.captcha-verify-image-slide')].length === 2 && [...document.querySelectorAll('#captcha_verify_image, img.captcha-verify-image-slide')].every(e => e.complete && e.naturalWidth > 0)", timeout=20000)
    urls = [await element.get_attribute("src") for element in (background, piece)]
    images = []
    for index, url in enumerate(urls):
        response = await context.request.get(url, timeout=10000)
        if not response.ok:
            raise ValueError("题图下载失败")
        image = cv2.imdecode(np.frombuffer(await response.body(), np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("题图解码失败")
        images.append(image)
        cv2.imwrite(record["image_prefix"] + f"-rotation-{index}.png", image)
    prediction = await asyncio.to_thread(Vision(ROOT).rotate_dual, *images)
    record["prediction"] = prediction.to_dict()
    if not prediction.usable:
        return "unanswered"
    await stable_boxes((background, piece, handle))
    start = await rotation_angle(frame)
    box, track = await handle.bounding_box(), await frame.locator(".captcha-slider").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    interaction = Interaction(page, track_pointer=False, straight_drag=False)
    await interaction.install()
    await interaction.pause("observe", .7, 1.2)
    if observation.generation != generation or urls != [await e.get_attribute("src") for e in (background, piece)]:
        return "changed"
    await interaction.move(x, y)
    await page.mouse.down()
    try:
        await interaction.pause("hold", .2, .35)
        await interaction.move(x + 12, y, drag=True)
        await asyncio.sleep(.15)
        ratio = ((await rotation_angle(frame) - start + 180) % 360 - 180) / 12
        if not .1 < ratio < 10 or observation.generation != generation:
            return "changed"
        displacement = (prediction.value % 360) / ratio
        if not 12 < displacement <= track["width"] - box["width"] + 1:
            return "unanswered"
        await interaction.move(x + displacement, y, drag=True)
        await interaction.pause("release_hold", .25, .45)
        actual = (await rotation_angle(frame) - start) % 360
        error = (prediction.value - actual + 180) % 360 - 180
        record["geometry"] = {"target": prediction.value, "actual": actual, "ratio": ratio,
                              "displacement": displacement, "error": error}
        if abs(error) > 2:
            return "unknown"
    finally:
        await page.mouse.up()
        record["actions"] = interaction.actions
    return await wait_for_receipt(page, observation, generation, record)


async def solve_text(page, frame, context, observation, record):
    from text.douyin import solve

    generation = observation.generation
    outcome = await solve(page, frame, observation, record, stable_boxes=stable_boxes)
    if outcome == "submitted":
        return await wait_for_receipt(page, observation, generation, record)
    if record.get("reason"):
        emit("status", message=record["reason"])
    return outcome


async def solve_semantic(page, frame, context, observation, record):
    from semantic.solver import SemanticError, solve

    generation = observation.generation
    try:
        outcome = await solve(page, frame, observation, record, stable_boxes=stable_boxes)
    except SemanticError as exc:
        record["reason"] = str(exc)
        emit("status", message=str(exc) + "，转人工验证")
        return "unavailable"
    if outcome == "submitted":
        return await wait_for_receipt(page, observation, generation, record)
    return outcome


async def present_http_challenge(page, observation, verification):
    """使用搜索页自身的 TTGCaptcha API 呈现原 HTTP 响应指定的挑战。"""
    try:
        data = json.loads(verification)
    except ValueError:
        data = json.loads(base64.b64decode(verification, validate=True))
    if not isinstance(data, dict):
        raise ValueError("HTTP 验证参数格式无效")
    await page.wait_for_function("() => window.TTGCaptcha?.render && window.TTGCaptcha?.init", timeout=30000)
    emit("status", message="呈现原 HTTP 请求指定的验证码")
    generation = observation.generation
    await page.evaluate("""data => {
        window.verifySDK?.closeCaptcha?.();
        window.TTGCaptcha.init({commonOptions:{aid:6383,did:'0',iid:'0'},
            captchaOptions:{host:'//verify.zijieapi.com/',showMode:'mask',autoClose:true}},()=>{},()=>{});
        window.TTGCaptcha.render({verify_data:data,captchaOptions:{
            ele:'fullscreen_capture',feedbackEle:'fullscreen_capture_feedback',host:'//verify.zijieapi.com/',
            successCb:()=>{},errorCb:()=>{},closeCb:()=>{}}});
    }""", data)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if observation.generation > generation and await challenge_frame(page):
            return
        await asyncio.sleep(.25)
    raise TimeoutError("原 HTTP 验证码未呈现")


async def automatic(page, context, observation, run_dir, *, confirm_original=False):
    solvers = {"slide": ("滑块", solve_slide), "whirl": ("旋转", solve_rotation),
               "semantic_reasoning": ("语义选图", solve_semantic), "text": ("点字", solve_text)}
    for attempt in range(getattr(observation, "attempts", 0) + 1, 4):
        frame = await challenge_frame(page)
        # /captcha/get 比 iframe 布局先完成，不能把尚未挂载的题目当作无题。
        if frame is None and observation.challenge and not observation.searches:
            for _ in range(60):
                await asyncio.sleep(.5)
                frame = await challenge_frame(page)
                if frame is not None or observation.searches:
                    break
        if frame is None:
            return "session_refresh" if observation.searches > 0 else False
        if not observation.challenge or observation.challenge.get("mode") not in solvers:
            emit("status", message="当前题型尚未完成抖音适配，转人工验证")
            return False
        generation, searches = observation.generation, observation.searches
        record = {"attempt": attempt, "kind": observation.challenge["mode"], "challenge_hash": hashlib.sha256(observation.challenge["id"].encode()).hexdigest()[:16]}
        record["image_prefix"] = str(run_dir / f"attempt-{attempt}")
        record["started_at"] = time.time()
        observation.attempts = attempt
        label, solver = solvers[record["kind"]]
        emit("status", message=f"无头{label}验证，第 {attempt}/3 次")
        try:
            record["outcome"] = await solver(page, frame, context, observation, record)
        except BaseException as exc:
            record.update(outcome="interrupted", error_type=type(exc).__name__)
            raise
        finally:
            record["finished_at"] = time.time()
            (run_dir / f"attempt-{attempt}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        emit("status", message=f"第 {attempt}/3 次结果：{record['outcome']}")
        if record["outcome"] == "accepted":
            if confirm_original:
                # 原 HTTP 题已通过，先确认原请求；页面随后出的题不能阻止这次确认。
                return "automatic"
            # 中间页通过后会跳转并更新会话，也可能在搜索页再次出题。
            # 等待这个转换；新题继续消耗本轮剩余次数，不能重置预算。
            for _ in range(30):
                if observation.generation != generation:
                    emit("status", message="验证通过后页面出现新题，继续使用本轮剩余自动次数")
                    break
                if observation.searches > searches:
                    return "automatic"
                await asyncio.sleep(.5)
            else:
                return "automatic"
            continue
        if record["outcome"] not in {"rejected", "unanswered"} or attempt == 3:
            return False
        await asyncio.sleep(2)
        if observation.generation == generation:
            await frame.locator(".vc-captcha-refresh").click(timeout=5000)
        deadline = time.monotonic() + 15
        while observation.generation == generation:
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(.2)
    return False


async def refresh_challenge(page, observation):
    """实测人工刷新可解除中间页；先在无头会话中验证一次同样的恢复动作。"""
    if await challenge_frame(page) is None:
        return False
    generation, searches = observation.generation, observation.searches
    emit("status", message="检测到验证码中间页，先在同一无头会话刷新一次并检查搜索是否恢复")
    await page.reload(wait_until="domcontentloaded", timeout=45000)
    deadline = time.monotonic() + 30
    while observation.generation == generation and observation.searches == searches and time.monotonic() < deadline:
        await asyncio.sleep(.25)
    return observation.searches > searches and await challenge_frame(page) is None


async def run(payload, commands):
    os.umask(0o077)
    timeout = float(payload["timeout"])
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("自动验证时间预算必须为正数")
    profile = Path(payload["profile"])
    profile.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir = profile.parent / "runs" / uuid.uuid4().hex
    run_dir.mkdir(parents=True, mode=0o700)
    report = {"browser_starts": [], "handoffs": [], "identity": payload.get("identity")}
    url = "https://www.douyin.com/search/" + quote(payload["keyword"], safe="") + "?type=general"
    options = browser_options(payload)
    if os.getenv("DOUYIN_CAPTCHA_RECORD_VIDEO", "false").lower() == "true":
        options.update(record_video_dir=str(run_dir / "videos"), record_video_size=options["viewport"])
    videos = []
    if payload.get("proxy"):
        options["proxy"] = browser_proxy(payload["proxy"])
    context = observation = None
    pending_verification = payload.get("verification")

    async def candidate(mode):
        nonlocal pending_verification
        for confirmation in range(3):
            if confirmation:
                emit("status", message="原请求仍要求验证，稍后读取浏览器最新会话重放；不会重新自动作答")
                await asyncio.sleep(3)
            # 验证通过后的页面跳转仍会更新 Cookie，每次确认前必须重新快照。
            cookies = await context.cookies("https://www.douyin.com/")
            header = "; ".join(f"{cookie['name']}={cookie['value']}" for cookie in cookies)
            emit("session", cookies=header, mode=mode, verify_params=observation.verify_params)
            answer = await commands.get()
            recovered = answer.get("recovered") is True
            pending_verification = answer.get("verification")
            report["handoffs"].append({"mode": mode, "recovered": recovered,
                                      "challenge_generation": observation.generation,
                                      "verify_receipts": list(observation.receipts),
                                      "browser_searches": observation.searches,
                                      "verify_param_names": sorted(observation.verify_params)})
            (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            if recovered or not answer.get("retry"):
                return recovered
            if pending_verification:
                return False  # 原请求的新挑战继续共享本轮剩余自动次数。
        return False

    async def present_manual_challenge():
        try:
            await present_http_challenge(page, observation, pending_verification)
        except Exception as exc:
            report["manual_render_error"] = type(exc).__name__
            emit("status", message="原请求的验证码暂未呈现，保留当前浏览器等待人工处理；可刷新页面或取消")

    async def open_browser(pw, headless, seed=False, session_cookies=None):
        nonlocal context, observation
        ctx = await pw.chromium.launch_persistent_context(str(profile), headless=headless, **options)
        obs = Observations(ctx)
        context, observation = ctx, obs
        if seed:
            await ctx.clear_cookies()
            await ctx.add_cookies(cookie_items(payload["cookies"]))
        elif session_cookies is not None:
            # Chromium 重启不会保留 session Cookie，必须转交当前会话的最新值。
            await ctx.add_cookies(session_cookies)
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        if options.get("record_video_dir"):
            videos.append((page.video, {"headless": headless, "started_at": time.time()}))
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        deadline = time.monotonic() + 30
        while not obs.challenge and not obs.searches and time.monotonic() < deadline:
            if pending_verification and await page.evaluate("() => Boolean(window.TTGCaptcha?.render)"):
                break
            await asyncio.sleep(.25)
        report["browser_starts"].append({"headless": headless,
                                        "challenge_type": (obs.challenge or {}).get("mode"),
                                        "browser_searches": obs.searches})
        return ctx, obs, page

    async with async_playwright() as pw:
        try:
            emit("status", message="启动本机无头 Chromium，保留原账号和搜索进度")
            try:
                deadline = time.monotonic() + timeout
                async with asyncio.timeout(timeout):
                    context, observation, page = await open_browser(pw, True, seed=True)
                while True:
                    async with asyncio.timeout(max(.01, deadline - time.monotonic())):
                        if pending_verification:
                            # 若导航先落入中间页，先完成它，再处理原 HTTP 挑战；两者共用三次。
                            if await challenge_frame(page) and not await page.evaluate("() => Boolean(window.TTGCaptcha?.render)"):
                                if not await automatic(page, context, observation, run_dir):
                                    break
                            await present_http_challenge(page, observation, pending_verification)
                            mode = await automatic(page, context, observation, run_dir, confirm_original=True)
                        else:
                            mode = "session_refresh" if await refresh_challenge(page, observation) else await automatic(page, context, observation, run_dir)
                    if mode and await candidate(mode):
                        return
                    if not mode or not pending_verification or observation.attempts >= 3:
                        break
            except Exception as exc:
                report["automatic_error"] = type(exc).__name__
                emit("status", message=f"自动阶段未完成（{type(exc).__name__}）")
            report["automatic_last_challenge_type"] = (observation.challenge or {}).get("mode") if observation else None
            if os.getenv("DOUYIN_CAPTCHA_AUTO_ONLY", "false").lower() == "true":
                report["auto_only_stopped"] = True
                emit("status", message="仅自动验证模式：原请求尚未恢复，保存记录并结束，不打开人工窗口")
                raise RuntimeError("自动验证未恢复原请求")
            session_cookies = await context.cookies() if context and not observation.closed else None
            if context:
                await context.close()
            if observation:
                await observation.finish()

            emit("status", message=f"正在打开同账号浏览器以确认验证状态；任务保持等待，可取消。搜索入口：{url}")
            context, observation, page = await open_browser(pw, False, seed=session_cookies is None, session_cookies=session_cookies)
            if pending_verification:
                await present_manual_challenge()
            manual_frame = await challenge_frame(page)
            if manual_frame:
                try:
                    await manual_frame.locator("#vc_captcha_box").screenshot(path=str(run_dir / "manual-challenge.png"), timeout=5000)
                    handle = manual_frame.locator(".captcha-slider .dragger-item")
                    if await handle.is_visible():
                        handle_state = await handle.evaluate("e => { const r=e.getBoundingClientRect(); const hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2); return {pointer_events:getComputedStyle(e).pointerEvents,handle_receives_pointer:hit===e || e.contains(hit)}; }")
                        (run_dir / "manual-ui.json").write_text(json.dumps(handle_state))
                except Exception:
                    pass  # 诊断截图失败不影响人工操作。
                emit("status", message="已确认验证码页面，请在此窗口完成验证；原搜索请求仍在等待")
            elif not observation.searches:
                emit("status", message="尚未检测到验证码或有效搜索，请检查窗口是否要求登录；当前请求未恢复")
            checked = None
            while not observation.closed:
                accepted = tuple(r["generation"] for r in observation.receipts if r["code"] == 200 and r["generation"] == observation.generation)
                fingerprint = (accepted, observation.searches)
                if (accepted or observation.searches) and fingerprint != checked and await challenge_frame(page) is None:
                    checked = fingerprint
                    if await candidate("manual" if accepted else "manual_session_refresh"):
                        return
                    if pending_verification:
                        await present_manual_challenge()
                    emit("status", message="原搜索请求仍未恢复，请在此窗口继续处理或刷新页面；自动次数不会重置")
                await asyncio.sleep(1)
            raise RuntimeError("人工验证窗口已关闭，原搜索尚未恢复")
        finally:
            if observation:
                report["challenge_errors"] = getattr(observation, "challenge_errors", [])
                report["last_challenge_type"] = (observation.challenge or {}).get("mode")
            (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            if context:
                await context.close()
            if observation:
                await observation.finish()
            if videos:
                report["videos"] = []
                for video, metadata in videos:
                    try:
                        report["videos"].append({**metadata, "path": str(await video.path())})
                    except Exception as exc:
                        report["videos"].append({**metadata, "error_type": type(exc).__name__})
                (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


async def main():
    reader = asyncio.StreamReader(limit=1024 * 1024)
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    commands = asyncio.Queue()
    payload = json.loads(await reader.readline())
    if payload.get("operation") == "identity":
        try:
            emit("identity", identity=await browser_identity(payload))
        finally:
            transport.close()
        return
    task = asyncio.create_task(run(payload, commands))

    async def listen():
        try:
            while line := await reader.readline():
                await commands.put(json.loads(line))
        finally:
            task.cancel()

    listener = asyncio.create_task(listen())
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await task
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)
        transport.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as exc:
        emit("error", message=f"本地验证码进程失败（{type(exc).__name__}），请检查浏览器和运行环境")
        raise SystemExit(1)
