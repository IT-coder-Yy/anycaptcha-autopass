"""开发期浏览器采样工具。命令由标准输入逐行 JSON 传入。"""
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".browsers"))
DATA = ROOT / "runtime" / os.environ.get("CAPTCHA_PROBE_DIR", "probe")
DATA.mkdir(parents=True, exist_ok=True, mode=0o700)


def safe_url(url):
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def redact(value, depth=0):
    if depth > 8:
        return "[深层结构省略]"
    if isinstance(value, dict):
        return {k: ("[已隐藏]" if re.search(r"token|secret|sign|cookie|session|ticket|password|device.*id|fingerprint|phone", k, re.I)
                    else redact(v, depth + 1)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(x, depth + 1) for x in value[:8]]
    if isinstance(value, str):
        if value.startswith("data:"):
            return "[内嵌图片]"
        if value.startswith(("http://", "https://")):
            return safe_url(value)
        return value[:500]
    return value


async def main():
    captures = []
    images = {}
    seq = 0
    tasks = set()
    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            str(ROOT / "runtime" / os.environ.get("CAPTCHA_PROFILE", "browser-profile")), headless=False,
            viewport={"width": 1320, "height": 900}, locale="zh-CN",
        )
        page = context.pages[0] if context.pages else await context.new_page()

        async def response(res):
            nonlocal seq
            url = safe_url(res.url)
            is_cap = bool(re.search(r"captcha|verify|verification|yidun|dun\.163|kgcaptcha|turing|sec\.douyin", url, re.I))
            is_search = "/search/" in url and res.request.resource_type in {"xhr", "fetch"}
            content_type = res.headers.get("content-type", "")
            if "image" in content_type and is_cap:
                try:
                    body = await res.body()
                    if len(body) > 4_000_000:
                        return
                    key = hashlib.sha256(body).hexdigest()[:16]
                    path = DATA / (key + ".img")
                    path.write_bytes(body)
                    images[res.url] = str(path)
                except Exception:
                    pass
            elif (is_cap or is_search) and res.request.resource_type in {"xhr", "fetch", "script"}:
                seq += 1
                item = {"seq": seq, "time": time.time(), "url": url, "status": res.status}
                if not is_search and ("json" in content_type or "/api/" in url):
                    try:
                        body = (await res.text())[:150_000]
                        if not body.lstrip().startswith(("{", "[")):
                            body = body[body.find("(") + 1:body.rfind(")")]
                        item["body"] = redact(json.loads(body))
                    except Exception:
                        item["body"] = "[非 JSON 响应]"
                elif is_search:
                    try:
                        body = await res.json()
                        item["summary"] = {"keys": list(body)[:40], "status_code": body.get("status_code"),
                                           "data_count": len(body.get("data", [])) if isinstance(body.get("data"), list) else None,
                                           "verify": redact(body.get("verify", None))}
                    except Exception:
                        pass
                captures.append(item)
                with (DATA / "network.jsonl").open("a") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        def schedule(res):
            task = asyncio.create_task(response(res))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        context.on("response", schedule)
        if len(sys.argv) > 1:
            try:
                await page.goto(sys.argv[1], wait_until="domcontentloaded", timeout=60000)
            except Exception as exc:
                print(json.dumps({"navigation": str(exc)[:300]}, ensure_ascii=False), flush=True)
        print(json.dumps({"ready": True, "url": safe_url(page.url)}), flush=True)
        try:
            while True:
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    break
                try:
                    cmd = json.loads(line)
                    action = cmd["action"]
                    frame = page.frames[cmd.get("frame", 0)]
                    if action == "goto":
                        await page.goto(cmd["url"], wait_until="domcontentloaded", timeout=60000)
                        out = {"url": safe_url(page.url)}
                    elif action == "state":
                        out = {"title": await page.title(), "url": safe_url(page.url),
                               "frames": [{"index": i, "url": safe_url(f.url)} for i, f in enumerate(page.frames)],
                               "text": (await frame.locator("body").inner_text())[:cmd.get("limit", 12000)]}
                    elif action == "elements":
                        out = await frame.locator(cmd["selector"]).evaluate_all("els => els.slice(0,80).map(e => ({tag:e.tagName,id:e.id,cls:e.className,text:e.textContent?.slice(0,160),attrs:Object.fromEntries([...e.attributes].filter(a=>['src','href','type','placeholder','role','aria-label'].includes(a.name)).map(a=>[a.name,a.value])),style:Object.fromEntries(['backgroundImage','cursor','transform','left','top','width','height'].map(k=>[k,getComputedStyle(e)[k]])),box:JSON.parse(JSON.stringify(e.getBoundingClientRect()))}))")
                        out = redact(out)
                    elif action == "click":
                        await frame.locator(cmd["selector"]).nth(cmd.get("index", 0)).click(timeout=10000)
                        out = {"clicked": True}
                    elif action == "hover":
                        await frame.locator(cmd["selector"]).nth(cmd.get("index", 0)).hover(timeout=10000)
                        out = {"hovered": True}
                    elif action == "fill":
                        await frame.locator(cmd["selector"]).fill(cmd["value"], timeout=10000)
                        out = {"filled": True}
                    elif action == "press":
                        await frame.locator(cmd["selector"]).press(cmd["key"], timeout=10000)
                        out = {"pressed": True}
                    elif action == "shot":
                        path = DATA / f"shot-{int(time.time())}.png"
                        if cmd.get("selector"):
                            await frame.locator(cmd["selector"]).screenshot(path=str(path))
                        else:
                            await page.screenshot(path=str(path))
                        out = {"path": str(path)}
                    elif action == "network":
                        out = captures[cmd.get("after", 0):][-cmd.get("limit", 20):]
                    elif action == "images":
                        out = [{"url": safe_url(k), "path": v} for k, v in images.items()]
                    elif action == "close":
                        break
                    else:
                        out = {"error": "未知命令"}
                    print(json.dumps(out, ensure_ascii=False), flush=True)
                except Exception as exc:
                    print(json.dumps({"error": str(exc)[:500]}, ensure_ascii=False), flush=True)
        finally:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await context.close()


asyncio.run(main())
