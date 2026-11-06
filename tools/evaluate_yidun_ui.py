"""本地 HTML 回归：嵌入图保留、弹窗移除、旧题终态均不误判。"""
import asyncio,json,os
from pathlib import Path
from types import SimpleNamespace
from playwright.async_api import async_playwright
import common.yidun as yidun
from common.demo import wait_result

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(ROOT/'.browsers'))
HTML='''<div class="yidun yidun--jigsaw yidun--embed">
  <div class="yidun_panel" style="position:relative;width:320px;height:160px">
    <img class="yidun_bg-img" style="width:320px;height:160px" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='320' height='160'%3E%3Crect width='320' height='160' fill='gray'/%3E%3C/svg%3E">
    <img class="yidun_jigsaw" style="position:absolute;left:0;top:0;width:60px;height:160px" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='160'%3E%3C/svg%3E">
  </div><div class="yidun_control" style="width:320px;height:40px">
    <div class="yidun_slider" style="width:40px;height:40px;background:blue"></div>
  </div></div>'''

async def main():
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True)
        page=await browser.new_page()
        outcomes={}
        for scenario in ['embedded_success','popup_removed','changed_image','server_reject']:
            await page.set_content(HTML)
            await page.locator('.yidun_bg-img').evaluate('e=>e.decode()')
            await yidun.start_trace(page)
            if scenario=='changed_image':
                await page.locator('.yidun_bg-img').evaluate("e=>e.src=e.src.replace('gray','green')")
            await page.locator('.yidun').evaluate("e=>e.classList.add('yidun--success')")
            await page.wait_for_timeout(25)
            if scenario=='popup_removed':await page.locator('.yidun').evaluate('e=>e.remove()')
            accepted=scenario!='server_reject'
            receipts=SimpleNamespace(results=[{'attempt_id':'current','accepted':accepted}])
            verdict,_,_=await wait_result(page,'yidun',receipts,'current','image',.25)
            outcomes[scenario]=verdict.value
            await yidun.trace(page,stop=True)
        await page.evaluate("""()=>{window.coordinates=[];for(const type of ['pointermove','mousemove'])
          window.addEventListener(type,e=>coordinates.push({type,x:e.clientX,y:e.clientY}));}""")
        await page.mouse.move(100.75,120.125)
        coordinates=await page.evaluate('coordinates')
        await page.set_content(HTML)
        await page.evaluate("""()=>{const p=document.querySelector('.yidun_panel'),c=document.querySelector('.yidun_control');
          p.style.opacity='0';p.style.transition='opacity .4s';
          c.addEventListener('mouseenter',()=>p.style.opacity='1');
          c.addEventListener('mouseleave',()=>p.style.opacity='0');}""")
        settled=await yidun.show(page,'slider')
        await browser.close()
    expected={'embedded_success':'passed','popup_removed':'passed','changed_image':'unknown','server_reject':'failed'}
    result={'outcomes':outcomes,'coordinates':coordinates,'animation_settled':yidun.ready(settled),
            'ok':outcomes==expected and yidun.ready(settled)}
    print(json.dumps(result,ensure_ascii=False))
    assert result['ok']

asyncio.run(main())
