"""易盾演示页的展示模式、稳定布局与被动 UI 记录。"""
import asyncio
import time
from playwright.async_api import TimeoutError as PlaywrightTimeout


MODES = {'trigger': '触发式', 'embedded': '嵌入式', 'popup': '弹出式'}

STATE_JS = r"""root => {
  const rect=e=>{if(!e)return null;const r=e.getBoundingClientRect();
    return {x:r.x,y:r.y,width:r.width,height:r.height}};
  const visible=e=>{if(!e||!e.isConnected)return false;
    const r=e.getBoundingClientRect();if(!r.width||!r.height)return false;
    for(let a=e;a;a=a.parentElement){const s=getComputedStyle(a);
      if(s.display==='none'||s.visibility==='hidden'||Number(s.opacity)<.99)return false;}return true;};
  const q=s=>root?.querySelector(s), bg=q('.yidun_bg-img'), piece=q('.yidun_jigsaw');
  const handle=q('.yidun_slider'), control=q('.yidun_control'), panel=q('.yidun_panel');
  const h=rect(handle), hit=h?document.elementFromPoint(h.x+h.width/2,h.y+h.height/2):null;
  return {t:performance.now(),classes:root?.className||'',connected:!!root?.isConnected,
    background:rect(bg),piece:rect(piece),handle:h,control:rect(control),panel:rect(panel),
    image_visible:visible(bg),piece_visible:visible(piece),handle_visible:visible(handle),
    decoded:!!(bg?.complete&&bg.naturalWidth&&(!piece||piece.complete&&piece.naturalWidth)),
    hit_handle:!!(hit&&handle?.contains(hit)),panel_style:panel?.getAttribute('style'),
    pointer:window.__captchaTestPointer?.point||null};
}"""


async def state(page):
    return await page.evaluate(f'() => ({STATE_JS})(document.querySelector(".yidun"))')


def ready(snapshot, kind='slider'):
    classes=set(snapshot['classes'].split())
    blocked={'yidun--loading','yidun--error','yidun--verifying','yidun--success'}
    return (snapshot['decoded'] and snapshot['image_visible'] and not classes & blocked
            and (kind != 'slider' or snapshot['piece_visible'] and snapshot['handle_visible'] and snapshot['hit_handle']))


def same_geometry(before, after, tolerance=.5):
    return all(before.get(k) and after.get(k) and
               all(abs(before[k][axis]-after[k][axis])<=tolerance for axis in ('x','y','width','height'))
               for k in ('background','piece','handle','control','panel'))


async def show(page, kind):
    control=page.locator('.yidun_control')
    await control.wait_for(state='visible')
    box=await control.bounding_box()
    # 只在空间不足时滚动；不要在每次读取坐标时重复触发页面滚动。
    if box and (box['y']<230 or box['y']+box['height']>page.viewport_size['height']-40):
        await control.evaluate("e=>e.scrollIntoView({block:'center',inline:'nearest',behavior:'instant'})")
    target=page.locator('.yidun_slider') if kind=='slider' else control
    await target.wait_for(state='visible')
    deadline=time.monotonic()+12
    previous=None
    stable=0
    hovered=False
    last_hover=0.
    while time.monotonic()<deadline:
        current=await state(page)
        if not hovered:
            # 直接在真正的把手上展开浮层，避免展开后再从验证条中间横穿。
            await target.hover()
            hovered=True
            last_hover=time.monotonic()
        if ready(current,kind):
            settled=previous is not None and (same_geometry(previous,current) if kind=='slider'
                else current['background']==previous['background'])
            stable=stable+1 if settled else 0
            if stable>=2:return current
            previous=current
        else:
            previous=None;stable=0
            if not current['image_visible'] and time.monotonic()-last_hover>=1.0:
                # 重新跨过边界，处理初始化或失败动画期间未注册/未响应的 mouseenter。
                b=await target.bounding_box()
                if b:
                    await page.mouse.move(max(2,b['x']-20),b['y']+b['height']/2)
                    await target.hover()
                    last_hover=time.monotonic()
        await asyncio.sleep(.15)
    raise RuntimeError('易盾题面未进入可操作的稳定状态')


async def select_mode(page, kind, mode):
    # 等默认实例初始化完成再切换，避免其异步回调重新插入第二个验证码实例。
    await show(page,kind)
    popup_clicks=0
    if mode!='trigger':
        await page.get_by_text(MODES[mode],exact=True).click()
        if mode=='popup':
            # 演示按钮可见早于弹窗监听器就绪；只在尚无题面时有限重试唤起。
            for _ in range(3):
                await page.locator('.tcapt-bind_btn--login.j-pop').click()
                popup_clicks+=1
                try:
                    await page.locator('.yidun--popup').wait_for(state='attached',timeout=2500)
                    break
                except PlaywrightTimeout:
                    if await page.locator('.yidun').count():raise
        expected='yidun--embed' if mode=='embedded' else 'yidun--popup'
        await page.wait_for_function("cls=>{const es=document.querySelectorAll('.yidun');return es.length===1&&es[0].classList.contains(cls)}",arg=expected)
    await show(page,kind)
    return {'mode':mode,'popup_open_clicks':popup_clicks}


async def start_trace(page):
    await page.evaluate('() => { const read=('+STATE_JS+');'+r"""
      window.__yidunObservation?.stop();
      const root=document.querySelector('.yidun'), source=()=>[...root.querySelectorAll('.yidun_bg-img,.yidun_jigsaw')].map(e=>e.src).join('|');
      const initial=source(), events=[];
      let terminal=null;
      const snapshot=()=>({...read(root),same_image:source()===initial});
      const record=(event,e)=>{
        const s=snapshot();if(events.length<500)events.push({...s,event,buttons:e?.buttons,x:e?.clientX,y:e?.clientY,hit:e?.target?.className});
        if(s.same_image&&s.classes.split(' ').includes('yidun--success'))terminal=s;
      };
      const onDown=e=>record('mousedown',e),onUp=e=>record('mouseup',e);
      root.addEventListener('mousedown',onDown,true);window.addEventListener('mouseup',onUp,true);
      const observer=new MutationObserver(ms=>{if(ms.some(m=>m.target===root||m.target.classList?.contains('yidun_panel')))record('state');});
      observer.observe(root,{attributes:true,subtree:true,attributeFilter:['class','style']});
      window.__yidunObservation={snapshot,events,get terminal(){return terminal},record,
        stop(){observer.disconnect();root.removeEventListener('mousedown',onDown,true);window.removeEventListener('mouseup',onUp,true)}};
      record('start');
    }""")


async def trace(page, stop=False):
    return await page.evaluate("""stop=>{const o=window.__yidunObservation;if(!o)return null;
      const result={current:o.snapshot(),terminal:o.terminal,events:o.events};if(stop)o.stop();return result;}""",stop)
