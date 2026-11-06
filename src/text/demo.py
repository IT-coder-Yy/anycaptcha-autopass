"""凯格与易盾的顺序点字题面读取和点击操作。"""
import math
import re
import cv2
import numpy as np
from common.demo import write_json, show_yidun, source_id


def normalize_prompt(text):
    return ''.join(re.findall(r'[\u4e00-\u9fff]',text))

async def text_input(page, site, folder):
    picture=page.locator("#KgDisplay img[src*='strbasemap']" if site=='kg' else '.yidun_bg-img')
    await picture.evaluate("""async e => {
        await e.decode();
        if(!e.complete || e.naturalWidth<16 || e.naturalHeight<16)throw new Error('题图未完整解码');
    }""")
    bg = page.locator("#KgDisplay" if site == "kg" else ".yidun_bg-img")
    hint = page.locator("#KgControl b" if site == "kg" else ".yidun_tips__point")
    b, h = await bg.bounding_box(), await hint.bounding_box()
    if not b or not h:
        raise ValueError("背景或提示不可见")
    x, y = math.floor(min(b["x"], h["x"])), math.floor(min(b["y"], h["y"]))
    right = math.ceil(max(b["x"] + b["width"], h["x"] + h["width"]))
    bottom = math.ceil(max(b["y"] + b["height"], h["y"] + h["height"]))
    clip = {"x": x, "y": y, "width": right - x, "height": bottom - y}
    chars = await hint.evaluate(r"""e => {
        const walker=document.createTreeWalker(e,NodeFilter.SHOW_TEXT), out=[];
        while(walker.nextNode()) { const n=walker.currentNode;
          for(let i=0;i<n.textContent.length;i++) {
            if(!/\p{Script=Han}/u.test(n.textContent[i])) continue;
            const r=document.createRange();r.setStart(n,i);r.setEnd(n,i+1);
            const b=r.getBoundingClientRect();out.push([b.x,b.y,b.right,b.bottom]);
          }
        } return out;
    }""")
    hints = [[a-x,c-y,d-x,e-y] for a,c,d,e in chars]
    data = await page.screenshot(clip=clip, path=str(folder/"challenge.png"))
    hint_text="".join(re.findall(r"[\u4e00-\u9fff]",await hint.inner_text()))
    write_json(folder/'input.json',{'hint_text':hint_text,'hint_boxes':hints,'clip':clip,'background_box':b})
    return cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR), hints, clip, b, hint_text

async def click_image_point(page,site,x,y,fingerprint,interaction):
    target=page.locator('#KgDisplay' if site=='kg' else '.yidun_bg-img')
    surface=target if site=='kg' else target.locator('..')
    for _ in range(3):
        if site=='yidun':await show_yidun(page,'text')
        if await source_id(page,site,'text')!=fingerprint:raise RuntimeError('点选期间题面变化')
        box=await target.bounding_box()
        if not box or not (0<=x<box['width'] and 0<=y<box['height']):raise ValueError('点选坐标不在图内')
        px,py=box['x']+x,box['y']+y
        await interaction.move(px,py,box=box)
        await interaction.pause('before_click',.10,.22)
        hit=await surface.evaluate("""(e,p)=>{const h=document.elementFromPoint(p.x,p.y);
          return !!h&&e.contains(h)&&!(h.tagName==='IMG'&&/refresh\\.png/.test(h.getAttribute('src')||''));}""",{'x':px,'y':py})
        if hit and await target.is_visible():
            if await source_id(page,site,'text')!=fingerprint:raise RuntimeError('点选前题面变化')
            await interaction.click()
            return
    raise RuntimeError('验证码图像未保持可点击，已避免点击背后表单')
