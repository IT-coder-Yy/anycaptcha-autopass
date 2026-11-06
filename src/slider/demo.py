"""凯格与易盾的滑块题面读取和位移反馈。"""
import asyncio
import re
import cv2
import numpy as np
from common import yidun
from common.demo import image_bytes, write_json, show_yidun


async def slider_input(page, site, folder):
    if site == "kg":
        bg, pc = page.locator("#KgBasemap"), page.locator("#KgBasemap > img[src^='data:']")
        css = await bg.evaluate("e => getComputedStyle(e).backgroundImage")
        match = re.fullmatch(r'url\([\"\']?(.*?)[\"\']?\)', css)
        if not match:
            raise ValueError("底图格式已变化")
        bsrc = match.group(1)
        # 实测拖块 cursor=move；避免依赖匿名 div 的序号。
        handle_index = await page.locator("#KgSlide > div").evaluate_all("es=>es.findIndex(e=>getComputedStyle(e).cursor==='move')")
        if handle_index < 0:
            raise ValueError("未找到拖动把手")
        handle = page.locator("#KgSlide > div").nth(handle_index)
    else:
        bg, pc, handle = page.locator(".yidun_bg-img"), page.locator(".yidun_jigsaw"), page.locator(".yidun_slider")
        bsrc = await bg.get_attribute("src")
    psrc = await pc.get_attribute("src")
    bgdata, pcdata = await asyncio.gather(image_bytes(page,bsrc),image_bytes(page,psrc))
    (folder/"background.png").write_bytes(bgdata)
    (folder/"piece.png").write_bytes(pcdata)
    bi = cv2.imdecode(np.frombuffer(bgdata,np.uint8),cv2.IMREAD_COLOR)
    pi = cv2.imdecode(np.frombuffer(pcdata,np.uint8),cv2.IMREAD_UNCHANGED)
    if bi is None or pi is None:
        raise ValueError("题图响应无法解码为图片")
    # CDN 下载期间浮动面板可能收起；获取坐标前重新展示，随后再校验题面身份。
    if site=="yidun": await show_yidun(page,"slider")
    await bg.wait_for(state="visible")
    await pc.wait_for(state="visible")
    b, p = await bg.bounding_box(), await pc.bounding_box()
    if not b or not p:
        raise ValueError("图片已解码，但背景或碎片布局不可见")
    scale = b["width"]/bi.shape[1]
    resized = cv2.resize(pi,(round(p["width"]/scale),round(p["height"]/scale)))
    write_json(folder/'input.json', {'background_box':b, 'piece_box':p, 'scale':scale,
        'background_size':list(bi.shape[:2]), 'piece_size':list(pi.shape[:2]),
        'resized_piece_size':list(resized.shape[:2]),
        'current_origin':[(p['x']-b['x'])/scale,(p['y']-b['y'])/scale]})
    return bi,resized,b,p,handle,pc,scale

async def drag_feedback(page, handle, piece, target_x, interaction, *, diagnostic=None, strategy='segmented'):
    """按页面实际碎片位移反馈调整拖动，避免固定死区或比例。"""
    h, p = await handle.bounding_box(), await piece.bounding_box()
    if not h or not p:
        raise ValueError("拖动元素不可见")
    x, y = h["x"]+h["width"]/2, h["y"]+h["height"]/2
    if target_x-p["x"] <= 1:
        raise ValueError("候选位移不为正")
    await interaction.move(x,y)
    await interaction.pause('before_drag',.12,.25)
    hit=await handle.evaluate('(e,p)=>e.contains(document.elementFromPoint(p.x,p.y))',{'x':x,'y':y})
    if not hit: raise ValueError('按下前鼠标未命中拖动把手')
    history=[]
    if diagnostic is not None:
        diagnostic.update({'strategy':strategy,'target_x':target_x,'steps':history,'before_down':await yidun.state(page)})
    await page.mouse.down()
    try:
        await interaction.pause('drag_hold',.12,.22)
        for i in range(6):
            previous=p["x"]
            error=target_x-previous
            if abs(error)<=1:
                break
            step=min(30,error*.6) if i==0 else (error if strategy=='calibrated' else max(-80,min(80,error)))
            x+=step
            if not 2 < x < page.viewport_size["width"]-2:
                raise ValueError("拖动超出视口")
            await interaction.move(x,y,drag=True)
            await page.wait_for_timeout(120)
            p=await piece.bounding_box()
            if not p:
                raise ValueError("拖动期间题面消失")
            history.append({"mouse_delta":step,"piece_delta":p["x"]-previous,"error":target_x-p["x"]})
        await interaction.pause('before_release',.14,.28)
        if diagnostic is not None:
            diagnostic['before_release']=await yidun.state(page)
            diagnostic['final_error']=target_x-p['x']
        if abs(target_x-p['x'])>1: raise ValueError('拖动结束仍未达到预测位置')
        return history
    finally:
        await page.mouse.up()
        if diagnostic is not None:
            diagnostic['after_release']=await yidun.state(page)
