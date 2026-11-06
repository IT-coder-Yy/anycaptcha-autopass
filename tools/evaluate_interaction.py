"""仅用本地 HTML 核验实际鼠标事件耗时、点击命中和拖动终点。"""
import asyncio
import json
import os
from pathlib import Path
import time

from playwright.async_api import async_playwright
from common.interaction import Interaction


async def main():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(root/'.browsers'))
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 800, 'height': 500})
        interaction = Interaction(page, scale=1, seed=42)
        await interaction.install()
        await page.set_content('''<button id="target" style="position:absolute;left:300px;top:200px;width:120px;height:80px">点击测试</button>
        <script>window.events=[];for(const kind of ['mousemove','mousedown','mouseup','click'])
        addEventListener(kind,e=>events.push({kind,t:performance.now(),x:e.clientX,y:e.clientY,target:e.target.id,buttons:e.buttons}));</script>''')
        await interaction.move(350, 240)
        await interaction.pause('before_click', .10, .22)
        await interaction.click()
        await interaction.pause('between_actions', .25, .55)
        await page.mouse.down()
        try:
            await interaction.move(600, 240, drag=True)
        finally:
            await page.mouse.up()
        events = await page.evaluate('events')
        down = next(i for i, e in enumerate(events) if e['kind'] == 'mousedown')
        up = next(i for i, e in enumerate(events) if e['kind'] == 'mouseup')
        moves = [e for e in events[:down] if e['kind'] == 'mousemove']
        checks = {'movement_ms': round(moves[-1]['t']-moves[0]['t'], 1),
                  'hold_ms': round(events[up]['t']-events[down]['t'], 1),
                  'clicked_target': any(e['kind'] == 'click' and e['target'] == 'target' for e in events),
                  'drag_ended_at_target': events[-1]['x'] == 600 and events[-1]['y'] == 240}
        result = {'config': interaction.config, 'checks': checks, 'actions': interaction.actions, 'events': events}
        folder = root/'runtime'/f"interaction-{time.strftime('%Y%m%d-%H%M%S')}"
        folder.mkdir(parents=True)
        (folder/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        await browser.close()
        print(json.dumps({'directory': str(folder), **checks}, ensure_ascii=False))
        assert checks['movement_ms'] >= 400 and checks['hold_ms'] >= 75
        assert checks['clicked_target'] and checks['drag_ended_at_target']


asyncio.run(main())
