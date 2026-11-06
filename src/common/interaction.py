"""有真实时间间隔的页面鼠标动作；用于可复现的交互测试，不代表行为风控通过保证。"""
import asyncio
import math
import os
import random
import time


POINTER_SCRIPT = """(() => {
  if (window.__captchaTestPointer) return;
  window.__captchaTestPointer = {point: null};
  // Chromium 的 mousemove 坐标取整；pointermove 保留小数，避免分段衔接时倒退。
  window.addEventListener('pointermove', e => {
    window.__captchaTestPointer.point = [e.clientX, e.clientY];
  }, true);
})()"""


def timed_path(start, end, duration, rng, bounds=None, straight=False, min_bend=0):
    """三次贝塞尔曲线配合平滑加减速；时间偏移以秒计，终点保持精确。"""
    x0, y0 = start
    dx, dy = end[0]-x0, end[1]-y0
    distance = math.hypot(dx, dy)
    bend = 0 if straight else rng.uniform(-1, 1)*min(12, distance*.08)
    if not straight and min_bend:
        bend = math.copysign(max(abs(bend), min_bend), bend)
    nx, ny = (-dy/distance, dx/distance) if distance else (0, 0)
    controls = [(x0+dx*t+nx*bend, y0+dy*t+ny*bend) for t in (.3, .7)]
    if bounds:
        left, top, right, bottom = bounds
        controls = [(min(right, max(left, x)), min(bottom, max(top, y))) for x, y in controls]
    count = max(2, math.ceil(duration*60))
    points = []
    for i in range(1, count+1):
        t = i/count
        s = 10*t**3-15*t**4+6*t**5
        weights = ((1-s)**3, 3*(1-s)**2*s, 3*(1-s)*s*s, s**3)
        anchors = (start, *controls, end)
        x, y = [sum(w*p[axis] for w, p in zip(weights, anchors)) for axis in (0, 1)]
        points.append((x, y, t*duration))
    points[-1] = (*end, duration)
    return points


class Interaction:
    def __init__(self, page, scale=None, seed=None, track_pointer=True, straight_drag=True):
        self.page = page
        self.track_pointer = track_pointer
        self.straight_drag = straight_drag
        self.scale = float(os.environ.get('CAPTCHA_ACTION_SCALE', '1')) if scale is None else scale
        if not math.isfinite(self.scale) or not .25 <= self.scale <= 4:
            raise ValueError('CAPTCHA_ACTION_SCALE 必须在 0.25–4 之间')
        seed = os.environ.get('CAPTCHA_ACTION_SEED') if seed is None else seed
        self.seed = int(seed) if seed is not None else random.SystemRandom().getrandbits(63)
        self.rng = random.Random(self.seed)
        self.position = (0, 0)
        self.actions = []

    async def install(self):
        if self.track_pointer:
            await self.page.add_init_script(POINTER_SCRIPT)
            await self.page.evaluate(POINTER_SCRIPT)

    @property
    def config(self):
        return {'profile': 'paced-v3', 'scale': self.scale, 'seed': self.seed}

    async def pause(self, stage, low, high):
        duration = self.rng.uniform(low, high)*self.scale
        started = time.monotonic()
        await asyncio.sleep(duration)
        self.actions.append({'action': stage, 'planned_ms': round(duration*1000, 1),
                             'elapsed_ms': round((time.monotonic()-started)*1000, 1)})

    async def move(self, x, y, *, drag=False, box=None):
        # 读取真实鼠标事件的位置，也覆盖页面准备阶段 locator.hover 的移动。
        # iframe 中的事件不会冒泡到主页面；此时沿用已发送的全局坐标。
        current = await self.page.evaluate('()=>window.__captchaTestPointer?.point ?? null') if self.track_pointer else None
        start = tuple(current) if current else self.position
        end = (x, y)
        distance = math.dist(start, end)
        if distance < .1:
            return
        bounds = None
        if box:
            bounds = (box['x'], box['y'], box['x']+box['width'], box['y']+box['height'])
            if not all(bounds[0] <= p[0] <= bounds[2] and bounds[1] <= p[1] <= bounds[3] for p in (start, end)):
                bounds = None
        duration = ((.35 if drag else .22) + .12*math.log2(distance/25+1))
        duration *= self.rng.uniform(.9, 1.1)*self.scale
        route = timed_path(start, end, duration, self.rng, bounds,
                           straight=(drag and self.straight_drag) or (box is not None and bounds is None),
                           min_bend=2 if drag and not self.straight_drag else 0)
        started = time.monotonic()
        actual = []
        for px, py, offset in route:
            # 按实际时钟推进；CDP 卡顿后跳过过期帧，不能无间隔补发旧轨迹。
            if offset < duration and time.monotonic() > started + offset:
                continue
            await asyncio.sleep(max(0, started+offset-time.monotonic()))
            await self.page.mouse.move(px, py)
            self.position = (px, py)
            actual.append([round(px, 2), round(py, 2), round((time.monotonic()-started)*1000, 1)])
        self.actions.append({'action': 'drag_move' if drag else 'move', 'planned_ms': round(duration*1000, 1),
                             'elapsed_ms': actual[-1][2], 'points': actual})

    async def click(self):
        await self.page.mouse.down()
        try:
            await self.pause('click_hold', .08, .16)
        finally:
            await self.page.mouse.up()


def can_retry(entry, allow_unanswered=False):
    """有限恢复：点字可明确启用未解出后的换题；旧回执和未知结果不得触发。"""
    if allow_unanswered and entry['outcome'] == 'low_confidence':
        return True
    return entry['outcome'] == 'failed' and any(
        r.get('attempt_id') == entry['attempt_id'] and r.get('accepted') is False
        for r in entry.get('receipts', []))
