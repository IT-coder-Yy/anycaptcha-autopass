"""抖音顺序点字适配：使用已有本地检测与字符图像匹配模型。"""

import asyncio
import hashlib
import math
import os
from pathlib import Path

from common.interaction import Interaction

BACKGROUND = "#vc_captcha_box #captcha_click_image"
HINT = "#vc_captcha_box .vc-captcha-verify-bar-img"
MARKERS = "#vc_captcha_box .vc-captcha-verify-img-point"
SUBMIT = "#vc_captcha_box .vc-captcha-verify-pb"


def recognize(background, hint, model_dir):
    import cv2
    import numpy as np

    from common.text_support import detect_characters
    from common.vision import Prediction, Vision

    vision = Vision(model_dir)
    # 页面提示字比候选小，放大参考图后检测；匹配器会把每个字符归一化。
    ratio = max(1, 64 / hint.shape[0])
    hint = cv2.resize(hint, (round(hint.shape[1] * ratio), round(hint.shape[0] * ratio)))
    # 检测器会归一化整张输入；给小提示图留白，避免字符被放大至几乎占满输入。
    padded = np.full((hint.shape[0] * 3, max(hint.shape[1], hint.shape[0] * 5), 3), 255, dtype=np.uint8)
    padded[:hint.shape[0], :hint.shape[1]] = hint
    hints = sorted(detect_characters(vision, padded), key=lambda box: box[0])
    if not 1 <= len(hints) <= 5:
        return Prediction("text", False, reason="无法完整检测提示字符")
    offset = hint.shape[0] + 16
    height, width = background.shape[:2]
    image = np.full((offset + height, max(width, hint.shape[1]), 3), 255, dtype=np.uint8)
    image[:hint.shape[0], :hint.shape[1]] = hint
    image[offset:, :width] = background
    prediction = vision.text(image, hint_boxes=hints, target_region=[0, offset, width, height],
                             detector="ddddocr", ocr_check=False)
    # 提示本身是图片，没有可靠的明文提示时使用原有 Siamese 匹配及歧义门槛。
    prediction.points = [[x, y - offset] for x, y in prediction.points]
    prediction.detail["hint_source"] = "douyin_prompt_image"
    prediction.detail["coordinate_space"] = "background_image_pixels"
    prediction.detail["reference_offset_y"] = offset
    return prediction


async def solve(page, frame, observation, record, *, stable_boxes):
    import cv2
    import numpy as np

    generation = observation.generation
    background, hint = frame.locator(BACKGROUND), frame.locator(HINT)
    for element in (background, hint):
        await element.wait_for(state="visible", timeout=10000)
        await element.evaluate("async e => { await e.decode(); if (!e.naturalWidth) throw new Error('题图为空'); }")
    await stable_boxes([background, hint])
    if await frame.locator(MARKERS).count():
        record["reason"] = "点字题已有选择，保留现场转人工"
        return "unavailable"
    sources = [await element.get_attribute("src") for element in (background, hint)]
    images, raw_images = [], []
    for element, name in ((background, "background"), (hint, "hint")):
        data = await element.screenshot(scale="css", **(
            {"path": record["image_prefix"] + f"-text-{name}.png"} if record.get("image_prefix") else {}))
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("点字题图解码失败")
        images.append(image)
        raw_images.append(data)
    model_dir = Path(os.getenv("DOUYIN_CAPTCHA_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "models"))).expanduser()
    if any(not (model_dir / name).is_file() for name in ("text-detector.onnx", "text-matcher.onnx")):
        record["reason"] = "缺少点字模型，请运行 captcha/tools/fetch_models.py 或配置 DOUYIN_CAPTCHA_MODEL_DIR"
        return "unavailable"

    async def unchanged():
        return (observation.generation == generation
                and sources == [await element.get_attribute("src") for element in (background, hint)])

    if not await unchanged():
        return "changed"
    prediction = await asyncio.to_thread(recognize, *images, model_dir)
    record["prediction"] = prediction.to_dict()
    record["image_hash"] = hashlib.sha256(b"".join(raw_images)).hexdigest()
    if not await unchanged():
        return "changed"
    if not prediction.usable:
        return "unanswered"
    height, width = images[0].shape[:2]
    points = prediction.points
    if (not 1 <= len(points) <= 5
            or any(len(point) != 2 or any(not math.isfinite(v) for v in point)
                   or not 0 < point[0] < width or not 0 < point[1] < height for point in points)
            or len({tuple(point) for point in points}) != len(points)):
        record["reason"] = "点字坐标无效，停止操作"
        return "unavailable"
    interaction = Interaction(page, track_pointer=False, straight_drag=False)
    await interaction.install()
    record["geometry"] = []
    try:
        for index, (px, py) in enumerate(points, 1):
            box, = await stable_boxes([background])
            x, y = box["x"] + px * box["width"] / width, box["y"] + py * box["height"] / height
            if not await unchanged():
                return "changed"
            await interaction.move(x, y)
            if not await unchanged():
                return "changed"
            await interaction.click()
            await interaction.pause("between_characters", .2, .35)
            record["geometry"].append({"order": index, "image_point": [px, py], "page_point": [x, y]})
            if not await unchanged():
                return "changed"
            if await frame.locator(MARKERS).count() != index:
                record["reason"] = "页面未确认点字顺序，停止提交"
                return "unknown"
        submit = frame.locator(SUBMIT)
        if await submit.evaluate("e => e.classList.contains('disable')"):
            return "unknown"
        if not await unchanged():
            return "changed"
        await submit.click(timeout=5000)
        return "submitted"
    finally:
        record["actions"] = interaction.actions
