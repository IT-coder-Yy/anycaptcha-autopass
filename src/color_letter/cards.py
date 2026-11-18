"""按题库记录的绘制参数生成印刷字卡和参考字形。"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont

REFERENCE_FONTS = ("DejaVuSans.ttf", "LiberationSans-Regular.ttf")


def glyph_png(letter, font_name, size, angle, ink, *, mask=False):
    root = Path(os.environ.get("CAPTCHA_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    directory = root / ".cache" / "color-letter"
    suffix = "-mask" if mask else ""
    name = f"{ord(letter)}-{font_name}-{size}-{angle}-{ink[1:]}{suffix}.png"
    path = directory / name
    if not path.is_file():
        try:
            font = ImageFont.truetype(font_name, size * 3)
        except OSError as exc:
            raise OSError(f"生成字卡需要字体 {font_name}，字体版本见 README。") from exc
        background = (246, 245, 240)
        image = Image.new("RGB", (144 * 3, 112 * 3), background)
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = draw.textbbox((0, 0), letter, font=font)
        xy = ((image.width - left - right) / 2, (image.height - top - bottom) / 2)
        draw.text(xy, letter, font=font, fill=ink)
        image = image.rotate(angle, Image.Resampling.BICUBIC, fillcolor=background)
        image = image.resize((144, 112), Image.Resampling.LANCZOS)
        if mask:
            image = image.convert("L").point(lambda gray: 255 if gray < 165 else 0, mode="1")
        directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=directory) as temporary:
            pending = Path(temporary) / name
            image.save(pending)
            pending.replace(path)
    return path.read_bytes()


def card_png(row):
    return glyph_png(row["label"][1:], row["font"], int(row["size"]), float(row["angle"]), row["ink"])
