"""对题库内的 COCO 原图逐张回答原始 VQA 是非题。"""
import base64
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import httpx

from semantic.solver import model_connection

PROMPT = "Answer the following question about this photograph. Reply with Yes or No only. "


def photo(row):
    root = Path(os.environ.get("CAPTCHA_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    cache = root / ".cache" / "photo-vqa"
    path = cache / row["image"]
    if path.is_file():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() == row["sha256"]:
            return data
    url = "https://s3.amazonaws.com/images.cocodataset.org/val2014/COCO_val2014_" + row["image"]
    with urlopen(url, timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError(f"COCO 原图校验失败：{row['image']}")
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache) as temp:
        pending = Path(temp) / path.name
        pending.write_bytes(data)
        pending.replace(path)
    return data


@contextmanager
def reader():
    config = model_connection(default_model="qwen3.7-plus")
    key = config["api_key"]
    headers = {"Authorization": "Bearer " + key} if key else {}
    with httpx.Client(timeout=90, trust_env=False, follow_redirects=False, headers=headers) as client:
        def read(jpeg, question):
            content = [{"type": "text", "text": PROMPT + question},
                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}}]
            payload = {"model": config["model"], "messages": [{"role": "user", "content": content}],
                       "temperature": 0, "max_tokens": 512}
            try:
                response = client.post(config["base_url"].rstrip("/") + "/chat/completions", json=payload)
            except httpx.RequestError as exc:
                raise ValueError(f"视觉模型连接失败（{type(exc).__name__}）") from None
            if not response.is_success:
                raise ValueError(f"视觉模型请求失败，HTTP {response.status_code}")
            try:
                content = response.json()["choices"][0]["message"]["content"]
                raw = (content or "").strip()
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                raise ValueError("视觉模型响应缺少有效的文本回答") from None
            raw = raw.replace(key, "[已脱敏]") if key else raw
            match = re.match(r"^(yes|no)\b", raw, re.I)
            return raw, match[1].lower() if match else None

        yield config["model"], read
