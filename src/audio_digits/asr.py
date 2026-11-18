"""用配置的语音转写服务听写本地录音，保留原文供核对。"""
import mimetypes
import os
from pathlib import Path
import re
import unicodedata

from dotenv import dotenv_values
import httpx


DIGITS = dict(zip("zero one two three four five six seven eight nine".split(), "0123456789"))
DIGITS.update(zip("零一二三四五六七八九", "0123456789"))
DIGITS.update({"oh": "0", "〇": "0", "幺": "1", "两": "2"})


def normalize_digits(text: str) -> str | None:
    text = re.sub(r"<\|[^|]*\|>", "", unicodedata.normalize("NFKC", text)).lower()
    out = []
    for token in re.findall(r"[a-z]+|[0-9]+|[^\s]", text):
        if token in DIGITS:
            out.append(DIGITS[token])
        elif token.isascii() and token.isdigit():
            out.append(token)
        elif not (len(token) == 1 and unicodedata.category(token).startswith("P")):
            return None
    return "".join(out) or None


def transcribe(path: Path, env_file: Path) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or ""
    if not mime.startswith("audio/"):
        raise ValueError("请选择本地音频文件，例如 WAV 或 MP3")
    env = dotenv_values(env_file, interpolate=False)
    settings = {}
    for name in ("ASR_BASE_URL", "ASR_MODEL", "ASR_API_KEY"):
        settings[name] = (os.getenv(name, env.get(name)) or "").strip()
        if not settings[name]:
            raise ValueError(f"请配置 {name}")
    try:
        url = httpx.URL(settings["ASR_BASE_URL"].rstrip("/") + "/audio/transcriptions")
    except httpx.InvalidURL:
        raise ValueError("ASR_BASE_URL 不是有效的地址") from None
    if url.scheme != "https" or not url.host or url.userinfo or url.query or url.fragment:
        raise ValueError("ASR_BASE_URL 需要不含凭据或查询参数的 HTTPS 基础地址")

    model, key = settings["ASR_MODEL"], settings["ASR_API_KEY"]
    with path.open("rb") as audio, httpx.Client(timeout=60, trust_env=False) as client:
        try:
            response = client.post(url, headers={"Authorization": "Bearer " + key},
                                   data={"model": model}, files={"file": ("audio" + path.suffix.lower(), audio, mime)})
        except httpx.RequestError as exc:
            raise ValueError(f"ASR 连接失败（{type(exc).__name__}）") from None
    if not response.is_success:
        raise ValueError(f"ASR 请求失败，HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError:
        raise ValueError("ASR 响应不是有效的 JSON") from None
    text = body.get("text") if isinstance(body, dict) else None
    if not isinstance(text, str):
        raise ValueError("ASR 响应缺少有效的 text 字段")
    text = text.replace(key, "[已脱敏]")
    return {"model": model, "text": text, "digits": normalize_digits(text)}
