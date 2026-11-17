"""校验离线 OCR 原文，保留大小写和前导零。"""


def answer_from_text(text: str) -> tuple[str | None, str]:
    if text.isascii() and text.isalnum():
        return text, ""
    return None, "需要一行非空的 ASCII 字母或数字"
