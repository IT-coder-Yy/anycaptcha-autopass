"""解释单行双操作数算式。"""
import re


def answer_from_text(text: str) -> tuple[str | None, str]:
    match = re.fullmatch(
        r"([0-9]+)[ \t]*([+*xX×/÷-])[ \t]*([0-9]+)(?:[ \t]*=[ \t]*[?？]?)?",
        text.strip(" \t"),
    )
    if not match:
        return None, "需要两个非负整数和一个运算符"
    left, op, right = match.groups()
    a, b = int(left), int(right)
    if op == "+":
        value = a + b
    elif op == "-":
        value = a - b
    elif op in "*xX×":
        value = a * b
    elif b == 0:
        return None, "除数为零"
    elif a % b:
        return None, "除法结果不是整数"
    else:
        value = a // b
    return str(value), ""
