"""两种输入题共用的固定印刷样本。标注只用于本地评测和出题。"""
import base64
import csv
from dataclasses import dataclass
from functools import cache
from io import BytesIO
from pathlib import Path

from PIL import Image


INPUT_KINDS = ("normal", "arithmetic")


@dataclass(frozen=True)
class Sample:
    image: Path
    text: str
    answer: str
    case_sensitive: bool | None = None

    def ignores_case(self, kind: str) -> bool:
        return kind == "normal" and self.case_sensitive is not True


def answer_matches(submitted: object, expected: str, ignore_case: bool) -> bool:
    if ignore_case:
        return isinstance(submitted, str) and submitted.isascii() and submitted.lower() == expected.lower()
    return submitted == expected


@cache
def samples(kind: str) -> tuple[Sample, ...]:
    directory = Path(__file__).resolve().parents[1] / kind / "data"
    with (directory / "labels.tsv").open(encoding="utf-8", newline="") as fp:
        flags = {"": None, "1": True, "0": False}
        return tuple(
            Sample(directory / row["image"], row["text"], row["answer"], flags[row.get("case_sensitive", "")])
            for row in csv.DictReader(fp, delimiter="\t")
        )


def generate_printed(kind, instruction, rng):
    sample = rng.choice(samples(kind))
    if kind == "normal" and sample.case_sensitive is not None:
        instruction += "区分大小写。" if sample.case_sensitive else "不区分大小写。"
    png = sample.image.read_bytes()
    with Image.open(BytesIO(png)) as image:
        width, height = image.size
    public = {"kind": kind, "instruction": instruction, "width": width, "height": height,
              "background": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}
    return public, {"text": sample.answer, "ignore_case": sample.ignores_case(kind)}
