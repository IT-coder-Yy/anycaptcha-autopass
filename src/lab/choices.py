"""静态照片和印刷字卡共用的四选项题库与整题判分。"""
import csv
from contextlib import nullcontext
from functools import cache
from io import BytesIO
from pathlib import Path

CHOICE_KINDS = ("photo-vqa", "color-letter")
SHEET_SIZE = (480, 320)


@cache
def bank(kind):
    directory = Path(__file__).resolve().parents[1] / kind.replace("-", "_") / "data"
    cases = {}
    with (directory / "labels.tsv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            cases.setdefault(row["case"], []).append(row)
    return cases


def images(kind, rows):
    if kind == "photo-vqa":
        from photo_vqa.solver import photo
        return [photo(row) for row in rows]
    from color_letter.cards import card_png
    return [card_png(row) for row in rows]


def target(kind, row):
    return "yes" if kind == "photo-vqa" else row["question"]


def generate(kind, rng):
    rows = rng.choice(tuple(bank(kind).values()))
    first = rows[0]
    public = {"kind": kind, "instruction": f"选出符合“{first['title']}”的图片，填写选项字母（如 BD）。",
              "width": SHEET_SIZE[0], "height": SHEET_SIZE[1],
              "background": f"/choice-sheet?kind={kind}&case={first['case']}"}
    return public, {"text": "".join(row["option"] for row in rows if row["label"] == target(kind, first))}


def sheet(kind, rows):
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", SHEET_SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    for index, (row, data) in enumerate(zip(rows, images(kind, rows), strict=True)):
        x, y = (index % 2) * 240, (index // 2) * 160
        with Image.open(BytesIO(data)) as picture:
            picture.thumbnail((206, 150), Image.Resampling.LANCZOS)
            canvas.paste(picture, (x + 28 + (206-picture.width)//2, y + (160-picture.height)//2))
        draw.text((x + 5, y + 5), row["option"], fill="black", font_size=18)
    out = BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()


def evaluate(kind, case_id=None):
    cases = bank(kind)
    if case_id is not None:
        if case_id not in cases:
            raise LookupError(f"{kind} 题库中没有题号 {case_id}")
        cases = {case_id: cases[case_id]}
    if kind == "photo-vqa":
        from photo_vqa.solver import reader
        engine = reader()
    else:
        from color_letter.solver import read
        engine = nullcontext(("color-letter-template", read))

    results = []
    with engine as (model, predict):
        for name, rows in cases.items():
            wanted = target(kind, rows[0])
            items, expected, selected, unknown = {}, [], [], []
            for row, image in zip(rows, images(kind, rows), strict=True):
                key = row["option"]
                raw, value = predict(image, row["question"])
                if row["label"] == wanted:
                    expected.append(key)
                if value is None:
                    unknown.append(key)
                elif value == wanted:
                    selected.append(key)
                items[key] = {"raw": raw, "value": value, "label": row["label"]}
            results.append({"id": name, "title": rows[0]["title"], "expected": expected, "selected": selected,
                            "unanswered": unknown, "correct": selected == expected and not unknown, "items": items})
    return {"kind": kind, "engine": model, "total": len(results), "correct": sum(r["correct"] for r in results),
            "options_total": sum(len(r["items"]) for r in results),
            "options_correct": sum(k not in r["unanswered"] and (k in r["expected"]) == (k in r["selected"])
                                   for r in results for k in r["items"]),
            "labels_correct": sum(v["value"] == v["label"] for r in results for v in r["items"].values()),
            "unanswered_options": sum(len(r["unanswered"]) for r in results), "samples": results}
