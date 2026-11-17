"""用带标注的本地题图测量实际 OCR 原文及答案。"""
import hashlib
from importlib.metadata import version
from itertools import groupby
import os
from pathlib import Path
from time import perf_counter

import cv2
import ddddocr

from arithmetic.solver import answer_from_text as arithmetic_answer
from common.text_ocr import prepare
from common.vision import Vision
from lab.printed import answer_matches, samples
from normal.solver import answer_from_text as normal_answer


def _load_engine(name):
    if name == "ddddocr":
        model = Path(ddddocr.__file__).with_name("common_old.onnx")
        engine_version = version("ddddocr")
        ocr = ddddocr.DdddOcr(show_ad=False, use_gpu=False)

        def read(image):
            return ocr.classification(image.read_bytes())
    else:
        root = Path(os.environ.get("CAPTCHA_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        model = root / "models" / f"{name}.onnx"
        if not model.is_file():
            raise FileNotFoundError(f"缺少 {model}，请运行 tools/fetch_models.py")
        session = Vision(model.parent).session(model.name)
        input_name = session.get_inputs()[0].name
        characters = [""] + session.get_modelmeta().custom_metadata_map["character"].splitlines() + [" "]
        engine_version = "ONNX Runtime " + version("onnxruntime")

        def read(image):
            pixels = cv2.imread(str(image))
            if pixels is None:
                raise OSError(f"无法读取图片 {image}")
            scores = session.run(None, {input_name: prepare(pixels)[None]})[0][0]
            # CTC 先合并连续重复项，再去掉空白项，保留由空白分隔的重复字符。
            return "".join(characters[i] for i, _ in groupby(scores.argmax(axis=1)) if i)

    return read, {"version": engine_version, "model": model.name,
                  "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}


def evaluate_samples(kind, image=None):
    cases = samples(kind)
    if image is not None:
        path = image.resolve()
        cases = [case for case in cases if case.image == path]
        if not cases:
            raise LookupError(f"请选择 {kind}/data 中带标注的题图，或省略图片以评测整组")

    interpret = normal_answer if kind == "normal" else arithmetic_answer
    readers, models = {}, {}
    rows = []
    for case in cases:
        ignore_case = case.ignores_case(kind)
        engine = "ddddocr" if ignore_case else "ppocr-v5-mobile"
        if engine not in readers:
            readers[engine], models[engine] = _load_engine(engine)
        raw = answer = None
        start = perf_counter()
        try:
            raw = readers[engine](case.image)
        except Exception as exc:
            outcome, reason = "error", str(exc)
        else:
            answer, reason = interpret(raw)
            if answer is None:
                outcome = "abstained"
            else:
                outcome = "correct" if answer_matches(answer, case.answer, ignore_case) else "wrong"
        rows.append({"image": case.image.name, "engine": engine, "case_sensitive": case.case_sensitive,
                     "expected_text": case.text, "raw_text": raw,
                     "text_exact": raw == case.text, "expected_answer": case.answer,
                     "answer": answer, "outcome": outcome, "reason": reason,
                     "elapsed_ms": round((perf_counter() - start) * 1000, 2)})

    report = {"kind": kind, "source": "labeled_local_images", "engines": models,
              "total": len(rows), "text_exact": sum(row["text_exact"] for row in rows),
              **{key: sum(row["outcome"] == key for row in rows)
                 for key in ("correct", "wrong", "abstained", "error")}, "samples": rows}
    if len(models) == 1:
        report.update(next(iter(models.values())))
    if "ddddocr" in models:
        report["ddddocr"] = models["ddddocr"]["version"]
    return report
