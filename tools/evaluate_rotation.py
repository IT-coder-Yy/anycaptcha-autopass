"""用已采集演示站图片构造已知角度的离线测试，不计为线上验证成功。"""
import json
import math
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from common.vision import Vision


def circular_error(actual, expected):
    return abs((actual-expected+180)%360-180)


def main():
    inputs=sorted((ROOT/"runtime").glob("kg-slider-*/case-*-attempt-1/background.png"))[-3:]
    if not inputs: raise SystemExit("请先运行滑块演示站采样")
    out=ROOT/"runtime"/f"rotation-offline-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir()
    vision=Vision(ROOT/"models")
    records=[]
    for n,path in enumerate(inputs):
        source=cv2.resize(cv2.imread(str(path)),(320,320))
        center=(159.5,159.5)
        mask=np.zeros((320,320),np.uint8)
        cv2.circle(mask,(160,160),94,255,-1)
        background=source.copy();background[mask>0]=0
        # 内盘保留边缘空间并包含 alpha，已知角度仅用于评测输出，绝不输入识别器。
        piece=cv2.cvtColor(source,cv2.COLOR_BGR2BGRA)
        piece[:,:,3]=mask
        piece=piece[60:260,60:260]
        for angle in (30,90,150,210,270):
            matrix=cv2.getRotationMatrix2D((99.5,99.5),-angle,1)
            rotated=cv2.warpAffine(piece,matrix,(200,200))
            result=vision.rotate_dual(background,rotated)
            expected=(-angle)%360
            error=circular_error(result.value,expected) if result.value is not None else None
            records.append({"source_index":n,"applied_clockwise":angle,"expected_correction":expected,
                            "prediction":result.to_dict(),"error_degrees":error,"within_7_2_degrees":error is not None and error<=7.2})
            if n==0:
                cv2.imwrite(str(out/f"dual-{angle}-background.png"),background)
                cv2.imwrite(str(out/f"dual-{angle}-piece.png"),rotated)
        # 单图只检查同一图片旋转后的预测相对变化，不能替代人工标注绝对正向。
        baseline=vision.rotate_single(source)
        variants=[]
        for angle in (0,90,180,270):
            matrix=cv2.getRotationMatrix2D(center,-angle,1)
            rotated=cv2.warpAffine(source,matrix,(320,320),borderMode=cv2.BORDER_REFLECT)
            result=vision.rotate_single(rotated)
            variants.append({"input_rotation":angle,"prediction":result.to_dict(),
                             "relative_cw_error":circular_error(result.value,(baseline.value-angle)%360),
                             "relative_ccw_error":circular_error(result.value,(baseline.value+angle)%360)})
        (out/f"single-relative-{n}.json").write_text(json.dumps(variants,ensure_ascii=False,indent=2))
    accepted=[r for r in records if r['prediction']['usable']]
    summary={"kind":"offline_synthetic_dual_rotation","images":len(inputs),"cases":len(records),
             "within_7_2_degrees":sum(r['within_7_2_degrees'] for r in records),"accepted":len(accepted),
             "accepted_correct":sum(r['within_7_2_degrees'] for r in accepted),
             "online_verification":False,"directory":str(out)}
    (out/"cases.json").write_text(json.dumps(records,ensure_ascii=False,indent=2))
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
