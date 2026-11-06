"""汇总实际报告；拒绝作答、异常与刷新均保留在分母中。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from common.evaluation import summarize_reports


def summarize(folder):
    reports=json.loads((folder/'report.json').read_text())
    return {'directory': str(folder), **summarize_reports(reports)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+',type=Path)
    args=parser.parse_args()
    print(json.dumps([summarize(p) for p in args.directories],ensure_ascii=False,indent=2))
