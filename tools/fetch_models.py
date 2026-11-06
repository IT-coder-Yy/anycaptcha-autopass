"""依据来源清单下载模型，校验固定 SHA-256；不运行上游安装脚本。"""
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models"
TARGET.mkdir(exist_ok=True)
ITEMS = json.loads((TARGET / "manifest.json").read_text())


def fetch(item):
    name, size = item['name'], item['bytes']
    dest = TARGET / name
    if dest.exists():
        data = dest.read_bytes()
    else:
        with urllib.request.urlopen(item['url'], timeout=120) as response:
            data = response.read(size + 1)
    if len(data) != size:
        raise ValueError(f"{name}: 文件大小不一致 {len(data)} != {size}")
    actual_git_sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if item.get('git_blob_sha1') and actual_git_sha != item['git_blob_sha1']:
        raise ValueError(f"{name}: Git 对象校验失败")
    if hashlib.sha256(data).hexdigest() != item['sha256']:
        raise ValueError(f"{name}: SHA-256 校验失败")
    temporary = dest.with_suffix(".partial")
    temporary.write_bytes(data)
    temporary.replace(dest)
    print(json.dumps({'name':name,'bytes':len(data),'sha256':item['sha256'],'verified':True}), flush=True)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, ITEMS))
