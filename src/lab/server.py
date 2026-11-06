"""本机题库与独立 HTTP 裁判。答案仅保存在服务端，不发送给页面或识别器。"""
from contextlib import contextmanager
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import random
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from restore.lab import generate as generate_restore
from drag2d.lab import generate as generate_drag2d
from icon.lab import generate as generate_icon
from rotate_single.lab import generate as generate_rotation
from common.language_lab import generate_language

GENERATORS = {
    'restore': generate_restore, 'drag2d': generate_drag2d, 'icon': generate_icon,
    'rotate-single': generate_rotation,
    'font': partial(generate_language, 'font'), 'order': partial(generate_language, 'order'),
}
KINDS = tuple(GENERATORS)


def generate(kind, rng):
    """返回公开题面和私有答案，服务端只把前者发送给浏览器。"""
    if kind not in KINDS: raise ValueError('未知题型')
    return GENERATORS[kind](rng)


class ChallengeStore:
    def __init__(self, seed=0):
        self.rng=random.Random(seed);self.answers={};self.lock=threading.Lock()

    def issue(self, kind):
        with self.lock:
            public,answer=generate(kind,self.rng)
            key=uuid.uuid4().hex
            now=time.monotonic()
            self.answers={k:v for k,v in self.answers.items() if v[0]>now}
            self.answers[key]=(now+180,kind,answer)
        return dict(public,key=key)

    def verify(self, key, submitted):
        with self.lock: record=self.answers.pop(key,None)
        accepted=False
        if record and record[0]>time.monotonic():
            _,kind,answer=record
            try:
                if kind=='rotate-single':
                    value=float(submitted['angle'])
                    accepted=math.isfinite(value) and abs((value-answer['angle']+180)%360-180)<=8
                elif kind=='restore':
                    value=float(submitted['shift'])
                    delta=(value-answer['shift'])%answer['width']
                    accepted=math.isfinite(value) and min(delta,answer['width']-delta)<=3
                else:
                    got=[submitted['point']] if kind=='drag2d' else submitted['points']
                    expected=[answer['point']] if kind=='drag2d' else answer['points']
                    tolerance=4 if kind=='drag2d' else 12
                    accepted=len(got)==len(expected) and all(
                        len(a)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in a)
                        and math.dist(a,b)<=tolerance for a,b in zip(got,expected))
            except (KeyError,TypeError,ValueError,OverflowError):pass
        return {'key':key,'accepted':bool(accepted)}


def make_server(seed=0, port=0):
    store=ChallengeStore(seed)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,body,content_type='application/json',status=200):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            parsed=urlsplit(self.path)
            if parsed.path=='/':
                return self.send(Path(__file__).with_name('lab.html').read_bytes(),'text/html; charset=utf-8')
            if parsed.path=='/challenge':
                kind=parse_qs(parsed.query).get('kind',['restore'])[0]
                if kind in KINDS:return self.send(store.issue(kind))
            self.send({'error':'not found'},status=404)
        def do_POST(self):
            if urlsplit(self.path).path!='/verify':return self.send({'error':'not found'},status=404)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<4096:raise ValueError('请求长度无效')
                data=json.loads(self.rfile.read(length))
                result=store.verify(data['key'],data['answer'])
            except (ValueError,KeyError,TypeError):return self.send({'error':'bad request'},status=400)
            self.send(result)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


@contextmanager
def local_server(seed=0):
    server=make_server(seed);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}/'
    finally:server.shutdown();server.server_close();thread.join()
