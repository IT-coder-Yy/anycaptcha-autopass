import argparse
import asyncio
import json
import os
from pathlib import Path
from lab.choices import FIXED_KINDS
from lab.printed import INPUT_KINDS

LAB_KINDS = ('restore', 'drag2d', 'icon', 'rotate-single', 'font', 'order', *INPUT_KINDS)


def main():
    parser=argparse.ArgumentParser(description="本地验证码识别与演示站验证")
    commands=parser.add_subparsers(dest="command",required=True)
    speech=commands.add_parser("transcribe-digits",help="调用 ASR 听写本地数字录音")
    speech.add_argument("audio",type=Path,metavar="AUDIO",help="本地音频文件，例如 WAV 或 MP3")
    demo=commands.add_parser("demo",help="对官方演示站进行有限次数验证")
    demo.add_argument("--site",choices=["kg","yidun","gocaptcha","lab"],required=True)
    demo.add_argument("--kind",choices=["text","slider","rotate",*LAB_KINDS],required=True,
                      help="normal/arithmetic 在 lab 使用已知标注验证输入表单")
    demo.add_argument("--cases",type=int,default=3)
    attempts=demo.add_mutually_exclusive_group()
    attempts.add_argument("--attempts",type=int,default=None,
                      help="点字默认最多 3 道，首题未解出或答错后有限换题；设为 1 仅测首题。其他题型默认 1")
    attempts.add_argument('--retries',type=int,help='首题之外的重试次数，0–3，3 表示总计最多 4 道')
    demo.add_argument('--retry-low-confidence',action='store_true',help='几何题型明确允许低置信度未提交后换题；仍计入分母')
    demo.add_argument("--seconds",type=float,default=float(os.environ.get("CAPTCHA_TIMEOUT","120")))
    demo.add_argument("--headless",action="store_true")
    demo.add_argument("--negative-control",action="store_true",help="故意错误点击/拖动，验证失败识别；不计入算法通过率")
    demo.add_argument("--recovery-control",action="store_true",help="新题型首题故意答错，再验证有限恢复；与正常通过率分开")
    demo.add_argument("--seed",type=int,default=0,help="自建题库的生成种子，不传给识别器")
    demo.add_argument("--scale",type=float,default=1,help="自建题库显示比例，0.6–1.4")
    demo.add_argument("--mode",choices=['trigger','embedded','popup'],default='trigger',
                      help="易盾滑块展示方式：触发式、嵌入式、弹出式")
    recognize=commands.add_parser("recognize",help="识别图片或评测本地题库，photo-vqa 调用视觉模型 API")
    recognize.add_argument("--kind",choices=["text","slider","rotate-dual",*LAB_KINDS,*FIXED_KINDS],required=True)
    recognize.add_argument("image",type=Path,nargs="?",metavar="IMAGE_OR_CASE",
                           help="输入图片。normal/arithmetic 可选一张内置标注图，明示区分大小写的 normal 及 arithmetic 使用 PP-OCRv5 mobile，其余 normal 使用默认 ddddocr。"
                                "photo-vqa/color-letter/shape-pairing 可选一个题号（如 object-01、letter-08、G14）。内置题库省略参数时评测整组。")
    recognize.add_argument("--piece",type=Path,help="碎片图；icon 模式传入提示图")
    recognize.add_argument('--icon-profile',choices=['generic','gocaptcha','yidun'],default='generic',help='图标配置：通用、GoCaptcha 静态模板、易盾提示轮廓')
    recognize.add_argument("--band",type=int,nargs=4,metavar=('X','Y','W','H'),help="restore 的可见移动横带区域")
    lab=commands.add_parser('serve-lab',help='启动仅监听本机的自建验证码题库')
    lab.add_argument('--port',type=int,default=8765)
    lab.add_argument('--seed',type=int,default=0)
    lab.add_argument('--kind',choices=(*LAB_KINDS,*FIXED_KINDS),default='restore',help='启动时显示的本地题型入口')
    args=parser.parse_args()
    root=Path(os.environ.get("CAPTCHA_PROJECT_ROOT",Path(__file__).resolve().parents[1]))
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH",str(root/".browsers"))
    if args.command=='serve-lab':
        from lab.server import make_server
        server=make_server(args.seed,args.port)
        print(f'本地题库：http://127.0.0.1:{server.server_port}/?kind={args.kind}',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()
        return
    if args.command=="transcribe-digits":
        from audio_digits.asr import transcribe
        try:
            result = transcribe(args.audio, root / ".env")
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command=="demo":
        if args.retries is not None or args.retry_low_confidence:
            if args.kind not in LAB_KINDS:
                parser.error('--retries 和 --retry-low-confidence 支持的题型为 '+', '.join(LAB_KINDS))
            if args.retry_low_confidence and args.kind in INPUT_KINDS:
                parser.error('固定标注的表单对照没有置信度，可用 --recovery-control 检查换题恢复')
        if args.retries is not None:
            if not 0<=args.retries<=3:parser.error('--retries 必须在 0–3 之间')
            args.attempts=args.retries+1
        if args.site!='lab' and (args.seed!=0 or args.scale!=1):
            parser.error('--seed 与 --scale 目前只用于自建题库 --site lab')
        if args.mode!='trigger' and (args.site!='yidun' or args.kind!='slider'):
            parser.error('--mode 目前只用于易盾滑块的三种展示方式')
        if args.attempts is None:
            args.attempts=int(os.environ.get("CAPTCHA_MAX_ATTEMPTS","3" if args.kind=="text" else "1"))
        if args.site=='yidun' and args.kind=='icon':
            from icon.demo import run_yidun_icons
            result=asyncio.run(run_yidun_icons(root,args.cases,args.attempts,args.seconds,args.headless,
                                             args.negative_control,args.recovery_control,args.retry_low_confidence))
        elif args.kind in LAB_KINDS:
            if not (args.site=='lab' or args.site=='gocaptcha' and args.kind in ('drag2d','icon') or args.site=='kg' and args.kind in ('font','order')):
                parser.error('新题型支持 lab；GoCaptcha 支持 drag2d/icon；凯格支持 font/order')
            if not .6<=args.scale<=1.4:parser.error('--scale 必须在 0.6–1.4 之间')
            from common.geometry_demo import run_geometry
            result=asyncio.run(run_geometry(root,args.site,args.kind,args.cases,args.attempts,args.seconds,
                                           args.headless,args.negative_control,args.recovery_control,args.seed,args.scale,
                                           args.retry_low_confidence))
        elif args.recovery_control or args.site=='lab':
            parser.error('本地题库与 --recovery-control 支持的题型为 '+', '.join(LAB_KINDS))
        elif args.site=="gocaptcha" and args.kind=="rotate":
            from rotate_overlap.demo import run_rotation
            result=asyncio.run(run_rotation(root,args.cases,args.attempts,args.seconds,args.headless,args.negative_control))
        else:
            if args.site=="gocaptcha" or args.kind=="rotate": parser.error("旋转演示目前使用 --site gocaptcha --kind rotate")
            from common.demo import run_demo
            result=asyncio.run(run_demo(root,args.site,args.kind,args.cases,args.attempts,args.seconds,args.headless,args.negative_control,mode=args.mode))
    elif args.kind in FIXED_KINDS:
        from lab.choices import evaluate
        try:
            result = evaluate(args.kind, str(args.image) if args.image else None)
        except (LookupError, OSError, ValueError) as exc:
            parser.error(str(exc))
    elif args.kind in INPUT_KINDS:
        from common.ocr import evaluate_samples
        try:
            result = evaluate_samples(args.kind, args.image)
        except (LookupError, FileNotFoundError) as exc:
            parser.error(str(exc))
    else:
        if args.image is None: parser.error('该识别题型需要输入图片')
        import cv2
        from common.vision import Vision
        vision=Vision(root/"models")
        image=cv2.imread(str(args.image))
        if image is None: parser.error("无法读取输入图片")
        if args.kind in ('restore','drag2d','icon'):
            from restore.solver import restore_band
            from drag2d.solver import drag_piece
            from icon.solver import click_icons
            if args.kind=='restore':
                if not args.band:parser.error('restore 需要 --band X Y W H')
                result=restore_band(image,args.band).to_dict()
            else:
                if not args.piece:parser.error('该模式需要 --piece 碎片或提示图')
                aux=cv2.imread(str(args.piece),cv2.IMREAD_UNCHANGED)
                if aux is None:parser.error('无法读取碎片或提示图')
                if args.kind=='icon' and args.icon_profile=='gocaptcha':
                    from icon.templates import click_gocaptcha_icons
                    result=click_gocaptcha_icons(image,aux).to_dict()
                elif args.kind=='icon' and args.icon_profile=='yidun':
                    from icon.yidun import click_yidun_icons
                    result=click_yidun_icons(image,aux).to_dict()
                else:result=(drag_piece(image,aux) if args.kind=='drag2d' else click_icons(image,aux)).to_dict()
        elif args.kind in ('font','order'):
            from font.solver import odd_font
            from order.solver import order_words
            result=(odd_font if args.kind=='font' else order_words)(vision,image).to_dict()
        elif args.kind=="text": result=vision.text(image).to_dict()
        elif args.kind=="rotate-single": result=vision.rotate_single(image).to_dict()
        else:
            if not args.piece: parser.error("该模式需要 --piece")
            if args.kind=="slider": result=vision.slider(args.image.read_bytes(),args.piece.read_bytes()).to_dict()
            else: result=vision.rotate_dual(image,cv2.imread(str(args.piece),cv2.IMREAD_UNCHANGED)).to_dict()
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.command=="demo":
        if args.negative_control:
            raise SystemExit(0 if result['failed']==result['cases'] else 2)
        raise SystemExit(0 if result['passed']==result['cases'] else (2 if result['unknown'] else 1))


if __name__=="__main__": main()
