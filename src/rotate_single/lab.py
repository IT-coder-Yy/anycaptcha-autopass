"""独立生成本题型的公开题面和私有答案。"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from lab.helpers import _png


def generate(rng):
    kind='rotate-single'
    width,height = rng.choice((320,360,400)),240
    public={"kind":kind,"width":width,"height":height}
    directory=Path(__file__).with_name('data')
    sources=json.loads((directory/'manifest.json').read_text())
    source=rng.choice(sources)
    image=Image.open(directory/source['file']).convert('RGB')
    w,h=image.size;side=min(w,h)
    image=image.crop(((w-side)//2,(h-side)//2,(w+side)//2,(h+side)//2)).resize((300,300),Image.Resampling.LANCZOS)
    angle=rng.randrange(360)
    rotated=image.rotate(angle,resample=Image.Resampling.BICUBIC)
    # 圆形显示避免旋转后四角空白；角度、原图和文件名均不传给浏览器。
    mask=Image.new('L',(300,300));ImageDraw.Draw(mask).ellipse((0,0,299,299),fill=255)
    visible=Image.composite(rotated,Image.new('RGB',(300,300),'white'),mask)
    public.update(width=300,height=300,background=_png(np.asarray(visible)[:,:,::-1]))
    answer={'angle':angle,'source':source['file']}
    return public,answer
