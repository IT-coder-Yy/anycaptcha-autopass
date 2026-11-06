"""独立绘制文字题；答案与选词不发送给页面。"""
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter,ImageChops

WORDS='画蛇添足 守株待兔 亡羊补牢 刻舟求剑 掩耳盗铃 拔苗助长 井底之蛙 滥竽充数 叶公好龙 自相矛盾 狐假虎威 杯弓蛇影 望梅止渴 胸有成竹 破釜沉舟 闻鸡起舞 水落石出 对牛弹琴 指鹿为马 鸟语花香 山清水秀 春暖花开 海阔天空 风和日丽 大公无私 全心全意 同舟共济 一心一意 三心二意 七上八下 九牛一毛 十全十美 千军万马 万紫千红 四面八方 五湖四海 百发百中 半途而废 名副其实 见多识广'.split()


def generate_language(kind,rng):
    from lab.helpers import _png
    directory=Path(os.environ.get('CAPTCHA_LAB_FONT_DIR','/usr/share/fonts/noto-cjk'))
    family=rng.choice(('NotoSansCJK','NotoSerifCJK'))
    weight=rng.choice(('Black','Bold')) if kind=='font' else rng.choice(('Regular','Medium'))
    font_path=directory/f'{family}-{weight}.ttc'
    if not font_path.is_file():raise ValueError('本地题库字体缺失；设置 CAPTCHA_LAB_FONT_DIR')
    word=rng.choice(WORDS);chars=list(word) if kind=='order' else rng.sample(list(dict.fromkeys(''.join(WORDS))),6)
    order=list(range(len(chars)));rng.shuffle(order)
    cells=[(18,10),(132,12),(248,15),(18,100),(132,95),(248,100)] if len(chars)==6 else [(30,16),(232,14),(75,100),(248,98)]
    image=Image.new('RGB',(360,180),tuple(rng.randrange(235,255) for _ in range(3)))
    points=[None]*len(chars);odd=rng.randrange(len(chars));hollow_majority=rng.choice((True,False))
    for position,index in enumerate(order):
        size=rng.randint(38,46);font=ImageFont.truetype(str(font_path),size,index=2)
        mask=Image.new('L',(70,70));d=ImageDraw.Draw(mask);d.text((35,35),chars[index],font=font,anchor='mm',fill=255)
        if kind=='font' and (hollow_majority != (index==odd)):
            mask=ImageChops.subtract(mask,mask.filter(ImageFilter.MinFilter(5)))
        angle=rng.randint(-25,25) if kind=='font' else rng.randint(-18,18)
        mask=mask.rotate(angle,Image.Resampling.BICUBIC)
        x,y=cells[position];x+=rng.randint(-4,4);y+=rng.randint(-3,3)
        image.paste(tuple(rng.randrange(20,140) for _ in range(3)),(x,y,x+70,y+70),mask)
        points[index]=[x+35,y+35]
    public={'kind':kind,'width':360,'height':180,'background':_png(np.asarray(image)[:,:,::-1]),
            'instruction':'请点击1个字体不同的文字' if kind=='font' else '按成语语序依次点击全部文字'}
    return public,{'points':[points[odd]] if kind=='font' else points}
