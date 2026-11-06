"""独立生成本题型的公开题面和私有答案。"""
import cv2
import numpy as np
from lab.helpers import _png, _texture


def generate(rng):
    kind='drag2d'
    width,height = rng.choice((320,360,400)),240
    public={"kind":kind,"width":width,"height":height}
    image=_texture(rng,width,height); size=rng.randint(48,65)
    mask=np.zeros((size,size),np.uint8)
    cv2.rectangle(mask,(8,12),(size-10,size-8),255,-1)
    cv2.circle(mask,(size//2,12),8,255,-1)
    cv2.circle(mask,(8,size//2),7,0,-1)
    x,y=rng.randint(95,width-size-5),rng.randint(10,height-size-5)
    piece=np.dstack((image[y:y+size,x:x+size].copy(),mask))
    shadow=image.copy(); region=shadow[y:y+size,x:x+size]
    region[mask>0]=(region[mask>0].astype(float)*.32).astype(np.uint8)
    # 干扰块保留矩形形状；独立评测真实 GoCaptcha 的多种轮廓干扰。
    dx,dy=25,rng.randint(10,height-size-5)
    shadow[dy:dy+size//2,dx:dx+size//2]//=2
    public.update(background=_png(shadow),piece=_png(piece),start=[8,height-size-8])
    answer={'point':[x,y]}
    return public,answer
