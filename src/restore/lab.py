"""独立生成本题型的公开题面和私有答案。"""
import numpy as np
from lab.helpers import _png, _texture


def generate(rng):
    kind='restore'
    width,height = rng.choice((320,360,400)),240
    public={"kind":kind,"width":width,"height":height}
    image=_texture(rng,width,height)
    top=rng.choice((0,45,80)); h=rng.randint(65,125)
    shift=rng.randrange(25,width-30)
    moved=image.copy()
    # 切片拼接而不是复用识别器里的 roll 操作。
    strip=image[top:top+h]
    moved[top:top+h]=np.concatenate((strip[:,shift:],strip[:,:shift]),axis=1)
    public.update(background=_png(moved),band=[0,top,width,h])
    answer={'shift':shift,'width':width}
    return public,answer
