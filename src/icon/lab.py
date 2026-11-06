"""独立生成本题型的公开题面和私有答案。"""
import numpy as np
from lab.helpers import _png
from icon.lab_shapes import _shape, _paint


def generate(rng):
    kind='icon'
    width,height = rng.choice((320,360,400)),240
    public={"kind":kind,"width":width,"height":height}
    image=np.full((height,width,3),rng.randint(230,250),np.uint8)
    shapes=rng.sample(['circle','triangle','square','cross','arrow','star'],5)
    cells=[(20,25),(130,20),(width-90,30),(65,140),(width-135,145)]
    centers=[]
    for shape,(x,y) in zip(shapes,cells):
        size=rng.randint(45,65);x+=rng.randint(-6,6);y+=rng.randint(-5,5)
        _paint(image,_shape(shape,size,rng.randrange(360)),x,y,[rng.randint(25,180) for _ in range(3)])
        centers.append([x+size/2,y+size/2])
    chosen=rng.sample(range(5),rng.choice((2,3)))
    hint=np.full((56,70*len(chosen),3),255,np.uint8)
    for i,index in enumerate(chosen):
        _paint(hint,_shape(shapes[index],40,rng.randrange(360)),i*70+15,8,[rng.randint(15,160) for _ in range(3)])
    public.update(background=_png(image),hint=_png(hint))
    answer={'points':[centers[i] for i in chosen]}
    return public,answer
