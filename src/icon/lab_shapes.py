"""独立绘制本地图标题库的形状。"""
import math
import numpy as np
from PIL import Image, ImageDraw


def _shape(kind, size, angle):
    # Pillow 高分辨率绘制，与 OpenCV 的识别/旋转实现独立。
    image=Image.new('L',(256,256)); draw=ImageDraw.Draw(image)
    if kind=='circle': draw.ellipse((35,35,221,221),fill=255)
    elif kind=='triangle': draw.polygon([(128,24),(231,218),(25,218)],fill=255)
    elif kind=='square': draw.rectangle((40,40,216,216),fill=255)
    elif kind=='cross':
        draw.rectangle((93,25,163,231),fill=255);draw.rectangle((25,93,231,163),fill=255)
    elif kind=='arrow':
        draw.polygon([(25,89),(137,89),(137,30),(235,128),(137,226),(137,167),(25,167)],fill=255)
    else:
        points=[(128+(104 if i%2==0 else 44)*math.cos(i*math.pi/5-math.pi/2),
                 128+(104 if i%2==0 else 44)*math.sin(i*math.pi/5-math.pi/2)) for i in range(10)]
        draw.polygon(points,fill=255)
    return np.asarray(image.rotate(angle,resample=Image.Resampling.BICUBIC).resize((size,size),Image.Resampling.LANCZOS))

def _paint(background, mask, x, y, color):
    h,w=mask.shape; alpha=mask[:,:,None].astype(np.float32)/255
    background[y:y+h,x:x+w] = np.round(background[y:y+h,x:x+w]*(1-alpha)+np.array(color)*alpha).astype(np.uint8)
