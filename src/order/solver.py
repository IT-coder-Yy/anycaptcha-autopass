"""本机 OCR 与固定词库的一对一排列；不调用在线模型。"""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import math
import time
import cv2
import numpy as np
from PIL import Image
from common.text_ocr import MODELS, prepare
from common.text_support import detect_characters
from common.vision import Prediction


@lru_cache(maxsize=1)
def vocabulary():
    directory=Path(__file__).with_name('data')
    return tuple(sorted(set((directory/'idioms.txt').read_text().splitlines()+
                            (directory/'poetry.txt').read_text().splitlines())))


def readings(vision,image,boxes):
    from font.solver import ink_mask
    views=[]
    angles=(-60,-30,0,30,60)
    for x1,y1,x2,y2 in boxes:
        crop=image[max(0,y1-2):min(image.shape[0],y2+2),max(0,x1-2):min(image.shape[1],x2+2)]
        mask=ink_mask(crop)
        normalized=np.repeat((255-mask*255)[:,:,None],3,axis=2)
        for variant in (crop,normalized):
            im=Image.fromarray(variant[:,:,::-1])
            for angle in angles:
                view=im.rotate(angle,Image.Resampling.BICUBIC,expand=True,fillcolor='white')
                views.append(prepare(np.asarray(view)[:,:,::-1]))
    models=[]
    for name in MODELS:
        session=vision.session(name)
        chars=['blank']+session.get_modelmeta().custom_metadata_map['character'].splitlines()+[' ']
        peaks=[]
        for start in range(0,len(views),8):
            raw=session.run(None,{session.get_inputs()[0].name:np.stack(views[start:start+8])})[0]
            if raw.shape[-1]!=len(chars) or not np.allclose(raw.sum(axis=-1),1,atol=.005):
                raise ValueError('OCR 字符表或概率格式变化')
            peaks.extend(raw.max(axis=1))
        peaks=np.array(peaks).reshape(len(boxes),10,len(chars)).max(axis=1)
        models.append([{chars[int(i)]:float(row[i]) for i in np.argsort(row)[-12:]
                        if len(chars[int(i)])==1 and '\u4e00'<=chars[int(i)]<='\u9fff'} for row in peaks])
    return models


def rank_words(observations,words):
    """只取与全部可见字数相等的词句，防止把局部成语当整题答案。"""
    n=len(observations);available=set().union(*(set(x) for x in observations));ranked=[]
    for word in words:
        if len(word)!=n or not set(word)<=available:continue
        candidates=[[i for i,o in enumerate(observations) if o.get(c,0)>=.10] for c in word]
        if any(not x for x in candidates):continue
        # n<=8；剪枝只枚举有实际 OCR 支持的位置。
        def assign(index,used,total):
            if index==n:
                ranked.append((total/n,word,tuple(used)));return
            for j in candidates[index]:
                if j not in used:assign(index+1,used+[j],total+math.log(observations[j][word[index]]))
        assign(0,[],0.)
    return sorted(ranked,reverse=True)


def order_words(vision,image):
    start=time.monotonic();boxes=sorted(detect_characters(vision,image));detail={'boxes':boxes}
    if not 4<=len(boxes)<=8:
        return Prediction('order',False,reason='需要完整检测到4至8个文字',detail=detail)
    models=readings(vision,image,boxes);detail['ocr_candidates']=models
    rankings=[rank_words(m,vocabulary()) for m in models]
    detail['rankings']=[[{'log_score':s,'word':w,'assignment':p} for s,w,p in r[:4]] for r in rankings]
    if any(not r for r in rankings):
        return Prediction('order',False,reason='OCR 未形成词库内的完整排列',detail=detail,elapsed_ms=(time.monotonic()-start)*1000)
    best=rankings[0][0];word,assignment=best[1:]
    # 同一组字能组成多个词句时，图像本身不能决定语义，必须弃权。
    anagrams=[w for w in vocabulary() if Counter(w)==Counter(word)]
    gaps=[r[0][0]-r[1][0] if len(r)>1 else 99. for r in rankings]
    floor=min(m[j].get(c,0.) for m in models for c,j in zip(word,assignment))
    usable=(rankings[1][0][1:]==best[1:] and len(anagrams)==1 and min(gaps)>=.25 and floor>=.5)
    detail.update(word=word,anagrams=anagrams,log_gaps=gaps,minimum_probability=floor)
    points=[[(boxes[j][0]+boxes[j][2])/2,(boxes[j][1]+boxes[j][3])/2] for j in assignment]
    return Prediction('order',bool(usable),floor,points=points,
                      reason='双模型支持唯一词序' if usable else 'OCR 分歧、位置歧义或同字异序',detail=detail,
                      elapsed_ms=(time.monotonic()-start)*1000)
