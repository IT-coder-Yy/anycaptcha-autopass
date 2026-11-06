"""通用简单图标顺序匹配。"""
from itertools import permutations
import math
import time
import cv2
import numpy as np
from common.vision import _finish


def _components(image,hint=False):
    """按实色区域分离图形；保留透明提示的原色，闭运算连接被细线切断的部分。"""
    a=image[:,:,:3];height,width=a.shape[:2]
    colors,counts=np.unique(a.reshape(-1,3),axis=0,return_counts=True)
    seeds=colors[np.argsort(counts)[-100:]]
    candidates=[]; used=[]
    for color in seeds[::-1]:
        if any(np.max(np.abs(color.astype(float)-old))<18 for old in used):continue
        used.append(color.astype(float))
        mask=(np.max(np.abs(a.astype(float)-color),axis=2)<(16 if hint else 24)).astype('uint8')
        if image.shape[2]==4:mask[image[:,:,3]<50]=0
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((4,4),np.uint8))
        n,cc,stats,_=cv2.connectedComponentsWithStats(mask)
        for i in range(1,n):
            x,y,w,h,area=map(int,stats[i]);limit=height*.9 if hint else min(height,width)*.4
            if area<50 or min(w,h)<10 or max(w,h)>limit or max(w,h)/min(w,h)>2.2:continue
            if x==0 or y==0 or x+w==width or y+h==height:continue
            if any(abs(x-b[0])+abs(y-b[1])+abs(w-b[2])+abs(h-b[3])<14 for b,_ in candidates):continue
            candidates.append(([x,y,w,h],(cc[y:y+h,x:x+w]==i).astype('uint8')))
    candidates.sort(key=lambda v:int(v[1].sum()),reverse=True)
    return sorted(candidates[:(5 if hint else 12)],key=lambda v:v[0][0])

def _normalize(mask):
    x,y,w,h = cv2.boundingRect(mask)
    if not w or not h:
        return np.zeros((64,64),np.uint8)
    scale = 52/max(w,h)
    resized = cv2.resize(mask[y:y+h,x:x+w],(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_NEAREST)
    out = np.zeros((64,64),np.uint8)
    y,x = (64-resized.shape[0])//2,(64-resized.shape[1])//2
    out[y:y+resized.shape[0],x:x+resized.shape[1]] = resized
    return out

def _hint_count(image):
    """用独立的前景投影检查是否漏掉提示；不把某一个高分匹配当成整题完整。"""
    if image.shape[2] == 4:
        mask = (image[:,:,3] > 100).astype(np.uint8)
    else:
        color = np.median(np.concatenate([image[0,:,:3], image[-1,:,:3]]), axis=0)
        mask = (np.max(np.abs(image[:,:,:3].astype(float)-color),axis=2)>30).astype(np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    occupied = np.flatnonzero(mask.sum(axis=0)>2)
    groups = np.split(occupied,np.where(np.diff(occupied)>6)[0]+1)
    count = 0
    for group in groups:
        if not len(group):continue
        part = mask[:,group[0]:group[-1]+1]
        _,_,w,h = cv2.boundingRect(part)
        if part.sum()<50 or min(w,h)<8:continue
        if max(w,h)>image.shape[0]*1.2:return None
        count += 1
    return count

def click_icons(background, hint):
    """仅简单、实色、相互分离的图形；颜色不参与身份匹配。"""
    start = time.monotonic()
    hints, targets = _components(hint,True), _components(background)
    expected = _hint_count(hint)
    detail = {'hint_boxes':[b for b,_ in hints],'target_boxes':[b for b,_ in targets],
              'foreground_hint_count':expected}
    if expected!=len(hints) or not 1 <= len(hints) <= 5 or not len(hints) <= len(targets) <= 12:
        return _finish('icon',start,reason='简单图标分割不完整',detail=detail)
    normalized = [_normalize(m) for _,m in targets]
    scores = np.zeros((len(hints),len(targets)))
    for i,(_,mask) in enumerate(hints):
        base = _normalize(mask)
        for angle in range(0,360,5):
            rotated = _normalize(cv2.warpAffine(base,cv2.getRotationMatrix2D((31.5,31.5),angle,1),(64,64),flags=cv2.INTER_NEAREST))
            for j,target in enumerate(normalized):
                union = np.count_nonzero(rotated | target)
                score = np.count_nonzero(rotated & target)/max(1,union)
                scores[i,j] = max(scores[i,j],score)
    ranked = sorted(((sum(math.log(max(scores[i,j],1e-8)) for i,j in enumerate(p)),p)
                     for p in permutations(range(len(targets)),len(hints))),reverse=True)
    total, assignment = ranked[0]
    # 比较整套排列的相对分数，避免提示数量改变同一个局部歧义的门槛。
    gap = total-ranked[1][0] if len(ranked)>1 else 1.
    score = min(scores[i,j] for i,j in enumerate(assignment))
    detail.update({'scores':scores.tolist(),'assignment':list(assignment),'margin':gap})
    usable = score >= .82 and gap >= .06
    points = [[targets[j][0][0]+targets[j][0][2]/2,targets[j][0][1]+targets[j][0][3]/2] for j in assignment]
    return _finish('icon',start,usable,score,points=points,
                   reason='简单图形排列候选' if usable else '图形不匹配或排列存在歧义',detail=detail)
