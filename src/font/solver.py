"""本地字体辨异：只在笔画厚度形成稳定离群值时点选，不猜任意字体风格。"""
import os
import time
import cv2
import numpy as np
from common.text_support import detect_characters
from common.vision import Prediction


def _foreground_distance(crop):
    """由边界估计平滑背景，保留与背景颜色不同的笔画。"""
    lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB).astype(np.float32)
    h,w=lab.shape[:2];yy,xx=np.mgrid[:h,:w]
    design=np.stack([np.ones((h,w)),xx/max(w,1),yy/max(h,1)],-1)
    border=(xx<2)|(xx>=w-2)|(yy<2)|(yy>=h-2)
    coef=np.linalg.lstsq(design[border],lab[border],rcond=None)[0]
    # 两轮剔除边界笔画和干扰线，避免前景污染背景拟合。
    for _ in range(2):
        residual=np.linalg.norm(lab-design@coef,axis=2)
        clean=border&(residual<=np.percentile(residual[border],70))
        if clean.sum()>12:coef=np.linalg.lstsq(design[clean],lab[clean],rcond=None)[0]
    distance=np.clip(np.linalg.norm(lab-design@coef,axis=2),0,255).astype(np.uint8)
    return lab,distance


def ink_mask(crop):
    """保留原分割语义，语序模块也使用此接口。"""
    _,distance=_foreground_distance(crop)
    threshold,_=cv2.threshold(distance,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return (distance>max(12,threshold)).astype(np.uint8)


def stroke_feature(mask):
    h,w=mask.shape
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    maxima=(distance>=cv2.dilate(distance,np.ones((3,3),np.uint8))-.01)&(distance>0)
    values=distance[maxima]
    return float(np.percentile(values,85)*2/max(h,w)) if len(values) else 0.


def odd_font_baseline(vision,image,count=1):
    start=time.monotonic();boxes=sorted(detect_characters(vision,image));features=[]
    for x1,y1,x2,y2 in boxes:
        crop=image[max(0,y1-3):min(image.shape[0],y2+3),max(0,x1-3):min(image.shape[1],x2+3)]
        mask=ink_mask(crop);features.append(stroke_feature(mask))
    detail={'boxes':boxes,'stroke_widths':features,'scope':'粗细/空心离群；任意字体家族辨识未覆盖'}
    if count!=1 or not 4<=len(boxes)<=8 or min(features,default=0)<=0:
        return Prediction('font',False,reason='仅支持单个离群字，或字符检测不完整',detail=detail)
    logs=np.log(np.array(features));deviation=np.abs(logs-np.median(logs));indices=np.argsort(deviation)[::-1]
    index=int(indices[0]);ratio=float(np.exp(deviation[index]));gap=float(deviation[index]-deviation[indices[1]])
    usable=ratio>=1.75 and gap>=.25
    x1,y1,x2,y2=boxes[index];detail.update(width_ratio=ratio,log_gap=gap,ratio_floor=1.75,gap_floor=.25)
    return Prediction('font',bool(usable),gap,points=[[(x1+x2)/2,(y1+y2)/2]],
                      reason='笔画厚度有唯一离群字' if usable else '候选字体差异不足或多个候选歧义',detail=detail,
                      elapsed_ms=(time.monotonic()-start)*1000)


def color_mask(crop):
    """用成片的前景主色分离笔画，避免高对比干扰线主导 Otsu 阈值。"""
    lab,distance=_foreground_distance(crop)
    candidate=distance>=12
    candidate[:2]=False;candidate[-2:]=False;candidate[:,:2]=False;candidate[:,-2:]=False
    pixels=lab[candidate]
    if len(pixels)<12:return ink_mask(crop)
    bins=(pixels/8).astype(int)
    keys,counts=np.unique(bins,axis=0,return_counts=True)
    for index in counts.argsort()[::-1][:12]:
        center=np.median(pixels[np.all(bins==keys[index],axis=1)],axis=0)
        mask=((np.linalg.norm(lab-center,axis=2)<=18)&(distance>=8)).astype(np.uint8)
        if _linearity(mask)>=.04:return mask
    return ink_mask(crop)


def _linearity(mask):
    coords=np.column_stack(np.nonzero(mask))
    if len(coords)<4:return 0.
    eigen=np.linalg.eigvalsh(np.cov(coords.T))
    return float(eigen[0]/max(eigen[-1],1))


def stroke_profile(mask):
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    maxima=(distance>=cv2.dilate(distance,np.ones((3,3),np.uint8))-.01)&(distance>0)
    values=distance[maxima]
    return np.percentile(values,[25,50,75,85])*2/max(mask.shape) if len(values) else np.zeros(4)


def profile_vote(features):
    """每个分位仍采用旧门槛；至少两个分位支持唯一位置。"""
    votes=[];gaps=[]
    for column in np.asarray(features).T:
        logs=np.log(np.maximum(column,1e-5));deviation=abs(logs-np.median(logs))
        indices=np.argsort(deviation)[::-1];first,second=indices[:2]
        gap=float(deviation[first]-deviation[second]);gaps.append(gap)
        votes.append(int(first) if np.exp(deviation[first])>=1.75 and gap>=.25 else -1)
    winners=[i for i in set(votes) if i>=0 and votes.count(i)>=2]
    return (winners[0] if len(winners)==1 else -1),votes,gaps


def odd_font(vision,image,count=1,engine=None):
    engine=engine or os.environ.get('CAPTCHA_FONT_ENGINE','profile')
    if engine=='baseline':
        result=odd_font_baseline(vision,image,count);result.detail['engine']=engine
        return result
    if engine!='profile':raise ValueError('CAPTCHA_FONT_ENGINE 必须是 baseline 或 profile')
    start=time.monotonic();boxes=sorted(detect_characters(vision,image))
    kept=[];features=[];residual_features=[];overlaps=[];discarded=[]
    for i,(x1,y1,x2,y2) in enumerate(boxes):
        crop=image[max(0,y1-3):min(image.shape[0],y2+3),max(0,x1-3):min(image.shape[1],x2+3)]
        original=ink_mask(crop);mask=color_mask(crop)
        edge=x1<=1 or y1<=1 or x2>=image.shape[1]-1 or y2>=image.shape[0]-1
        # 只过滤被截断的边缘细条；内部“一”字也可能只有一条笔画，不能误删。
        if edge and min(x2-x1,y2-y1)<.3*max(x2-x1,y2-y1):
            discarded.append(i);continue
        kept.append(i);features.append(stroke_profile(mask));residual_features.append(stroke_profile(original))
        overlaps.append(float((mask&original).sum()/max(min(mask.sum(),original.sum()),1)))
    detail={'engine':engine,'boxes':boxes,'kept_indices':kept,'discarded_indices':discarded,
            'stroke_profiles':np.asarray(features).tolist(),'residual_profiles':np.asarray(residual_features).tolist(),
            'mask_overlap':overlaps,'ratio_floor':1.75,'gap_floor':.25,
            'scope':'粗细/空心局部特征；不保证任意字体家族分类'}
    if count!=1 or not 4<=len(kept)<=8 or any(np.min(f)<=0 for f in features):
        return Prediction('font',False,reason='仅支持单个离群字，或字符检测不完整',detail=detail)
    features=np.array(features);residual_features=np.array(residual_features)
    index,votes,gaps=profile_vote(features)
    fallback,old_votes,old_gaps=profile_vote(residual_features)
    source='color'
    if index<0 and fallback>=0 and overlaps[fallback]>=.75:
        shift=np.max(abs(np.log(features[fallback]/np.maximum(residual_features[fallback],1e-5))))
        if shift<=np.log(2.1):index=fallback;source='residual'
    detail.update(votes=votes,residual_votes=old_votes,source=source)
    usable=index>=0;points=[];score=0.
    if usable:
        x1,y1,x2,y2=boxes[kept[index]];points=[[(x1+x2)/2,(y1+y2)/2]]
        selected_votes,selected_gaps=(votes,gaps) if source=='color' else (old_votes,old_gaps)
        score=min(g for v,g in zip(selected_votes,selected_gaps) if v==index)
        detail['selected_index']=kept[index]
    return Prediction('font',usable,score,points=points,
                      reason='多个笔画分位支持唯一离群字' if usable else '笔画分布不足以区分字体或候选歧义',
                      detail=detail,elapsed_ms=(time.monotonic()-start)*1000)
