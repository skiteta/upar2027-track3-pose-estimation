"""Bounded, label-consistent synthesis in five declared difficulty coordinates.

Descriptors are observable proxies, not identifiable latent causes. No PETA
statistics, model fitting, image generation service or external assets here.
"""
from itertools import combinations
import numpy as np
from PIL import Image, ImageFilter

AXES = ('information', 'observability', 'correspondence', 'geometry', 'context')
HELD_OUT_PAIRS = ((0, 2), (1, 4))
TRAIN_PAIRS = tuple(p for p in combinations(range(5), 2) if p not in HELD_OUT_PAIRS)
PAIRS = ((5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


def descriptors(pixels, keypoints):
    """Five difficulty proxies; GT-derived descriptors are for training only."""
    gray = np.asarray(Image.fromarray(np.asarray(pixels, dtype=np.uint8)).convert('L'), dtype=float)/255
    points = np.asarray(keypoints, dtype=float)
    valid = points[:, 2] > 0
    if not valid.any(): raise ValueError('No labelled target')
    height, width = gray.shape
    gradient = np.abs(np.diff(gray, axis=0)).mean()+np.abs(np.diff(gray, axis=1)).mean()
    information = 1/(1+20*gradient)
    observability = 1-np.mean(points[valid, 2] == 2)
    distances = [np.linalg.norm(points[a,:2]-points[b,:2])/max(width,height)
                 for a,b in PAIRS if valid[a] and valid[b]]
    correspondence = 1/(1+10*np.median(distances)) if distances else .5
    xy = points[valid,:2]/np.array([width-1,height-1])
    geometry = np.mean(np.min(np.c_[xy,1-xy],axis=1) < .08)
    core = gray[height//4:3*height//4,width//4:3*width//4]
    border = np.concatenate((gray[:max(1,height//8)].ravel(),gray[-max(1,height//8):].ravel()))
    context = np.clip(border.std()/(core.std()+border.std()+1e-6),0,1)
    return np.array([information,observability,correspondence,geometry,context], dtype=float)


def sample_plan(rng, weights, *, mode):
    """Same clean/single/pair rates and strength law; only axis weights differ."""
    if mode not in ('uniform','calibrated'): raise ValueError('Unknown sampling mode')
    weights=np.asarray(weights,dtype=float)
    if weights.shape!=(5,) or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError('Five positive finite axis weights required')
    weights=np.ones(5) if mode=='uniform' else weights
    # Consume a fixed number of random draws irrespective of mode/axis choices.
    kind,choice,rare=rng.random(3); strengths=rng.uniform(.25,.75,5)
    if rare<.05: strengths=np.minimum(.85,strengths+.1)
    if kind<.5: axes=()
    elif kind<.75:
        index=min(4,int(np.searchsorted(np.cumsum(weights/weights.sum()),choice,side='right')))
        axes=(index,)
    else:
        probabilities=np.array([weights[a]*weights[b] for a,b in TRAIN_PAIRS]);probabilities/=probabilities.sum()
        index=min(len(TRAIN_PAIRS)-1,int(np.searchsorted(np.cumsum(probabilities),choice,side='right')))
        axes=TRAIN_PAIRS[index]
    return {'axes':list(axes),'strengths':strengths.tolist(),
            'draw_seed':int(rng.integers(0,2**31-1)), 'held_out_combination':False}


def affine_matrix(strength, rng):
    angle=rng.uniform(-12,12)*strength*np.pi/180
    sx,sy=np.exp(rng.uniform(-.15,.15,2)*strength)
    c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s],[s,c]])@np.diag([sx,sy])
    translation=127.5-rotation@np.array([127.5,127.5])+rng.uniform(-12,12,2)*strength
    matrix=np.eye(3);matrix[:2,:2]=rotation;matrix[:2,2]=translation
    return matrix


def transform_points(points, matrix):
    result=np.asarray(points,dtype=float).copy();valid=result[:,2]>0
    result[valid,:2]=(np.c_[result[valid,:2],np.ones(valid.sum())]@matrix.T)[:,:2]
    return result


def guard_observability(original, changed):
    before=np.asarray(original);after=np.asarray(changed)
    visible=before[:,2]==2; remaining=after[:,2]==2
    inside=((after[:,:2]>=0)&(after[:,:2]<=255)).all(1)
    required=min(6,int(np.ceil(visible.sum()*.6)))
    torso=np.array([5,6,11,12]);required_torso=min(2,int(visible[torso].sum()))
    return bool((remaining&inside).sum()>=required and (remaining&inside)[torso].sum()>=required_torso)


def apply_plan(pixels, keypoints, plan, donor=None):
    """Return actual pixels, transformed target labels and full intervention log.

    Target coordinates do not move under occlusion. Outside-crop labels become
    unlabelled for this generated sample only; original annotations never change.
    Donor joints are logged but NOT silently made new target supervision.
    """
    original=np.asarray(pixels,dtype=np.uint8)
    if original.shape!=(256,256,3): raise ValueError('Square256 RGB required')
    original_points=np.asarray(keypoints,dtype=float)
    if original_points.shape!=(17,3) or not np.isfinite(original_points).all(): raise ValueError('Finite17x3 target required')
    axes=tuple(sorted(plan['axes']))
    if len(set(axes))!=len(axes) or any(i not in range(5) for i in axes): raise ValueError('Invalid axes')
    if not axes:return original.copy(),original_points.copy(),{'axes':[],'accepted':True,'changed_pixels':0}
    strengths=np.asarray(plan['strengths'],dtype=float)
    if strengths.shape!=(5,) or not np.isfinite(strengths).all() or np.any((strengths<0)|(strengths>1)):
        raise ValueError('Strengths must lie in0..1')
    rng=np.random.default_rng(plan['draw_seed']);image=Image.fromarray(original)
    points=original_points.copy();events=[]
    if 3 in axes:
        matrix=affine_matrix(strengths[3],rng);inverse=np.linalg.inv(matrix)
        image=image.transform((256,256),Image.Transform.AFFINE,tuple(inverse[:2].ravel()),Image.Resampling.BILINEAR)
        points=transform_points(points,matrix)
        events.append({'axis':'geometry','matrix':matrix.tolist()})
    if 2 in axes:
        if donor is None: raise ValueError('Source-only donor with known coordinates required')
        donor_pixels,donor_points,donor_id=donor
        height=int(90+80*strengths[2]);width=max(24,min(100,round(height*donor_pixels.shape[1]/donor_pixels.shape[0])))
        # Place the distractor near the target, not trivially in blank padding.
        # The later visibility guard rejects destruction of identification cues.
        target=points[points[:,2]>0,:2];center=np.median(target,axis=0)
        separation=.30*np.ptp(target[:,0])+.20*width
        px=int(center[0]+rng.choice([-1,1])*separation-width/2)
        py=int(center[1]-height/2+rng.uniform(-18,18))
        patch=Image.fromarray(donor_pixels).resize((width,height),Image.Resampling.BILINEAR)
        image.paste(patch,(px,py));valid=points[:,2]>0
        covered=valid&(points[:,0]>=px)&(points[:,0]<px+width)&(points[:,1]>=py)&(points[:,1]<py+height)
        points[covered,2]=1
        matrix=np.array([[width/donor_pixels.shape[1],0,px],[0,height/donor_pixels.shape[0],py],[0,0,1.]])
        moved=transform_points(donor_points,matrix)
        events.append({'axis':'correspondence','donor_annotation_id':int(donor_id),
            'donor_keypoints':moved.tolist(),'rectangle':[px,py,width,height],
            'covered_target_joints':int(covered.sum()),
            'limit':'Rectangular pasted competitor; no segmentation or instance-classification claim'})
    if 1 in axes:
        valid=np.flatnonzero(points[:,2]>0);joint=int(rng.choice(valid))
        width=int(12+36*strengths[1]);height=int(10+28*strengths[1])
        px=int(np.clip(points[joint,0]-width/2,0,256-width));py=int(np.clip(points[joint,1]-height/2,0,256-height))
        texture=rng.integers(48,209,(height,width,3),dtype=np.uint8)
        image.paste(Image.fromarray(texture),(px,py))
        covered=(points[:,2]>0)&(points[:,0]>=px)&(points[:,0]<px+width)&(points[:,1]>=py)&(points[:,1]<py+height)
        points[covered,2]=1
        events.append({'axis':'observability','rectangle':[px,py,width,height],'targeted_joint':joint})
    if 4 in axes:
        array=np.array(image);valid=points[:,2]>0;xy=points[valid,:2]
        lower=np.maximum(0,np.floor(xy.min(0)-12)).astype(int);upper=np.minimum(256,np.ceil(xy.max(0)+13)).astype(int)
        mask=np.ones((256,256),bool);mask[lower[1]:upper[1],lower[0]:upper[0]]=False
        # A black crop-padding band is not observed background. Leave it exact.
        mask &= np.any(array!=0,axis=2)
        tint=rng.integers(0,256,3);alpha=.25+.4*strengths[4]
        array[mask]=np.clip((1-alpha)*array[mask]+alpha*tint,0,255).astype(np.uint8)
        image=Image.fromarray(array)
        events.append({'axis':'context','protected_envelope':[lower.tolist(),upper.tolist()],
                       'changed_area_fraction':float(mask.mean()),'limit':'Outside labelled-joint envelope, not a semantic background mask'})
    if 0 in axes:
        side=int(round(256/(1+1.4*strengths[0])))
        image=image.resize((side,side),Image.Resampling.BILINEAR).resize((256,256),Image.Resampling.BILINEAR)
        image=image.filter(ImageFilter.GaussianBlur(.7*strengths[0]))
        events.append({'axis':'information','effective_side':side,'blur_radius':.7*strengths[0]})
    accepted=guard_observability(original_points,points)
    if not accepted:
        return original.copy(),original_points.copy(),{'axes':list(axes),'accepted':False,'events':events,
            'rejection':'Insufficient retained target cues','changed_pixels':0}
    outside=((points[:,:2]<0)|(points[:,:2]>255)).any(1);points[outside,2]=0
    changed=np.array(image)
    return changed,points,{'axes':list(axes),'accepted':True,'events':events,
        'changed_pixels':int(np.any(changed!=original,axis=2).sum()),
        'label_note':'Coordinates transformed only by recorded affine; visibility/crop mask consistent with synthetic intervention.'}


def calibrate_axis_weights(features, losses):
    """Conservative rank-stratified association, not a causal difficulty model."""
    features=np.asarray(features,dtype=float);losses=np.asarray(losses,dtype=float)
    if features.ndim!=2 or features.shape[1]!=5 or len(features)!=len(losses) or len(losses)<100:
        raise ValueError('At least100 aligned source-held-out observations required')
    if not np.isfinite(features).all() or not np.isfinite(losses).all() or np.any((losses<0)|(losses>1)):
        raise ValueError('Invalid profile values')
    rows=[];weights=[];base=float(losses.mean())
    for index,name in enumerate(AXES):
        threshold=float(np.quantile(features[:,index],.75));mask=features[:,index]>=threshold
        if mask.all() and np.any(features[:,index]>threshold):mask=features[:,index]>threshold
        excess=max(0.,float(losses[mask].mean())-base)
        # Retain every axis. Do not let a weak/noisy proxy monopolize synthesis.
        weight=min(2.,1+excess/max(base,.02));weights.append(weight)
        rows.append({'axis':name,'threshold':threshold,'count':int(mask.sum()),
            'mean_loss':float(losses[mask].mean()),'overall_loss':base,'weight':weight})
    return {'weights':weights,'strata':rows,'interpretation':'Source-only marginal association. Correlated proxies; not causality or Public prevalence.'}
