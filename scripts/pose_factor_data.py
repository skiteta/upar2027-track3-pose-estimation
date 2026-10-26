"""Factor augmentation adapter; all inherited base augmentation stays intact."""
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from pose_research_data import ResearchDataset
from train_scratch_cnn import image_path
import train_highcap_robust_native_highres_v16 as native
from upar_pose.factor_synthesis import sample_plan,apply_plan
from upar_pose.pipeline import write_json_atomic
from upar_pose.scratch_cnn import normalize_image


class SynthesisStreams:
    """Expose the old RNG API while checkpointing the separate factor stream."""
    def __init__(self,base,factor):self.base,self.factor=base,factor
    @property
    def bit_generator(self):return self
    @property
    def state(self):
        return {'schema':'factor_streams_v1','base':self.base.bit_generator.state,'factor':self.factor.bit_generator.state}
    @state.setter
    def state(self,state):
        if state.get('schema')=='factor_streams_v1':
            self.base.bit_generator.state=state['base'];self.factor.bit_generator.state=state['factor']
        else:self.base.bit_generator.state=state
    def random(self,*a,**k):return self.base.random(*a,**k)
    def integers(self,*a,**k):return self.base.integers(*a,**k)


def load_donor(item):
    x,y,w,h=item['box']
    with Image.open(image_path(item['file_name'])) as image:
        pixels=np.array(image.convert('RGB').crop((x,y,x+w,y+h)))
    points=item['keypoints'].astype(float).copy();points[:,:2]-=[x,y]
    return pixels,points,item['annotation_id']


class FactorDataset(ResearchDataset):
    def __init__(self,items,recipe,*,settings,weights,output):
        super().__init__(items,recipe)
        self.factor_rng=np.random.default_rng(settings['seed'])
        self.synthetic_rng=SynthesisStreams(self.synthetic_rng,self.factor_rng)
        self.settings,self.axis_weights,self.output=settings,weights,Path(output)
        self.counts=Counter();self.calls=0;self.examples=[]

    def __getitem__(self,index):
        if self.settings.get('mode')=='off':   # knob K3: no factor synthesis at all (no factor RNG draws)
            return super().__getitem__(index)
        original=super().__getitem__(index)
        inputs,targets,weights,coordinates,visibility,ids=original
        plan=sample_plan(self.factor_rng,self.axis_weights,mode=self.settings['mode'])
        # Donor selection uses a plan-local stream: differing sampled axes do
        # not disturb later base crops/pastes/shuffles or next factor draws.
        donor=None
        if 2 in plan['axes']:
            rng=np.random.default_rng(plan['draw_seed']+1);di=int(rng.integers(len(self.items)-1))
            if di>=index:di+=1
            donor=load_donor(self.items[di])
        if plan['axes']:
            pixels=np.rint(np.clip((inputs.numpy().transpose(1,2,0)*.25+.5)*255,0,255)).astype(np.uint8)
            points=np.c_[coordinates.numpy()*255/63,np.where(weights.numpy()>0,visibility.numpy(),0)]
            transformed,keypoints,log=apply_plan(pixels,points,plan,donor)
            if log['accepted']:
                maps,new_weights=native.make_heatmaps_highres(keypoints)
                original=(torch.from_numpy(normalize_image(transformed)),torch.from_numpy(maps),
                    torch.from_numpy(new_weights),torch.from_numpy((keypoints[:,:2]*63/255).astype(np.float32)),
                    torch.from_numpy(keypoints[:,2].astype(np.float32)),ids)
        else:log={'accepted':True,'axes':[],'changed_pixels':0}
        self.calls+=1;self.counts[str(plan['axes'])+(' accepted' if log['accepted'] else ' rejected')]+=1
        if len(self.examples)<24 and plan['axes']:
            self.examples.append({'annotation_id':int(ids[0]),'file_name':self.items[index]['file_name'],'plan':plan,'applied':log})
        if self.calls==1 or self.calls%1024==0:
            write_json_atomic(self.output/'synthesis_observation.json',{'samples':self.calls,'counts':dict(self.counts),
                'first_interventions':self.examples,'source_only':True,'settings':self.settings,
                'limits':'Counts describe the present process; both RNG streams are saved in the optimizer checkpoint.'})
        return original
