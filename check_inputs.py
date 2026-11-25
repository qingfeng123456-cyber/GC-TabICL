"""Development-only input-removal and model wiring checks; no external input."""
from pathlib import Path
import argparse
import numpy as np
import torch
import common as C
import classical
import features
import neural


def run(path):
    d=C.read_dataset(path,development=True)
    d=C.subset(d,np.arange(min(12,len(d['x']))))
    report={}
    for variant in C.VARIANTS:
        altered={k:v.copy() for k,v in d.items()}
        if variant=='no_arrival_history':
            altered['x'][...,[13,*range(17,22),*range(25,30)]]+=17
        elif variant=='no_absolute_position':
            altered['x'][...,[*range(2,5),*range(14,17),*range(22,25)]]+=11
            altered['prefix'][...,:3]+=7;altered['spawn_xy']+=12345
        else:
            altered['prefix'][:,:,1:]+=3
            altered['prefix'][:,:,0,3:9]+=5;altered['prefix'][:,:,0,10]+=4
        a,b=C.variant(d,variant),C.variant(altered,variant)
        np.testing.assert_array_equal(features.build(a,np.arange(len(d['x']))),features.build(b,np.arange(len(d['x']))))
        np.testing.assert_array_equal(classical._own_features(a),classical._own_features(b))
        checks={}
        for family in C.NEURAL:
            ia,norm=neural.prepare(a,np.arange(len(d['x'])),family=family)
            ib,_=neural.prepare(b,norm=norm,family=family)
            for x,y in zip(ia,ib):torch.testing.assert_close(x,y,atol=0,rtol=0)
            torch.manual_seed(2026);model=neural.Model(family,norm).eval()
            with torch.inference_mode():result=model(*(v[:2] for v in ia)).exp()
            torch.testing.assert_close(result.sum((1,2)),torch.ones(2),atol=1e-5,rtol=0)
            checks[family]=dict(inputs_identical_after_removed_fields_changed=True,forward_normalized=True)
        report[variant]=dict(foundation_inputs_identical=True,catboost_inputs_identical=True,families=checks)
    return dict(status='passed',external_file_opened=False,checks=report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development',type=Path,default=Path(__file__).parent/'data/development.npz');p.add_argument('--out',type=Path)
    a=p.parse_args();torch.set_num_threads(2);r=run(a.development)
    if a.out:C.write_json(a.out,r)
    print(r)
