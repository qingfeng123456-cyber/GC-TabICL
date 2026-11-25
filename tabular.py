"""Deterministic causal history and shared linear choice controls."""
from pathlib import Path
import time
import numpy as np
import torch
from features import build
from common import read,write_json,sha,combine,metrics,subset


def history_conditional(d):
    counts=np.expm1(np.asarray(d['x'],dtype=np.float64)[...,[21,29]]).transpose(0,2,1)
    counts=np.maximum(counts,0.)+1.
    return counts/counts.sum(-1,keepdims=True)


def fit_linear(d,train,valid,out,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'record.json').exists():return read(out/'record.json')
    started=time.perf_counter()
    own=build(d,np.arange(len(d['x'])))[...,:91]
    labels=(d['mask'][...,None]&(1<<np.arange(5)))>0;unique=(labels.sum(-1)==1)&d['risk']
    tr=own[train][unique[train]];targets=labels[train][unique[train]].argmax(-1)
    mean=tr.mean((0,1));scale=tr.std((0,1));scale[scale<1e-6]=1.
    allx=np.clip((own-mean)/scale,-10,10)
    x=torch.from_numpy(allx[train][unique[train]]).double();y=torch.from_numpy(targets)
    options=[];best=None
    for l2 in ([.01] if smoke else [.0001,.001,.01,.1]):
        w=torch.zeros(91,dtype=torch.float64,requires_grad=True)
        optimizer=torch.optim.LBFGS([w],lr=1.,max_iter=12 if smoke else 300,tolerance_grad=1e-8,line_search_fn='strong_wolfe')
        evaluations=0
        def closure():
            nonlocal evaluations
            evaluations+=1;optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(x@w,y)+l2*w.square().sum()/2
            loss.backward();return loss
        optimizer.step(closure)
        value=w.detach().numpy();logit=allx[valid]@value;logit-=logit.max(-1,keepdims=True)
        c=np.exp(logit);c/=c.sum(-1,keepdims=True)
        p,j=combine(c,np.full((len(valid),2),.5));score=metrics(subset(d,valid),p,j)[0]['identity_nll']
        options.append(dict(l2=l2,validation_identity_nll=score,closure_evaluations=evaluations))
        if best is None or score<best[0]:best=score,l2,value.copy()
    np.savez_compressed(out/'linear.npz',mean=mean,scale=scale,weight=best[2])
    record=dict(family='linear_choice',engine='linear',seed=0,deterministic=True,selected_l2=best[1],
        selection='Equal-series unique identity NLL on validation',options=options,features=91,
        objective='Mean training unique-event CE + L2*sum(w^2)/2',model_sha256=sha(out/'linear.npz'),parameters=91,seconds=time.perf_counter()-started)
    write_json(out/'record.json',record);return record


def infer_linear(out,d):
    with np.load(Path(out)/'linear.npz',allow_pickle=False) as z:
        x=np.clip((build(d,np.arange(len(d['x'])))[...,:91]-z['mean'])/z['scale'],-10,10)
        logits=x@z['weight']
    logits-=logits.max(-1,keepdims=True);c=np.exp(logits);c/=c.sum(-1,keepdims=True)
    return c,{}
