"""Five scratch-trained families, common selection protocol and resumable fits."""
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from data_io import fit_normalization,transform
from model import PrefixChoice,risk_loss
from progress_model import ProgressChoice
from competitive_graph import CompetitiveGraph
from task_model import ResponsibilityTransformer
from common import read,write_json,save_pt,sha,subset,metrics,conditional
from integrity import checked_transfer,model_digest,check_repeat,check_model_unchanged


def prepare(d,train=None,norm=None,family='compact_gru'):
    goal=family!='flat_prefix'
    if norm is None:
        norm=fit_normalization(d,train,goal)
        if 'transformer' in family:
            p=d['prefix'].astype(np.float64);mean=p[train].mean((0,1,2));scale=p[train].std((0,1,2));scale[scale<1e-6]=1.
            norm.update(motion_mean=mean,motion_scale=scale)
    static,motion=transform(d,norm,goal)
    if 'transformer' not in family:return (static,motion),norm
    p=d['prefix'].astype(np.float64)
    raw=torch.from_numpy(((p-norm['motion_mean'])/norm['motion_scale']).clip(-10,10).astype(np.float32))
    xy=torch.from_numpy((p[...,:2]*4.).astype(np.float32));v=torch.from_numpy(p[...,6:8].astype(np.float32))
    centers=torch.from_numpy(np.broadcast_to(d['centers'][None]/1000.,(len(p),2,2)).copy().astype(np.float32))
    return (static,raw,motion,xy,v,centers),norm


class Model(nn.Module):
    def __init__(self,family,norm):
        super().__init__();self.family=family
        if family=='flat_prefix':self.core=PrefixChoice('flat_prefix',prefix_mean=norm['prefix_mean'],prefix_scale=norm['prefix_scale'])
        elif family=='compact_gru':self.core=ProgressChoice('direct_goal')
        elif family=='dynamic_graph':self.core=CompetitiveGraph('dynamic_graph')
        else:self.core=ResponsibilityTransformer('plain_player_transformer' if family=='plain_transformer' else 'task_transformer',32,.15)
    def forward(self,*inputs):return self.core(*inputs)


@torch.inference_mode()
def predict(model,inputs,rows,device):
    model.eval();parts=[];before=model_digest(model)
    for start in range(0,len(rows),128):
        transferred,verify=checked_transfer(tuple(x[rows[start:start+128]] for x in inputs),device)
        probability=model(*transferred).exp();verify()
        if start==0:
            repeat=model(*transferred).exp();verify();check_repeat(probability,repeat,device)
        parts.append(probability.cpu().numpy())
    check_model_unchanged(model,before)
    joint=np.concatenate(parts);prob=np.stack((joint.sum(2),joint.sum(1)),1)
    return conditional(prob),prob,joint


def fit(family,seed,d,train,valid,out,device,epochs=72,lr=.002,wd=.01):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'record.json').exists():
        record=read(out/'record.json')
        if sha(out/'best.pt')!=record['model_sha256']:raise RuntimeError('Changed neural checkpoint')
        return record
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    inputs,norm=prepare(d,train,family=family);np.savez_compressed(out/'normalization.npz',**norm)
    torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=Model(family,norm).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    mask=torch.from_numpy(d['mask'].astype(np.int64));risk=torch.from_numpy(d['risk'].astype(bool))
    history=[];best=float('inf');resume=out/'resume.pt';previous_peak=0
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=True);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];best=state['best'];torch.set_rng_state(state['rng'].cpu())
        previous_peak=state.get('peak_gpu_allocated_bytes',0)
        if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    for epoch in range(len(history)+1,epochs+1):
        model.train();started=time.perf_counter();total=0.;order=np.random.default_rng(seed+893*epoch).permutation(train)
        for start in range(0,len(order),64):
            ids=order[start:start+64];optimizer.zero_grad(set_to_none=True)
            transferred,verify=checked_transfer(tuple(x[ids] for x in inputs)+(mask[ids],risk[ids]),device)
            loss=risk_loss(model(*transferred[:-2]),transferred[-2],transferred[-1]);verify()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())*len(ids)
        c,p,j=predict(model,inputs,valid,device);mean,_=metrics(subset(d,valid),p,j);score=mean['identity_nll']
        if score<best:best=score;save_pt(out/'best.pt',model.state_dict())
        history.append(dict(epoch=epoch,loss=total/len(train),validation_identity_nll=score,seconds=time.perf_counter()-started))
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best=best,rng=torch.get_rng_state())
        state['peak_gpu_allocated_bytes']=max(previous_peak,int(torch.cuda.max_memory_allocated(device))) if device.type=='cuda' else 0
        if device.type=='cuda':state['cuda_rng']=torch.cuda.get_rng_state_all()
        save_pt(resume,state);write_json(out/'history.json',history)
        print(f'{family:20s} seed={seed} {epoch}/{epochs} loss={history[-1]["loss"]:.5f} val={score:.6f} best={best:.6f} {len(train)/history[-1]["seconds"]:.0f}round/s GPU={torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0:.0f}MiB',flush=True)
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True));c,p,j=predict(model,inputs,valid,device)
    mean,series=metrics(subset(d,valid),p,j);np.savez_compressed(out/'validation_choice.npz',conditional=c,prob=p,joint=j,row=d['source_rows'][valid])
    record=dict(family=family,engine='neural',seed=seed,epochs=epochs,lr=lr,wd=wd,parameters=sum(p.numel() for p in model.parameters()),
        model_sha256=sha(out/'best.pt'),normalization_sha256=sha(out/'normalization.npz'),
        best_epoch=min(history,key=lambda r:r['validation_identity_nll'])['epoch'],metrics=mean,series=series,
        seconds=sum(r['seconds'] for r in history),
        read_only_inference_contract='All parameters/buffers unchanged; first actual query batch repeated; all transferred inputs unchanged',
        peak_gpu_allocated_bytes=max(previous_peak,int(torch.cuda.max_memory_allocated(device))) if device.type=='cuda' else None,
        peak_gpu_scope='This fit, including validation; reset before model construction')
    write_json(out/'record.json',record);return record


def infer(family,out,d,device):
    started=time.perf_counter();out=Path(out)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    with np.load(out/'normalization.npz',allow_pickle=False) as z:norm={k:z[k] for k in z.files}
    inputs,_=prepare(d,norm=norm,family=family);model=Model(family,norm).to(device)
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True))
    if device.type=='cuda':torch.cuda.synchronize(device)
    ready=time.perf_counter();c,_,_=predict(model,inputs,np.arange(len(d['x'])),device)
    if device.type=='cuda':torch.cuda.synchronize(device)
    ended=time.perf_counter()
    return c,dict(choice_inference_seconds=np.asarray(ended-ready),choice_end_to_end_seconds=np.asarray(ended-started),
        choice_inference_rounds=np.asarray(len(d['x'])),choice_inference_peak_gpu_bytes=np.asarray(torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0))
