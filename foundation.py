"""Grouped conditional adaptation and label-free inference of the bundled checkpoint."""
from pathlib import Path
import contextlib
import hashlib
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,read,write_json,sha,subset,metrics,combine
from features import build,labelrecords
from integrity import checked_transfer,model_digest,check_repeat,check_model_unchanged
sys.path.insert(0,str(HERE/'vendor'))
from tabicl_raw._model.tabicl import TabICL

def save(path,value):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)

def amp_context(device):
    return torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else contextlib.nullcontext()

def frozen_digest(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def adapter_state(model):
    return {name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}

def restore_adapter(model,state):
    named=dict(model.named_parameters())
    with torch.no_grad():
        for name,value in state.items():named[name].copy_(value.to(named[name].device))

def load_model(checkpoint,device):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=TabICL(**state['config']);model.load_state_dict(state['state_dict'])
    for p in model.parameters():p.requires_grad_(False)
    for module in [model.icl_predictor.tf_icl.blocks[-1],model.icl_predictor.ln,model.icl_predictor.decoder]:
        for p in module.parameters():p.requires_grad_(True)
    # All modules stay in training mode. The eval path uses inference managers
    # and different postprocessing; frozen does not mean switching that path.
    model.train();model.to(device)
    assert state['config']['dropout']==0.0
    return model,state['config']

def episode(bank,rng,support_events=64,query_events=8):
    unique=np.unique(bank['series'])
    query_series=rng.choice(unique,size=query_events,replace=False)
    query=np.asarray([rng.choice(np.flatnonzero(bank['series']==series)) for series in query_series])
    support_pool=np.flatnonzero(~np.isin(bank['series'],query_series))
    support=rng.choice(support_pool,size=support_events,replace=False)
    assert not set(bank['series'][support])&set(bank['series'][query])
    assert not set(bank['round'][support])&set(bank['round'][query])
    assert np.all(bank['y'][support].sum(-1)==1) and np.all(bank['y'][query].sum(-1)==1)
    return support,query

def candidate_logits(model,context_x,context_y,query_x,device):
    """Only support labels enter col/row/ICL; no query label argument exists."""
    count=len(query_x)
    assert len(context_x)==len(context_y) and count%5==0
    (x,y),verify=checked_transfer((torch.from_numpy(np.concatenate((context_x,query_x),axis=0))[None],
        torch.from_numpy(context_y.astype(np.float32))[None]),device)
    with amp_context(device):
        raw=model(x,y,embed_with_test=False)
        assert raw.shape==(1,count,10)
        # Slice active binary classes BEFORE deriving conditional log odds.
        active=raw[..., :2].float()
    verify()
    return (active[...,1]-active[...,0]).reshape(-1,5)


def bank_and_queries(d,train,valid):
    x,y,rounds,goal,player,series=labelrecords(d,train)
    x=x[:,:91].reshape(-1,5,91);y=y.reshape(-1,5)
    mean=x.mean((0,1),dtype=np.float64);scale=x.std((0,1),dtype=np.float64);scale[scale<1e-6]=1.
    x=np.clip((x-mean)/scale,-4,4).astype(np.float32)
    q=build(d,valid)[...,:91];q=np.clip((q-mean)/scale,-4,4).astype(np.float32)
    bank=dict(x=x,y=y.astype(np.float32),round=rounds.reshape(-1,5)[:,0],goal=goal.reshape(-1,5)[:,0],
        series=series.reshape(-1,5)[:,0],winner=y.argmax(-1).astype(np.int64))
    assert np.all(y.sum(-1)==1)
    return bank,q,dict(mean=mean,scale=scale,clip=np.asarray(4.))


@torch.no_grad()
def choices(model,bank,query,device,chunk_events=20):
    model.train()  # fixed pretrained configuration uses dropout=0.
    before=model_digest(model)
    flat=query.reshape(-1,91);parts=[]
    for start in range(0,len(flat),5*chunk_events):
        logits=candidate_logits(model,bank['x'].reshape(-1,91),bank['y'].reshape(-1),flat[start:start+5*chunk_events],device)
        probability=logits.softmax(-1)
        if start==0:
            repeat=candidate_logits(model,bank['x'].reshape(-1,91),bank['y'].reshape(-1),flat[start:start+5*chunk_events],device).softmax(-1)
            check_repeat(probability,repeat,device)
        parts.append(probability.cpu().numpy())
    check_model_unchanged(model,before)
    return np.concatenate(parts).reshape(len(query),2,5)


def fit(seed,d,train,valid,out,device,steps=100,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'record.json').exists():
        record=read(out/'record.json')
        if sha(out/'best.pt')!=record['model_sha256']:raise RuntimeError('Changed foundation adapter')
        return record
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    checkpoint=HERE/'pretrained/tabicl-classifier-v2-20260212.ckpt'
    model,config=load_model(checkpoint,device)
    bank,q,norm=bank_and_queries(d,train,valid)
    np.savez_compressed(out/'normalization.npz',**norm);np.savez_compressed(out/'bank.npz',**bank)
    if len(np.unique(bank['series']))<2:raise ValueError('Foundation needs at least two training series for support/query separation')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    frozen=frozen_digest(model);history=[];episodes=[];best=float('inf');best_step=0;resume=out/'resume.pt'
    started=time.perf_counter();done=0;previous_peak=0;previous_seconds=0.
    def assess(step):
        c=choices(model,bank,q,device,2 if smoke else 20)
        p,j=combine(c,np.full((len(valid),2),.5));mean,series=metrics(subset(d,valid),p,j)
        np.savez_compressed(out/f'validation_step_{step:03d}.npz',conditional=c,prob=p,joint=j,row=d['source_rows'][valid])
        return mean,series
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=True);restore_adapter(model,state['adapter']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        history=state['history'];episodes=state['episodes'];best=state['best'];best_step=state['best_step'];done=state['step']
        previous_peak=state.get('peak_gpu_allocated_bytes',0);previous_seconds=state.get('elapsed_seconds',0.)
    else:
        mean,_=assess(0);best=mean['identity_nll'];history.append(dict(step=0,**mean));save(out/'best.pt',adapter_state(model))
    every=1 if smoke else 20
    for step in range(done+1,steps+1):
        query_count=1 if smoke else min(8,len(np.unique(bank['series']))-1)
        rng=np.random.default_rng(seed+step*7919)
        # Select query series first, so complete event rows and all same-series
        # labels are excluded from support. The standard 59/15 setting is 64/8.
        unique=np.unique(bank['series']);query_series=rng.choice(unique,size=query_count,replace=False)
        query=np.asarray([rng.choice(np.flatnonzero(bank['series']==s)) for s in query_series])
        pool=np.flatnonzero(~np.isin(bank['series'],query_series));support=rng.choice(pool,size=min(2 if smoke else 64,len(pool)),replace=False)
        assert not set(bank['series'][support])&set(bank['series'][query])
        model.train();optimizer.zero_grad(set_to_none=True)
        logits=candidate_logits(model,bank['x'][support].reshape(-1,91),bank['y'][support].reshape(-1),bank['x'][query].reshape(-1,91),device)
        loss=F.cross_entropy(logits,torch.from_numpy(bank['winner'][query]).to(device))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite grouped adaptation loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update()
        episodes.append(dict(step=step,support_rounds=bank['round'][support].tolist(),query_rounds=bank['round'][query].tolist(),
            support_series=bank['series'][support].tolist(),query_series=bank['series'][query].tolist()))
        if step%every==0 or step==steps:
            mean,_=assess(step);history.append(dict(step=step,train_loss=float(loss.detach()),seconds=previous_seconds+time.perf_counter()-started,**mean))
            if mean['identity_nll']<best:best=mean['identity_nll'];best_step=step;save(out/'best.pt',adapter_state(model))
            write_json(out/'history.json',history)
            print(f'foundation seed={seed} {step}/{steps} loss={float(loss.detach()):.5f} val={mean["identity_nll"]:.6f} best={best:.6f} GPU={torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0:.0f}MiB',flush=True)
        save(resume,dict(step=step,adapter=adapter_state(model),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            history=history,episodes=episodes,best=best,best_step=best_step,
            elapsed_seconds=previous_seconds+time.perf_counter()-started,
            peak_gpu_allocated_bytes=max(previous_peak,int(torch.cuda.max_memory_allocated(device))) if device.type=='cuda' else 0))
    assert frozen_digest(model)==frozen,'Frozen pretrained parameters changed'
    restore_adapter(model,torch.load(out/'best.pt',map_location='cpu',weights_only=True))
    c=choices(model,bank,q,device,2 if smoke else 20);p,j=combine(c,np.full((len(valid),2),.5))
    mean,series=metrics(subset(d,valid),p,j);np.savez_compressed(out/'validation_choice.npz',conditional=c,prob=p,joint=j,row=d['source_rows'][valid])
    write_json(out/'episodes.json',episodes);write_json(out/'history.json',history)
    record=dict(family='foundation_hurdle',engine='foundation',seed=seed,steps=steps,best_step=best_step,
        metrics=mean,series=series,model_sha256=sha(out/'best.pt'),bank_sha256=sha(out/'bank.npz'),
        normalization_sha256=sha(out/'normalization.npz'),frozen_parameters_unchanged=True,
        pretrained_sha256=sha(checkpoint),parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        query_chunk_events=2 if smoke else 20,
        read_only_inference_contract='All parameters/buffers unchanged; first actual query batch repeated; all transferred inputs unchanged',
        seconds=previous_seconds+time.perf_counter()-started,training_groups=len(bank['x']),objective='Five-candidate conditional CE',
        peak_gpu_allocated_bytes=max(previous_peak,int(torch.cuda.max_memory_allocated(device))) if device.type=='cuda' else None,
        peak_gpu_scope='This fit, including validation; reset before model construction')
    write_json(out/'record.json',record);return record


def infer(out,d,device):
    started=time.perf_counter();out=Path(out)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    with np.load(out/'bank.npz',allow_pickle=False) as z:bank={k:z[k] for k in z.files}
    with np.load(out/'normalization.npz',allow_pickle=False) as z:mean,scale=z['mean'],z['scale']
    query=build(d,np.arange(len(d['x'])))[...,:91];query=np.clip((query-mean)/scale,-4,4).astype(np.float32)
    model,_=load_model(HERE/'pretrained/tabicl-classifier-v2-20260212.ckpt',device)
    restore_adapter(model,torch.load(out/'best.pt',map_location='cpu',weights_only=True))
    if device.type=='cuda':torch.cuda.synchronize(device)
    ready=time.perf_counter();c=choices(model,bank,query,device,read(out/'record.json')['query_chunk_events'])
    if device.type=='cuda':torch.cuda.synchronize(device)
    ended=time.perf_counter()
    return c,dict(conditional_entropy=-(c*np.log(np.maximum(c,1e-30))).sum(-1),
        choice_inference_seconds=np.asarray(ended-ready),choice_end_to_end_seconds=np.asarray(ended-started),
        choice_inference_rounds=np.asarray(len(d['x'])),choice_inference_peak_gpu_bytes=np.asarray(torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0))
