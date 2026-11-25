"""Shared files, causal input variants, datasets and probability accounting."""
from pathlib import Path
import hashlib
import json
import numpy as np
from data_io import ARRAY_KEYS,GLOBAL_KEYS,read_dataset,subset,write_json,save_pt,sha
from model import group_scores,mean_metric

HERE=Path(__file__).resolve().parent
SEEDS=[2026,2027,2028]
CHOICE_METHODS=['velocity_rule','history_prior','linear_choice','catboost_prefix',
    'flat_prefix','compact_gru','dynamic_graph','plain_transformer','task_transformer','foundation_hurdle']
NEURAL=['flat_prefix','compact_gru','dynamic_graph','plain_transformer','task_transformer']
VARIANTS=['no_arrival_history','no_observed_motion','no_absolute_position']


def read(path):return json.loads(Path(path).read_text('utf-8'))
def source_hashes():
    return {p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.rglob('*.py'))
            if not any(x in p.relative_to(HERE).parts for x in ['results','__pycache__'])}


def variant(d,name='full'):
    if name=='full':return d
    if name not in VARIANTS:raise ValueError(name)
    result=dict(d);x=d['x'].copy();p=d['prefix'].copy()
    if name=='no_arrival_history':x[...,[13,*range(17,22),*range(25,30)]]=0
    elif name=='no_observed_motion':
        p=np.repeat(p[:,:,:1],p.shape[2],axis=2);p[...,3:9]=0.;p[...,10]=0.
    else:
        x[...,[*range(2,5),*range(14,17),*range(22,25)]]=0;p[...,:3]=0
        # Goal-relative scratch inputs derive coordinates from this auxiliary
        # observed field. Mask it as well, before any family's feature builder.
        result['spawn_xy']=np.zeros_like(d['spawn_xy'])
    result['x']=x;result['prefix']=p
    return result


def metrics(d,prob,joint):
    np.testing.assert_allclose(prob.sum(-1),1.,atol=5e-6)
    np.testing.assert_allclose(joint.sum((1,2)),1.,atol=5e-6)
    series=group_scores(d,np.arange(len(d['x'])),prob,joint)
    return {key:mean_metric(series.values(),key) for key in ['identity_nll','identity_accuracy','all_joint_nll','entry_brier']},series


def combine(conditional,entry):
    c=np.maximum(np.asarray(conditional,dtype=np.float64),1e-30);c=c/c.sum(-1,keepdims=True)
    e=np.clip(np.broadcast_to(np.asarray(entry,dtype=np.float64),c.shape[:2]),1e-12,1-1e-12)
    p=np.concatenate(((1-e)[...,None],e[...,None]*c),-1)
    return p,p[:,0,:,None]*p[:,1,None,:]


def conditional(prob):return prob[...,1:]/prob[...,1:].sum(-1,keepdims=True).clip(1e-30)


def save_forecast(path,d,c,entry,diagnostics=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    p,j=combine(c,entry)
    reserved={'row','match','mask','risk','conditional','entry','prob','joint'}
    extra={k:v for k,v in (diagnostics or {}).items() if k not in reserved}
    np.savez_compressed(path,row=d['source_rows'],match=d['match'],mask=d['mask'],risk=d['risk'],
        conditional=c,entry=entry,prob=p,joint=j,**extra)
    mean,series=metrics(d,p,j)
    return dict(metrics=mean,series=series,prediction_sha256=sha(path))


def order_series(d,train):
    return sorted(set(map(str,d['match'][train])),key=lambda x:hashlib.sha256(('CS2-training-amount-v1:'+x).encode()).hexdigest())


def training_rows(d,train,count):return train[np.isin(d['match'][train].astype(str),order_series(d,train)[:count])]


def check_external(d,external):
    if not set(external['split'].astype(str))<={'test','external'}:raise ValueError('External file must contain only new test/external rows')
    if set(d['match'])&set(external['match']):raise ValueError('External series overlaps development')
    if set(map(int,d['source_rows']))&set(map(int,external['source_rows'])):raise ValueError('External source rows overlap development')
    if len(set(external['source_rows']))!=len(external['source_rows']):raise ValueError('Duplicate external source rows')
    for key in GLOBAL_KEYS:np.testing.assert_array_equal(d[key],external[key])
    return dict(development_rows=len(d['x']),external_rows=len(external['x']),external_series=len(set(external['match'])),series_disjoint=True,source_rows_disjoint=True)


def verify_lock(run):
    lock=read(run/'LOCK.json')
    if lock['source_sha256']!=source_hashes():raise RuntimeError('Source changed after selection lock')
    if lock['contract_sha256']!=sha(run/'contract.json'):raise RuntimeError('Contract changed')
    for relative,digest in lock['artifacts'].items():
        if sha(run/relative)!=digest:raise RuntimeError('Frozen artifact changed: '+relative)
    return lock
