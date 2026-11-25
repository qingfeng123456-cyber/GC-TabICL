"""Training sees a physical development file; only final_scorer opens full data."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ARRAY_KEYS=('x','prefix','mask','risk','match','split','source_rows','spawn_xy',
            'uid','round','freeze_end_tick','prefix_tick','prefix_native','prefix_active')
GLOBAL_KEYS=('feature_names','centers','prefix_names','prefix_grid')


def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle,'sha256').hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    temporary.replace(path)


def save_pt(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    torch.save(value,temporary);temporary.replace(path)


def read_dataset(path,development=False):
    with np.load(path,allow_pickle=False) as archive:
        if development and any(k.startswith('future') for k in archive.files):
            raise ValueError('Development release must not contain future-supervision fields')
        result={k:archive[k] for k in ARRAY_KEYS+GLOBAL_KEYS}
    split=result['split'].astype(str)
    if development and not set(split)<={'train','validation','val','dev'}:
        raise ValueError('Test rows forbidden in the physical development file')
    if result['x'].shape!=(len(split),5,30) or result['prefix'].shape!=(len(split),5,5,11):
        raise ValueError('Unexpected current/prefix tensor shape')
    if result['mask'].shape!=result['risk'].shape or result['risk'].shape!=(len(split),2):
        raise ValueError('Unexpected target shape')
    if not np.isfinite(result['x']).all() or not np.isfinite(result['prefix']).all():
        raise ValueError('Nonfinite observed input')
    return result


def subset(d,rows):
    return {k:(v[rows] if k in ARRAY_KEYS else v) for k,v in d.items()}


def raw_goal_features(d):
    raw=d['x'].astype(np.float64);p=d['prefix'].astype(np.float64);centers=d['centers'].astype(np.float64)
    absolute=p[...,:3]*4000.;displacement=p[...,3:6];velocity=p[...,6:9]
    peer_displacement=displacement-displacement.mean(1,keepdims=True)
    peer_velocity=velocity-velocity.mean(1,keepdims=True)
    states=[];motions=[]
    for goal in range(2):
        base=raw[...,:14].copy();base[...,2:4]=(d['spawn_xy'].astype(np.float64)-centers[goal])/1000.
        states.append(np.concatenate((base,raw[...,14+8*goal:22+8*goal],raw[...,14+8*(1-goal):22+8*(1-goal)]),-1))
        position=absolute.copy()/1000.;position[...,:2]-=centers[goal]/1000.
        distance=np.linalg.norm(position[...,:2],axis=-1)
        progress=distance[...,:1]-distance
        radial=-(velocity[...,:2]*position[...,:2]).sum(-1)/np.maximum(distance,1e-6)
        motions.append(np.concatenate((position,displacement,velocity,p[...,9:11],distance[...,None],
                                        progress[...,None],radial[...,None],peer_displacement,peer_velocity),-1))
    return np.stack(states,1),np.stack(motions,1)


def fit_normalization(d,train,goal):
    x,prefix=raw_goal_features(d) if goal else (d['x'].astype(np.float64),d['prefix'].astype(np.float64))
    xaxis=(0,1,2) if goal else (0,1)
    paxis=(0,1,2,3) if goal else (0,1,2)
    mean=x[train].mean(xaxis);scale=x[train].std(xaxis);scale[scale<1e-6]=1.
    pm=prefix[train].mean(paxis);ps=prefix[train].std(paxis);ps[ps<1e-6]=1.
    return dict(x_mean=mean,x_scale=scale,prefix_mean=pm,prefix_scale=ps)


def transform(d,norm,goal):
    x,prefix=raw_goal_features(d) if goal else (d['x'].astype(np.float64),d['prefix'].astype(np.float64))
    x=((x-norm['x_mean'])/norm['x_scale']).clip(-10,10).astype(np.float32)
    prefix=((prefix-norm['prefix_mean'])/norm['prefix_scale']).clip(-10,10).astype(np.float32)
    assert np.isfinite(x).all() and np.isfinite(prefix).all()
    return torch.from_numpy(x),torch.from_numpy(prefix)


def source_hashes(directory): return {p.name:sha(p) for p in sorted(Path(directory).glob('*.py'))}


def verify_cohort_alignment(development,full):
    """Full-data verification occurs only after the training lock exists."""
    ds=development['split'].astype(str);fs=full['split'].astype(str)
    if set(ds)!={'train','validation'} or set(fs)!={'train','validation','test'}:
        raise RuntimeError('Unexpected cohort split names')
    if len(set(map(int,development['source_rows'])))!=len(ds) or len(set(map(int,full['source_rows'])))!=len(fs):
        raise RuntimeError('Repeated source rows in a cohort')
    test=fs=='test';known=~test
    if set(full['match'][test])&set(full['match'][known]): raise RuntimeError('Series crosses development/test boundary')
    if set(development['match'])&set(full['match'][test]): raise RuntimeError('Development includes a test series')
    if set(map(int,development['source_rows']))!=set(map(int,full['source_rows'][known])):
        raise RuntimeError('Full cohort development rows differ from the locked development file')
    lookup={int(row):i for i,row in enumerate(full['source_rows'])}
    aligned=np.asarray([lookup[int(row)] for row in development['source_rows']],dtype=np.int64)
    for key in ARRAY_KEYS:
        np.testing.assert_array_equal(development[key],full[key][aligned],err_msg='Development/full mismatch: '+key)
    for key in GLOBAL_KEYS:
        np.testing.assert_array_equal(development[key],full[key],err_msg='Global data mismatch: '+key)
    for name,count in [('train',59),('validation',15),('test',20)]:
        if len(set(full['match'][fs==name]))!=count: raise RuntimeError('Wrong series count for '+name)
    return dict(status='passed',development_rows=len(ds),test_rows=int(test.sum()),
                row_arrays_equal=list(ARRAY_KEYS),global_arrays_equal=list(GLOBAL_KEYS),
                source_rows_unique=True,series_partitions_disjoint=True)


def verify_lock(directory,run):
    run=Path(run);lock=json.loads((run/'LOCK.json').read_text('utf-8'))
    contract=json.loads((run/'contract.json').read_text('utf-8'))
    if sha(run/'contract.json')!=lock['contract_sha256']: raise RuntimeError('Contract changed after locking')
    if source_hashes(directory)!=lock['source_sha256']: raise RuntimeError('Program source changed after locking')
    for name,digest in lock['artifact_sha256'].items():
        if sha(run/name)!=digest: raise RuntimeError('Locked artifact changed: '+name)
    return lock,contract
