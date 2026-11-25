"""Self-contained causal-prefix CatBoost and constant-velocity controls.

Development fitting and frozen inference are separate APIs. ``fit_all`` never
accepts a test argument. ``predict`` accesses only x, prefix, centers and the
prefix grid; target/future arrays may be absent or deliberately inaccessible.
No dependency is installed automatically and no workspace import is required.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

DEPTHS=(4,6)
RULE_SCALES=(120.,240.,480.,960.)
FEATURE_COUNTS=dict(candidate=275,entry=184,own=91)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    os.replace(temporary,path)


@contextmanager
def native_folder(folder):
    previous=Path.cwd();os.chdir(Path(folder).resolve())
    try:yield
    finally:os.chdir(previous)


def _catboost():
    try:
        import catboost
    except ImportError as error:
        raise RuntimeError('CatBoost 1.2.8 is required. Install the supplied requirements in the selected environment.') from error
    return catboost


def _softmax(values):
    values=np.asarray(values,dtype=np.float64)
    weight=np.maximum(np.exp(values-values.max(-1,keepdims=True)),1e-30)
    return weight/weight.sum(-1,keepdims=True)


def _combine(conditional,entry):
    conditional=np.asarray(conditional,dtype=np.float64)
    conditional=np.maximum(conditional,1e-30)
    conditional=conditional/conditional.sum(-1,keepdims=True)
    entry=np.clip(np.broadcast_to(np.asarray(entry,dtype=np.float64),conditional.shape[:2]),1e-12,1-1e-12)
    prob=np.concatenate(((1-entry)[...,None],entry[...,None]*conditional),-1)
    joint=prob[:,0,:,None]*prob[:,1,None,:]
    if not np.isfinite(prob).all() or not np.isfinite(joint).all():raise ValueError('Invalid classical probabilities')
    np.testing.assert_allclose(prob.sum(-1),1.,atol=1e-10)
    np.testing.assert_allclose(joint.sum((1,2)),1.,atol=1e-10)
    return prob,joint


def _current_inputs(d):
    # Deliberate whitelist: never inspect a future/label array for inference.
    x=np.asarray(d['x'],dtype=np.float64)
    p=np.asarray(d['prefix'],dtype=np.float64)
    centers=np.asarray(d['centers'],dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(5,30):raise ValueError('Expected x[N,5,30]')
    if p.shape!=(len(x),5,5,11):raise ValueError('Expected prefix[N,5,5,11]')
    if centers.shape!=(2,2):raise ValueError('Expected two fixed region centers')
    if not all(np.isfinite(v).all() for v in (x,p,centers)):raise ValueError('Nonfinite observed features')
    return x,p,centers


def _own_features(d):
    x,p,centers=_current_inputs(d)
    xy0=p[:,:,0,:2]*4000.;xyt=p[:,:,-1,:2]*4000.;v=p[:,:,-1,6:8]*250.
    offset=centers[None,:,None]-xyt[:,None]
    current=np.linalg.norm(offset,axis=-1)
    initial=np.linalg.norm(centers[None,:,None]-xy0[:,None],axis=-1)
    radial=(offset*v[:,None]).sum(-1)/np.maximum(current,1e-6)
    difference=current[:,:,:,None]-current[:,:,None,:]
    equals=np.abs(difference)<=1e-5
    rank=(difference>1e-5).sum(-1)+(equals.sum(-1)-1)*.5
    geometry=np.stack((initial/4000.,current/4000.,radial/250.,(initial-current)/1000.,
                       rank/4.,(current-current.min(-1,keepdims=True))/1000.),-1)
    ordinary=np.concatenate((x,p.reshape(len(p),5,-1)),-1)
    own=np.concatenate((np.broadcast_to(ordinary[:,None],(len(p),2,5,85)),geometry),-1)
    assert own.shape==(len(p),2,5,91)
    return own


def candidate_features(d,mean,scale):
    own=((_own_features(d)-mean)/scale).clip(-10,10).astype(np.float32)
    precise=own.astype(np.float64)
    context=np.concatenate((precise.mean(2),precise.std(2)),-1).astype(np.float32)
    region=np.broadcast_to(np.eye(2,dtype=np.float32),(len(own),2,2))
    entry=np.concatenate((context,region),-1)
    candidate=np.concatenate((own,np.broadcast_to(entry[:,:,None],(len(own),2,5,184))),-1)
    assert candidate.shape[-1]==275 and entry.shape[-1]==184
    return np.ascontiguousarray(candidate),np.ascontiguousarray(entry)


def _risk(d):
    r=np.asarray(d['risk'],dtype=bool)
    if r.shape!=(len(d['x']),2):raise ValueError('Expected risk[N,2]')
    return r


def group_scores(d,rows,prob,joint=None):
    """Development scorer; risk masks apply to every target and score."""
    rows=np.asarray(rows,dtype=np.int64)
    masks=np.asarray(d['mask'])[rows]
    risk=_risk(d)[rows]
    labels=(masks[:,:,None]&(1<<np.arange(5)))>0
    unique=(labels.sum(-1)==1)&risk
    truth=labels.argmax(-1)
    cond=prob[:,:,1:]/np.maximum(prob[:,:,1:].sum(-1,keepdims=True),1e-30)
    true=np.take_along_axis(cond,truth[:,:,None],-1)[:,:,0].clip(1e-30)
    compatible=np.concatenate(((masks==0)[:,:,None],labels),-1)
    joint_nll=(-np.log((prob*compatible).sum(-1).clip(1e-30))*risk).sum(-1)
    brier=(1-prob[:,:,0]-(masks!=0))**2
    result={}
    for match in sorted(set(np.asarray(d['match'])[rows])):
        take=np.asarray(d['match'])[rows]==match;u=unique[take];r=risk[take]
        result[str(match)]=dict(rounds=int(take.sum()),risk_regions=int(r.sum()),unique=int(u.sum()),
            identity_nll=float((-np.log(true[take]))[u].mean()) if u.any() else None,
            identity_accuracy=float((cond[take].argmax(-1)==truth[take])[u].mean()) if u.any() else None,
            all_joint_nll=float(joint_nll[take][r.any(-1)].mean()),
            entry_brier=float(brier[take][r].mean()))
    return result


def _mean(scores,key):
    values=[r[key] for r in scores.values() if r[key] is not None]
    if not values or not np.isfinite(values).all():raise ValueError('No finite development '+key)
    return float(np.mean(values))


def _prior(d,train):
    r=_risk(d)[train]
    return (.5+((np.asarray(d['mask'])[train]!=0)&r).sum(0))/(1+r.sum(0))


def _signature(d,train,valid):
    digest=hashlib.sha256()
    # Only training/development values; no file containing held-out values is opened.
    for key in ('x','prefix','mask','risk','match'):
        value=np.ascontiguousarray(np.asarray(d[key])[np.concatenate((train,valid))])
        digest.update(str((key,value.shape,value.dtype)).encode('utf-8'));digest.update(value.tobytes())
    digest.update(np.asarray(d['centers']).tobytes())
    digest.update(np.asarray(train,dtype=np.int64).tobytes());digest.update(np.asarray(valid,dtype=np.int64).tobytes())
    return digest.hexdigest()


def _fit_binary(role,depth,config,tx,ty,vx,vy,folder):
    cb=_catboost();folder.mkdir(parents=True,exist_ok=True)
    model_path=folder/f'{role}_depth{depth}.cbm';meta_path=folder/f'{role}_depth{depth}.json'
    expected=dict(config,role=role,depth=int(depth))
    if model_path.exists() and meta_path.exists():
        saved=json.loads(meta_path.read_text('utf-8'))
        if saved['config']!=expected or saved['model_sha256']!=sha(model_path):raise ValueError('Changed cached classical fit')
        with native_folder(folder):model=cb.CatBoostClassifier().load_model(model_path.name)
        return model,saved
    if set(np.unique(ty))!={0,1} or len(vy)==0:
        raise ValueError('Classical fit needs a training sample of each binary class and nonempty validation rows')
    model=cb.CatBoostClassifier(iterations=config['iterations'],depth=depth,
        learning_rate=.05,l2_leaf_reg=3.,random_seed=config['seed'],task_type='CPU',thread_count=2,
        loss_function='Logloss',eval_metric='Logloss',allow_writing_files=False)
    started=time.perf_counter()
    model.fit(tx,ty,eval_set=(vx,vy),use_best_model=True,early_stopping_rounds=50,verbose=False)
    temporary=model_path.with_suffix('.tmp.cbm')
    with native_folder(folder):model.save_model(temporary.name,format='cbm')
    os.replace(temporary,model_path)
    saved=dict(config=expected,trees=int(model.tree_count_),best_iteration=int(model.get_best_iteration()),
               best_binary_logloss=float(model.get_best_score()['validation']['Logloss']),
               seconds=time.perf_counter()-started,model_sha256=sha(model_path))
    write(meta_path,saved)
    print(f'CatBoost prefix {role} depth={depth} seed={config["seed"]} trees={saved["trees"]}',flush=True)
    return model,saved


def _conditional(model,features):
    logits=np.asarray(model.predict(features.reshape(-1,features.shape[-1]),prediction_type='RawFormulaVal',thread_count=2))
    return _softmax(logits.reshape(features.shape[:3]))


def _entry(model,features):
    return np.asarray(model.predict_proba(features.reshape(-1,features.shape[-1]),thread_count=2)[:,1]).reshape(features.shape[:2])


def _rule_conditional(d,scale,cutoff,projection=8.):
    _,p,centers=_current_inputs(d)
    projected=p[:,:,-1,:2]*4000.+p[:,:,-1,6:8]*250.*(projection-cutoff)
    distance2=((projected[:,None]-centers[None,:,None])**2).sum(-1)
    return _softmax(-distance2/(2*scale**2)),projected


def _save_validation(folder,d,valid,prob,joint):
    rows=np.asarray(d.get('source_rows',np.arange(len(d['x']))))[valid]
    np.savez_compressed(folder/'validation.npz',row=rows,prob=prob,joint=joint)
    scores=group_scores(d,valid,prob,joint)
    return dict(validation_identity_nll=_mean(scores,'identity_nll'),
                validation_identity_accuracy=_mean(scores,'identity_accuracy'),
                validation_joint_nll=_mean(scores,'all_joint_nll'),
                validation_entry_brier=_mean(scores,'entry_brier'),series=scores)


def fit_all(dev,train,valid,seeds,out,smoke=False):
    """Fit on development data only; return JSON-safe selection records.

    The same records are persisted in ``out/selections.json``. Pass ``out`` to
    :func:`predict` after the caller has locked all model/code fingerprints.
    """
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    train=np.asarray(train,dtype=np.int64);valid=np.asarray(valid,dtype=np.int64)
    if not len(train) or not len(valid) or set(train)&set(valid):raise ValueError('Invalid development indices')
    if 'split' in dev and not set(np.asarray(dev['split']).astype(str))<={'train','validation','val','dev'}:
        raise ValueError('fit_all refuses a dataset containing held-out/test rows')
    if set(np.asarray(dev['match'])[train])&set(np.asarray(dev['match'])[valid]):raise ValueError('Series overlap')
    _current_inputs(dev);risk=_risk(dev)
    cb=_catboost();depths=(4,) if smoke else DEPTHS
    config=dict(code_sha256=sha(__file__),development_sha256=_signature(dev,train,valid),
        seeds=list(map(int,seeds)),depths=list(depths),iterations=12 if smoke else 600,
        learning_rate=.05,l2_leaf_reg=3.,thread_count=2,early_stopping_rounds=50,
        task_type='CPU',loss_function='Logloss',catboost_version=cb.__version__,
        stage='development_only',test_scored=False,smoke=bool(smoke),python=platform.python_version(),
        candidate_features=275,entry_features=184,future_features=False)
    contract=out/'contract.json'
    if contract.exists() and json.loads(contract.read_text('utf-8'))!=config:raise ValueError('Changed classical development contract')
    write(contract,config)
    own=_own_features(dev);selected=own[train][risk[train]]
    mean=selected.mean((0,1));scale=selected.std((0,1));scale[scale<1e-6]=1.
    normalization=out/'normalization.npz'
    np.savez_compressed(normalization,mean=mean,scale=scale,centers=np.asarray(dev['centers']),
                        train_rows=np.asarray(dev.get('source_rows',np.arange(len(own))))[train])
    normalization_sha=sha(normalization)
    c,e=candidate_features(dev,mean,scale);prior=_prior(dev,train)
    permutation=np.asarray([3,0,4,1,2]);changed=dict(dev,x=np.asarray(dev['x'])[:,permutation],prefix=np.asarray(dev['prefix'])[:,permutation])
    pc,pe=candidate_features(changed,mean,scale)
    np.testing.assert_allclose(pc,c[:,:,permutation],atol=1e-6);np.testing.assert_allclose(pe,e,atol=1e-6)
    labels=(np.asarray(dev['mask'])[:,:,None]&(1<<np.arange(5)))>0;unique=(labels.sum(-1)==1)&risk
    tx=c[train][unique[train]].reshape(-1,275);ty=labels[train][unique[train]].reshape(-1).astype(np.int64)
    vx=c[valid][unique[valid]].reshape(-1,275);vy=labels[valid][unique[valid]].reshape(-1).astype(np.int64)
    et=e[train][risk[train]];ey=(np.asarray(dev['mask'])[train]!=0)[risk[train]].astype(np.int64)
    ev=e[valid][risk[valid]];evy=(np.asarray(dev['mask'])[valid]!=0)[risk[valid]].astype(np.int64)
    cutoff=float(np.asarray(dev.get('prefix_grid',[0,.5,1,1.5,2]))[-1])
    records=[];folder=out/'constant_velocity';folder.mkdir(exist_ok=True)
    best=None;options=[]
    for scale_value in RULE_SCALES:
        cond,_=_rule_conditional(dev,scale_value,cutoff)
        p,j=_combine(cond[valid],prior);loss=_mean(group_scores(dev,valid,p,j),'identity_nll')
        options.append(dict(scale_world_units=scale_value,validation_identity_nll=loss))
        if best is None or loss<best[0]:best=loss,scale_value,p,j
    record=dict(method='constant_velocity',kind='constant_velocity',seed=0,deterministic=True,
        selected_scale_world_units=best[1],observed_cutoff_seconds=cutoff,projection_seconds=8.,
        candidates=options,entry_prior=prior.tolist(),centers=np.asarray(dev['centers']).tolist(),
        selection_file='constant_velocity/selection.json',validation_file='constant_velocity/validation.npz',
        **_save_validation(folder,dev,valid,best[2],best[3]))
    write(folder/'selection.json',record);records.append(record)
    for seed in seeds:
        folder=out/'catboost_prefix'/f'seed_{int(seed)}';folder.mkdir(parents=True,exist_ok=True)
        best_c=None;best_e=None;options=[];cfg=dict(config,seed=int(seed))
        for depth in depths:
            cm,cs=_fit_binary('candidate',depth,cfg,tx,ty,vx,vy,folder)
            cond=_conditional(cm,c[valid]);p,j=_combine(cond,prior)
            nll=_mean(group_scores(dev,valid,p,j),'identity_nll')
            if best_c is None or nll<best_c[0]:best_c=nll,depth,cm,cs
            em,es=_fit_binary('entry',depth,cfg,et,ey,ev,evy,folder)
            ep=_entry(em,e[valid]);p,j=_combine(np.full((len(valid),2,5),.2),ep)
            brier=_mean(group_scores(dev,valid,p,j),'entry_brier')
            if best_e is None or brier<best_e[0]:best_e=brier,depth,em,es
            options.append(dict(depth=depth,validation_identity_nll=nll,validation_entry_brier=brier,candidate=cs,entry=es))
        p,j=_combine(_conditional(best_c[2],c[valid]),_entry(best_e[2],e[valid]))
        relative=folder.relative_to(out).as_posix()
        record=dict(method='catboost_prefix',kind='catboost_prefix',seed=int(seed),
            selected_candidate_depth=best_c[1],selected_entry_depth=best_e[1],candidates=options,
            normalization_file='normalization.npz',normalization_sha256=normalization_sha,
            candidate_model=f'{relative}/candidate_depth{best_c[1]}.cbm',candidate_model_sha256=best_c[3]['model_sha256'],
            entry_model=f'{relative}/entry_depth{best_e[1]}.cbm',entry_model_sha256=best_e[3]['model_sha256'],
            selection_file=f'{relative}/selection.json',validation_file=f'{relative}/validation.npz',
            observed_cutoff_seconds=cutoff,checks=dict(player_equivariant_features=True,future_features=False),
            **_save_validation(folder,dev,valid,p,j))
        write(folder/'selection.json',record);records.append(record)
        print(f'Classical selected seed={seed} valNLL={record["validation_identity_nll"]:.6f}',flush=True)
    write(out/'selections.json',dict(stage='development_only',test_scored=False,records=records,
                                    code_sha256=config['code_sha256'],contract_sha256=sha(contract)))
    return records


def predict(method,seed,source,raw_d):
    """Predict without opening any label/future values or writing artifacts.

    ``source`` normally is the classical output directory. A mapping with
    ``root`` and ``record`` is also accepted for an already resolved selection.
    The normal directory API verifies each used frozen model/normalizer hash.
    """
    if isinstance(source,dict):
        out=Path(source['root']).resolve();record=source['record']
    else:
        out=Path(source).resolve();selection=json.loads((out/'selections.json').read_text('utf-8'))
        if selection['code_sha256']!=sha(__file__):raise ValueError('Classical inference code changed after selection')
        matches=[r for r in selection['records'] if r['method']==method and int(r['seed'])==int(seed)]
        if len(matches)!=1:raise ValueError('Missing or ambiguous classical selection')
        record=matches[0]
    if record['method']!=method or int(record['seed'])!=int(seed):raise ValueError('Selection identifier mismatch')
    if method=='constant_velocity':
        np.testing.assert_array_equal(np.asarray(raw_d['centers']),np.asarray(record['centers']))
        cond,projected=_rule_conditional(raw_d,record['selected_scale_world_units'],record['observed_cutoff_seconds'],record['projection_seconds'])
        p,j=_combine(cond,np.asarray(record['entry_prior']))
        return p,j,dict(projected_xy=projected,conditional=cond,
                        scale_world_units=np.asarray(record['selected_scale_world_units']))
    if method!='catboost_prefix':raise ValueError('Unknown classical method '+str(method))
    def verified(relative,expected):
        path=(out/relative).resolve()
        if out not in path.parents or sha(path)!=expected:raise ValueError('Changed/outside classical artifact '+relative)
        return path
    norm=verified(record['normalization_file'],record['normalization_sha256'])
    with np.load(norm,allow_pickle=False) as z:
        mean,scale=z['mean'],z['scale']
        np.testing.assert_array_equal(z['centers'],np.asarray(raw_d['centers']))
    c,e=candidate_features(raw_d,mean,scale)
    cb=_catboost();models=[]
    for role in ('candidate','entry'):
        path=verified(record[role+'_model'],record[role+'_model_sha256'])
        with native_folder(path.parent):models.append(cb.CatBoostClassifier().load_model(path.name))
    cond=_conditional(models[0],c);entered=_entry(models[1],e)
    p,j=_combine(cond,entered)
    return p,j,dict(conditional=cond,entry=entered,
                    candidate_depth=np.asarray(record['selected_candidate_depth']),entry_depth=np.asarray(record['selected_entry_depth']))
