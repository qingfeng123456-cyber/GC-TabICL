"""One-command fixed-development experiment and locked external assessment."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import sys
import time
import traceback
from zipfile import ZipFile,ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
ACTIVE=None


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(run,kind='RESULTS'):
    target=run.parent/f'CS2_Application_{kind}_RETURN.zip';files={}
    for p in run.rglob('*'):
        if p.is_file() and p.suffix!='.zip' and not p.name.endswith('.tmp') and p.name!='RETURN_RECEIPT.json':files['run/'+p.relative_to(run).as_posix()]=p
    for p in HERE.rglob('*'):
        if not p.is_file() or any(x in p.relative_to(HERE).parts for x in ['results','data','pretrained','__pycache__']):continue
        if p.suffix in ['.py','.md','.json','.txt','.cmd','.ps1'] or p.name=='LICENSE':files['code/'+p.relative_to(HERE).as_posix()]=p
    manifest={n:dict(bytes=p.stat().st_size,sha256=digest(p)) for n,p in files.items()}
    with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for n,p in sorted(files.items()):z.write(p,'CS2_Application_Return/'+n)
        z.writestr('CS2_Application_Return/CONTENTS.json',json.dumps(dict(kind=kind,files=manifest),indent=2))
    with ZipFile(target) as z:
        if z.testzip() is not None:raise RuntimeError('Return archive CRC failure')
        for n,meta in manifest.items():
            if hashlib.sha256(z.read('CS2_Application_Return/'+n)).hexdigest()!=meta['sha256']:raise RuntimeError('Return archive hash failure')
    (run/'RETURN_RECEIPT.json').write_text(json.dumps(dict(archive=target.name,bytes=target.stat().st_size,sha256=digest(target),kind=kind),indent=2),'utf-8')
    print('RETURN ARCHIVE:',target,flush=True)


def failure(error):
    run=ACTIVE or HERE/'results/preflight_failure';run.mkdir(parents=True,exist_ok=True)
    previous={}
    if (run/'STATE.json').exists():previous=json.loads((run/'STATE.json').read_text('utf-8'))
    report=dict(status='failed',completed=False,previous_stage=previous.get('stage','preflight'),error=str(error),
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (run/'FAILURE.json').write_text(json.dumps(report,indent=2),'utf-8')
    (run/'TRACEBACK.txt').write_text(traceback.format_exc(),'utf-8')
    (run/'STATE.json').write_text(json.dumps(report,indent=2),'utf-8')
    archive(run,'FAILURE')


def budget_report(run):
    """Keep tuning, selected fits and shared occurrence costs separate."""
    import common as C
    registry=C.read(run/'registry.json')
    choices=[dict(scope=r['scope'],family=r['family'],seed=r['seed'],
                  seconds=r.get('fit',{}).get('seconds'),engine=r['engine'],
                  parameters=r.get('fit',{}).get('parameters'),
                  trainable_parameters=r.get('fit',{}).get('trainable_parameters'))
             for r in registry]
    search=[dict(path=p.relative_to(run).as_posix(),**C.read(p)) for p in (run/'tuning').rglob('record.json')]
    occurrence=[]
    for p in (run/'occurrence').rglob('candidate_depth*.json'):
        record=C.read(p);occurrence.append(dict(path=p.relative_to(run).as_posix(),role='candidate',seconds=record['seconds'],trees=record['trees']))
    for p in (run/'occurrence').rglob('entry_depth*.json'):
        record=C.read(p);occurrence.append(dict(path=p.relative_to(run).as_posix(),role='occurrence',seconds=record['seconds'],trees=record['trees']))
    C.write_json(run/'BUDGET.json',dict(environment=C.read(run/'environment.json'),
        selected_choice_fits=choices,neural_search_fits=search,classical_fits=occurrence,
        selected_choice_seconds=sum(r['seconds'] for r in choices if r['seconds'] is not None),
        neural_search_seconds=sum(r.get('seconds',0.) for r in search),
        classical_seconds=sum(r['seconds'] for r in occurrence),
        upstream_pretraining='Performed by TabICL authors; no time or equal-compute claim is imputed',
        note='CPU/GPU costs and model-search budgets are recorded separately; shared heads are counted once per fitted scope. Rule calibration/linear timing is reported when available.'))


def main():
    global ACTIVE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development',type=Path,default=HERE/'data/development.npz')
    parser.add_argument('--external',type=Path,default=HERE/'data/external_motion.npz')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--methods',nargs='+')
    args=parser.parse_args();run=(args.out or HERE/'results'/('smoke' if args.smoke else 'formal')).resolve();ACTIVE=run;run.mkdir(parents=True,exist_ok=True)
    import numpy as np
    import torch
    import common as C
    import classical
    import neural
    import foundation
    import tabular
    C.HERE=HERE;torch.set_num_threads(2)
    if not args.smoke and args.device=='auto' and not torch.cuda.is_available():
        raise RuntimeError('The complete study needs a working CUDA GPU. Activate a GPU-enabled PyTorch environment compatible with your card; this program does not replace PyTorch. Use --smoke for the CPU setup check.')
    selected=('cpu' if args.device=='auto' else args.device) if args.smoke else ('cuda' if args.device=='auto' else args.device)
    device=torch.device(selected)
    if selected=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable')
    if selected=='cuda':
        try:
            from preflight import check_cuda
            C.write_json(run/'CUDA_PREFLIGHT.json',check_cuda(device))
        except Exception as error:
            raise RuntimeError('CUDA computation or tensor-integrity check failed: '+str(error)+'. Check GPU/driver/PyTorch compatibility in your chosen environment (RTX 5090 needs a Blackwell-compatible build). No environment change was made and study fitting did not start.') from error
    classical._catboost()
    if not args.smoke and not (HERE/'figures.py').exists():raise RuntimeError('Formal run requires figures.py before training')
    if (HERE/'figures.py').exists():__import__('figures')
    __import__('study_figures')
    pretrained=HERE/'pretrained/tabicl-classifier-v2-20260212.ckpt'
    if digest(pretrained)!='bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0':raise RuntimeError('Pretrained checkpoint changed')
    protocol=C.read(HERE/'protocol.json')
    methods=args.methods or (['foundation_hurdle','compact_gru'] if args.smoke else C.CHOICE_METHODS)
    if not set(methods)<=set(C.CHOICE_METHODS):raise ValueError('Unknown method')
    if not args.smoke and methods!=C.CHOICE_METHODS:raise ValueError('Formal run uses the complete registered method list')
    d=C.read_dataset(args.development,development=True)
    if args.smoke:
        tr=list(dict.fromkeys(map(str,d['match'][d['split']=='train'])))[:6]
        va=list(dict.fromkeys(map(str,d['match'][d['split']=='validation'])))[:3]
        d=C.subset(d,np.flatnonzero(np.isin(d['match'].astype(str),tr+va)))
    train=np.flatnonzero(d['split']=='train');valid=np.flatnonzero(d['split']=='validation')
    if not args.smoke and (len(set(d['match'][train])),len(set(d['match'][valid])))!=(59,15):raise ValueError('Unexpected development cohort')
    if set(d['match'][train])&set(d['match'][valid]):raise ValueError('Development series overlap')
    seeds=[2026] if args.smoke else C.SEEDS
    contract=dict(smoke=args.smoke,protocol=protocol,protocol_sha256=C.sha(HERE/'protocol.json'),source_sha256=C.source_hashes(),
        development_sha256=C.sha(args.development),external_sha256=None if args.smoke else C.sha(args.external),
        pretrained_sha256=C.sha(pretrained),methods=methods,seeds=seeds,device=str(device),
        train_source_rows=d['source_rows'][train].tolist(),validation_source_rows=d['source_rows'][valid].tolist(),
        train_series_order=C.order_series(d,train),external_arrays_opened_for_training=False)
    if (run/'contract.json').exists():
        if C.read(run/'contract.json')!=contract:raise RuntimeError('Code, data or configuration changed; use a fresh output directory')
    else:C.write_json(run/'contract.json',contract)
    C.write_json(run/'environment.json',dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__),device=str(device),
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,cuda=torch.version.cuda,
        scratch_precision='float32',foundation_precision='float16_autocast_with_float32_logits' if device.type=='cuda' else 'float32',
        integrity_contract='Device preflight, transferred input equality, repeated first prediction batch, and unchanged inference parameters/buffers'))
    if (run/'LOCK.json').exists():
        C.verify_lock(run)
        if args.smoke:finish_smoke(run,d,valid)
        else:score_external(run,args.external,d,device)
        archive(run,'SMOKE' if args.smoke else 'RESULTS');return
    C.write_json(run/'STATE.json',dict(stage='development_training',completed=False,external_scored=False))
    occurrence_cache={};registry=[];configurations={}

    def occurrence(scope,raw,training):
        target=run/'occurrence'/scope
        if scope not in occurrence_cache:
            classical.fit_all(raw,training,valid,seeds,target,smoke=args.smoke)
            values=[]
            for seed in seeds:
                _,_,diag=classical.predict('catboost_prefix',seed,target,C.subset(raw,valid));values.append(diag['entry'])
            occurrence_cache[scope]=(target,np.mean(values,axis=0))
        return occurrence_cache[scope]

    def choose_recipe(family):
        if family in configurations:return configurations[family]
        path=run/'tuning'/family/'selected_config.json'
        if path.exists():configurations[family]=C.read(path);return configurations[family]
        options=[]
        rates=[.001] if args.smoke else protocol['neural_search']['learning_rates']
        wds=[.01] if args.smoke else protocol['neural_search']['weight_decays']
        for lr in rates:
            for wd in wds:
                target=run/'tuning'/family/f'lr_{lr}_wd_{wd}'
                result=neural.fit(family,2026,d,train,valid,target,device,2 if args.smoke else 40,lr,wd)
                options.append(dict(lr=lr,wd=wd,validation_identity_nll=result['metrics']['identity_nll']))
        selected=dict(min(options,key=lambda x:x['validation_identity_nll']));selected['options']=options;C.write_json(path,selected)
        configurations[family]=selected;return selected

    def train_record(family,seed,scope='full',variant='full',training=None,steps=None):
        training=train if training is None else training
        raw=C.variant(d,variant);target=run/'models'/scope/family/f'seed_{seed}'
        occ,entry=occurrence(scope,raw,training)
        if family in C.NEURAL:
            recipe=choose_recipe(family)
            result=neural.fit(family,seed,raw,training,valid,target,device,2 if args.smoke else 72,recipe['lr'],recipe['wd'])
            with np.load(target/'validation_choice.npz',allow_pickle=False) as z:c=z['conditional']
            engine='neural'
        elif family in ['foundation_hurdle','foundation_frozen']:
            result=foundation.fit(seed,raw,training,valid,target,device,(1 if args.smoke else 100) if steps is None else steps,args.smoke)
            with np.load(target/'validation_choice.npz',allow_pickle=False) as z:c=z['conditional']
            engine='foundation'
        elif family=='linear_choice':
            result=tabular.fit_linear(raw,training,valid,target,args.smoke);c,_=tabular.infer_linear(target,C.subset(raw,valid));engine='linear'
        elif family=='history_prior':
            target.mkdir(parents=True,exist_ok=True);result=dict(deterministic=True,alpha=1.);c=tabular.history_conditional(C.subset(raw,valid));engine='history'
            C.write_json(target/'rule.json',result)
        else:
            kind='constant_velocity' if family=='velocity_rule' else 'catboost_prefix'
            p,_,_=classical.predict(kind,0 if kind=='constant_velocity' else seed,occ,C.subset(raw,valid));c=C.conditional(p);result=dict(kind=kind);engine='classical'
            target.mkdir(parents=True,exist_ok=True);C.write_json(target/'rule.json',result)
        forecast=run/'validation'/scope/family/f'seed_{seed}'/'predictions.npz'
        evidence=C.save_forecast(forecast,C.subset(d,valid),c,entry)
        record=dict(family=family,seed=seed,scope=scope,variant=variant,engine=engine,train_series=len(set(d['match'][training])),
            model_dir=target.relative_to(run).as_posix(),occurrence_dir=occ.relative_to(run).as_posix(),occurrence_seeds=seeds,
            validation_path=forecast.relative_to(run).as_posix(),fit=result,**evidence)
        C.write_json(target/'registry_record.json',record);return record

    for family in methods:
        for seed in ([0] if family in ['velocity_rule','history_prior','linear_choice'] else seeds):
            registry.append(train_record(family,seed));C.write_json(run/'registry_partial.json',registry)
    means={family:float(np.mean([r['metrics']['identity_nll'] for r in registry if r['family']==family and r['scope']=='full'])) for family in methods}
    primary=min(means,key=means.get)
    C.write_json(run/'PRIMARY_SELECTION.json',dict(primary=primary,criterion='Mean validation equal-series conditional identity NLL',methods=means,external_opened=False))
    if not args.smoke:
        registry.append(train_record('foundation_frozen',0,steps=0))
        for v in C.VARIANTS:
            for seed in ([0] if primary in ['velocity_rule','history_prior','linear_choice'] else seeds):registry.append(train_record(primary,seed,scope=v,variant=v))
        for amount in [15,30]:
            training=C.training_rows(d,train,amount)
            for family in dict.fromkeys([primary,'compact_gru','task_transformer']):
                for seed in ([0] if family in ['velocity_rule','history_prior','linear_choice'] else seeds):registry.append(train_record(family,seed,scope=f'train_{amount}',training=training))
    C.write_json(run/'registry.json',registry)
    budget_report(run)
    artifacts={p.relative_to(run).as_posix():C.sha(p) for folder in ['models','occurrence','tuning','validation'] for p in (run/folder).rglob('*') if p.is_file() and not p.name.endswith('.tmp')}
    for name in ['registry.json','PRIMARY_SELECTION.json','BUDGET.json']:artifacts[name]=C.sha(run/name)
    C.write_json(run/'LOCK.json',dict(primary=primary,artifacts=artifacts,source_sha256=C.source_hashes(),contract_sha256=C.sha(run/'contract.json'),
        external_sha256=contract['external_sha256'],all_models_selected_before_external=True))
    C.write_json(run/'STATE.json',dict(stage='locked',completed=False,external_scored=False))
    if args.smoke:finish_smoke(run,d,valid)
    else:score_external(run,args.external,d,device)
    archive(run,'SMOKE' if args.smoke else 'RESULTS')


def record_forecast(run,record,raw,device):
    import numpy as np
    import common as C
    import classical,neural,foundation,tabular
    import torch
    if device.type=='cuda':torch.cuda.synchronize(device)
    started=time.perf_counter()
    d=C.variant(raw,record['variant']);model=run/record['model_dir'];family=record['family']
    if record['engine']=='neural':c,diag=neural.infer(family,model,d,device)
    elif record['engine']=='foundation':c,diag=foundation.infer(model,d,device)
    elif record['engine']=='linear':c,diag=tabular.infer_linear(model,d)
    elif record['engine']=='history':c=tabular.history_conditional(d);diag={}
    else:
        kind='constant_velocity' if family=='velocity_rule' else 'catboost_prefix';p,_,diag=classical.predict(kind,0 if kind=='constant_velocity' else record['seed'],run/record['occurrence_dir'],d);c=C.conditional(p)
    entered=[]
    for seed in record['occurrence_seeds']:
        _,_,entry_diag=classical.predict('catboost_prefix',seed,run/record['occurrence_dir'],d);entered.append(entry_diag['entry'])
    if device.type=='cuda':torch.cuda.synchronize(device)
    diag['hurdle_end_to_end_seconds']=np.asarray(time.perf_counter()-started)
    diag['inference_rounds']=np.asarray(len(d['x']))
    return c,np.mean(entered,axis=0),diag


def finish_smoke(run,d,valid):
    import common as C
    C.verify_lock(run)
    C.write_json(run/'SMOKE_COMPLETED.json',dict(status='completed',full_external_arrays_opened=False,external_scored=False,
        primary=C.read(run/'PRIMARY_SELECTION.json')['primary'],records=len(C.read(run/'registry.json'))))
    C.write_json(run/'STATE.json',dict(stage='smoke_completed',completed=True,external_scored=False))


def score_external(run,path,development,device):
    import numpy as np
    import common as C
    C.verify_lock(run);contract=C.read(run/'contract.json')
    if contract['smoke']:raise RuntimeError('Smoke cannot score external data')
    if C.sha(path)!=contract['external_sha256']:raise RuntimeError('External file fingerprint changed')
    completed=run/'EXTERNAL_ASSESSMENT.json'
    if completed.exists():
        receipt=C.read(completed)
        for name,digest in receipt['output_sha256'].items():
            if C.sha(run/name)!=digest:raise RuntimeError('Completed external output changed')
        C.write_json(run/'STATE.json',dict(stage='completed',completed=True,external_scored=True))
        print('Verified completed external assessment without another model evaluation.',flush=True);return
    C.write_json(run/'STATE.json',dict(stage='external_scoring',completed=False,all_models_locked=True))
    external=C.read_dataset(path,development=False)
    C.write_json(run/'EXTERNAL_ALIGNMENT.json',C.check_external(development,external))
    registry=C.read(run/'registry.json');forecasts={};records=[];files=[]
    for record in registry:
        target=run/'external'/record['scope']/record['family']/f'seed_{record["seed"]}'
        target.mkdir(parents=True,exist_ok=True);forecast=target/'predictions.npz';receipt=target/'RECEIPT.json'
        if receipt.exists():
            saved=C.read(receipt)
            if C.sha(forecast)!=saved['prediction_sha256'] or saved['lock_sha256']!=C.sha(run/'LOCK.json'):raise RuntimeError('Partial external forecast changed')
        else:
            c,entry,diag=record_forecast(run,record,external,device)
            saved=C.save_forecast(forecast,external,c,entry,diag);saved.update(lock_sha256=C.sha(run/'LOCK.json'))
            C.write_json(receipt,saved)
        with np.load(forecast,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['row'],external['source_rows'])
            p,j=z['prob'],z['joint'];diag={k:z[k] for k in z.files if k not in ['row','match','mask','risk','prob','joint','conditional','entry']}
        mean,series=C.metrics(external,p,j);records.append(dict(family=record['family'],seed=record['seed'],scope=record['scope'],metrics=mean,series=series))
        if record['scope']=='full':forecasts.setdefault(record['family'],[]).append(dict(seed=record['seed'],prob=p,joint=j,diagnostics=diag))
        files.extend([forecast,receipt]);print(f'External frozen score {record["scope"]}/{record["family"]}/{record["seed"]}: NLL={mean["identity_nll"]:.6f}',flush=True)
    mechanisms=run/'mechanisms';mechanisms.mkdir(exist_ok=True);np.savez_compressed(mechanisms/'observed_external_inputs.npz',**external)
    primary=C.read(run/'PRIMARY_SELECTION.json')['primary']
    import figures
    manifest=figures.generate(external,forecasts,run/'figures',primary=primary)
    C.write_json(run/'external_summary.json',dict(stage='locked_independent_assessment',primary=primary,records=records,figure_manifest=manifest,external_model_selection=False))
    import study_figures
    manifest['study_figures']=study_figures.from_run(run,run/'figures')
    files.extend([mechanisms/'observed_external_inputs.npz',run/'external_summary.json',run/'EXTERNAL_ALIGNMENT.json'])
    files.extend(p for p in (run/'figures').rglob('*') if p.is_file())
    C.write_json(completed,dict(status='completed',external_series=len(set(external['match'])),external_rounds=len(external['x']),
        primary=primary,output_sha256={p.relative_to(run).as_posix():C.sha(p) for p in files}))
    C.write_json(run/'STATE.json',dict(stage='completed',completed=True,external_scored=True))


if __name__=='__main__':
    try:main()
    except (Exception,KeyboardInterrupt) as error:
        print(traceback.format_exc(),file=sys.stderr,flush=True)
        try:failure(error)
        except Exception:print(traceback.format_exc(),file=sys.stderr,flush=True)
        raise SystemExit(1)
