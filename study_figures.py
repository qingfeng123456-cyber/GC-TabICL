"""CPU-only ablation, data-amount and compute figures from returned records."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from figures import _style,_save,_write,_name,_color,INK,TEAL,NAVY,AMBER

SCOPES=('no_arrival_history','no_observed_motion','no_absolute_position')
LABELS=('Completed-round history','Observed motion','Absolute-position cues')
METRICS=('identity_nll','identity_accuracy','all_joint_nll','entry_brier')


def _sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _lookup(records):
    result={}
    for row in records:
        key=(row['scope'],row['family'],int(row['seed']))
        if key in result:raise ValueError('Duplicated fit record: '+str(key))
        result[key]=row
    return result


def _ablations(e,out):
    lookup=_lookup(e['records']);primary=e['primary'];rows=[]
    for scope,label in zip(SCOPES,LABELS):
        seeds=sorted(k[2] for k in lookup if k[0]==scope and k[1]==primary and ('full',primary,k[2]) in lookup)
        if not seeds:continue
        values={key:[lookup[(scope,primary,s)]['metrics'][key]-lookup[('full',primary,s)]['metrics'][key] for s in seeds] for key in METRICS}
        rows.append(dict(scope=scope,label=label,seeds=seeds,differences=values))
    if not rows:return None,[], 'No retrained information-ablation records'
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.6),layout='constrained')
    for panel,(ax,key,axis) in enumerate(zip(axes,['identity_nll','all_joint_nll'],['Conditional identity NLL increase (nat)','Joint regional NLL increase (nat)'])):
        ax.axvline(0,color='#99a6ad',lw=1)
        for i,row in enumerate(rows):
            values=np.asarray(row['differences'][key]);ax.plot([values.min(),values.max()],[i,i],color=TEAL)
            ax.scatter(values,np.full(len(values),i),s=24,facecolors='white',edgecolors=TEAL,zorder=3)
            ax.scatter(values.mean(),i,marker='D',s=34,color=TEAL,zorder=4)
        ax.set_yticks(range(len(rows)),[r['label'] for r in rows] if panel==0 else [])
        ax.invert_yaxis();ax.set_xlabel(axis);ax.grid(axis='x',alpha=.6);ax.set_title('('+chr(97+panel)+')',loc='left',fontweight='bold')
    caption=('Retrained input ablations of '+_name(primary)+'. Each point is an ablated-minus-complete difference for a matched fitted seed, using equal-series scores. '
        'Diamonds are means; horizontal spans show the fitted-seed range. Both choice and occurrence heads receive the transformed inputs and are retrained. '
        'Absolute-position ablation preserves relative motion and other regional cues. Positive values favor the complete model. '
        'All predeclared information ablations are shown; ranges describe training-seed variation.')
    return dict(rows=rows,caption=caption),_save(fig,out,'08_retrained_information_ablations'),None


def _scales(e,out):
    lookup=_lookup(e['records']);primary=e['primary'];families=list(dict.fromkeys([primary,'compact_gru','task_transformer']));curves={}
    fig,(ax,bx)=plt.subplots(1,2,figsize=(8.4,3.7),layout='constrained')
    for family in families:
        points=[]
        for amount,scope in [(15,'train_15'),(30,'train_30'),(59,'full')]:
            fits=[r for (s,f,_),r in lookup.items() if s==scope and f==family]
            if not fits:continue
            points.append(dict(series=amount,seeds=[r['seed'] for r in fits],nll=[r['metrics']['identity_nll'] for r in fits],accuracy=[r['metrics']['identity_accuracy'] for r in fits]))
        if len(points)<3:continue
        curves[family]=points
        for axis,key in [(ax,'nll'),(bx,'accuracy')]:
            xs=[p['series'] for p in points];mean=[np.mean(p[key]) for p in points]
            axis.plot(xs,mean,color=_color(family),marker='D',ms=4,label=_name(family))
            for p in points:axis.scatter([p['series']]*len(p[key]),p[key],s=20,facecolors='white',edgecolors=_color(family),zorder=3)
    if not curves:plt.close(fig);return None,[], 'Incomplete nested data-amount records'
    for i,axis in enumerate([ax,bx]):axis.set_xticks([15,30,59]);axis.set_xlabel('Whole training series');axis.grid(alpha=.6);axis.set_title('('+chr(97+i)+')',loc='left',fontweight='bold')
    ax.set_ylabel('Conditional identity NLL');bx.set_ylabel('First-choice accuracy');bx.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(fontsize=8)
    caption=('Training-data scaling on fixed evaluation series. Training subsets contain 15, 30 and 59 whole series in a nested, predeclared hash order. '
        'Diamonds give means and open points give fitted seeds. The 59-series points reuse the complete-model fits. '
        'Architecture settings and hyperparameters are inherited from development selection; each subset refits its preprocessing and models. '
        'The occurrence head uses the same training subset and is shared across the compared choice models within that subset.')
    return dict(curves=curves,caption=caption),_save(fig,out,'09_whole_series_data_scaling'),None


def _budget(e,out):
    rows=[]
    for r in e['registry']:
        if r['scope']!='full':continue
        fit=r.get('fit',{});seconds=fit.get('seconds');params=fit.get('parameters',fit.get('trainable_parameters'))
        if seconds is None:continue
        rows.append(dict(family=r['family'],seed=r['seed'],seconds=float(seconds),parameters=params,
            validation_identity_nll=r['metrics']['identity_nll'],engine=r['engine']))
    if not rows:return None,[], 'No measured fitting times'
    fig,ax=plt.subplots(figsize=(6.4,3.8),layout='constrained')
    for family in dict.fromkeys(r['family'] for r in rows):
        fits=[r for r in rows if r['family']==family]
        x=np.array([r['seconds'] for r in fits]);y=np.array([r['validation_identity_nll'] for r in fits]);keep=x>0
        if not keep.any():continue
        ax.scatter(x[keep],y[keep],color=_color(family),alpha=.5,s=23)
        ax.scatter(x[keep].mean(),y[keep].mean(),color=_color(family),marker='D',s=38,label=_name(family))
    ax.set_xscale('log');ax.set_xlabel('Measured choice-model fitting time (s)');ax.set_ylabel('Development identity NLL');ax.grid(alpha=.5);ax.legend(fontsize=7.8)
    caption=('Observed fitting cost and development log loss. Each small point is one completed full-data fit; diamonds are within-method means. '
        'Times cover the recorded choice-model fitting loops and their validation calls; shared occurrence-head fitting, hyperparameter search and upstream foundation pretraining are tabulated separately in the return records. '
        'CPU and GPU model families use the recorded run environment. The plot describes this run, including its hardware, rather than hardware-independent efficiency. '
        'Only measured times are plotted; unavailable costs are not imputed.')
    return dict(rows=rows,caption=caption),_save(fig,out,'10_measured_fit_cost'),None


def render(e,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);_style();_write(out/'study_evidence.json',e)
    manifest=dict(schema=1,primary=e['primary'],source='saved scalar records; no inference or fitting',figures={})
    captions=[]
    for name,fn in [('08_retrained_information_ablations',_ablations),('09_whole_series_data_scaling',_scales),('10_measured_fit_cost',_budget)]:
        data,files,reason=fn(e,out)
        if data is None:manifest['figures'][name]=dict(status='skipped',reason=reason)
        else:
            _write(out/(name+'.json'),data);manifest['figures'][name]=dict(status='generated',files=files,sha256={f:_sha(out/f) for f in files})
            captions.extend(['## '+name,'',data['caption'],''])
    (out/'study_captions.md').write_text('\n'.join(captions),'utf-8');_write(out/'study_manifest.json',manifest)
    fields=['scope','family','seed',*METRICS]
    with (out/'all_conditions_table.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in e['records']:w.writerow({**{k:r[k] for k in ['scope','family','seed']},**{k:r['metrics'][k] for k in METRICS}})
    return manifest


def from_run(run,out):
    run=Path(run);summary=json.loads((run/'external_summary.json').read_text('utf-8'))
    registry=json.loads((run/'registry.json').read_text('utf-8'))
    evidence=dict(schema=1,primary=summary['primary'],records=summary['records'],registry=registry,
        source_sha256={'external_summary.json':_sha(run/'external_summary.json'),'registry.json':_sha(run/'registry.json')})
    return render(evidence,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();render(json.loads(args.evidence.read_text('utf-8')),args.out)
