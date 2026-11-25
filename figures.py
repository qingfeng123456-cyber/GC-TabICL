"""CPU-only publication figures from frozen, same-row forecasts.

generate(d, forecasts, out, primary='foundation_hurdle') saves all numerical evidence,
the lightweight raw arrays needed to redraw, and seven distinct figure families.
redraw('replot_arrays.npz', out) requires numpy/matplotlib only, not training,
checkpoints, DEM recordings, torch, or a network connection.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import textwrap

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter, FuncFormatter

ORDER=('foundation_hurdle','compact_gru','dynamic_graph','task_transformer',
       'plain_transformer','flat_prefix','catboost_prefix','linear_choice','history_prior','velocity_rule')
NAMES={'foundation_hurdle':'Adapted foundation','compact_gru':'Goal-conditioned GRU',
       'dynamic_graph':'Competitive graph','task_transformer':'Structured Transformer',
       'plain_transformer':'Set Transformer','flat_prefix':'Flat prefix',
       'catboost_prefix':'CatBoost','linear_choice':'Linear choice',
       'history_prior':'History prior','velocity_rule':'Constant velocity'}
INK='#21364a';TEAL='#007e79';NAVY='#386c9e';AMBER='#b7781e';GRID='#dde5e9'
COLORS={'foundation_hurdle':TEAL,'compact_gru':NAVY,'dynamic_graph':'#ad6350',
        'task_transformer':'#77618c','plain_transformer':'#9b83aa','flat_prefix':'#6f8d78',
        'catboost_prefix':'#264e70','linear_choice':'#b7781e','history_prior':'#9d8176','velocity_rule':'#89939a'}
PLAYER_COLORS=(TEAL,NAVY,AMBER,'#77618c','#ad6350')
DIVERGING=LinearSegmentedColormap.from_list('gain_teal_copper',['#b86746','#faf9f4','#007e79'])
SEQUENTIAL=LinearSegmentedColormap.from_list('rate_teal',['#f2f6f4','#7eb7af','#006b68'])
DATA_FIELDS=('x','prefix','mask','risk','match','source_rows','centers',
             'feature_names','prefix_names','prefix_grid','round','uid','date','event','split')
BOXES={'A':(400.,1800.,1200.,2200.),'B':(-2200.,-700.,1300.,2450.)}
SITE_BOXES={'A':(700.,1650.,1900.,3200.),'B':(-2200.,-950.,2100.,3350.)}


def _name(method): return NAMES.get(method,method.replace('_',' ').title())
def _color(method): return COLORS.get(method,'#6b7e8b')
def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _ordered(methods): return sorted(methods,key=lambda k:(ORDER.index(k) if k in ORDER else len(ORDER),k))


def _json(value):
    if isinstance(value,dict): return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_json(v) for v in value]
    if isinstance(value,np.ndarray): return _json(value.tolist())
    if isinstance(value,np.generic): return _json(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def _write(path,value):
    Path(path).write_text(json.dumps(_json(value),ensure_ascii=False,indent=2,allow_nan=False),'utf-8')


def _array_sha(array):
    array=np.ascontiguousarray(array)
    header=json.dumps(dict(shape=list(array.shape),dtype=str(array.dtype)),sort_keys=True).encode()
    return hashlib.sha256(header+array.tobytes()).hexdigest()


def _style():
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9.5,'axes.titlesize':10,
        'axes.labelsize':9.5,'xtick.labelsize':8.8,'ytick.labelsize':8.8,'legend.fontsize':8.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,
        'axes.edgecolor':'#81909b','axes.labelcolor':INK,'text.color':INK,
        'grid.color':GRID,'grid.linewidth':.6,'legend.frameon':False,
        'pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})


def _save(fig,out,name):
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',pad_inches=.08,
                metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    return [name+'.pdf',name+'.png']


def _series(values,keep,groups):
    result={}
    for group in sorted(set(groups.astype(str))):
        take=keep&(groups.astype(str)==group)[:,None]
        if take.any(): result[group]=float(values[take].mean())
    return (float(np.mean(list(result.values()))) if result else None),result


def _pairs(primary,other):
    a={str(v['seed']):v for v in primary['fits']}
    b={str(v['seed']):v for v in other['fits']}
    common=sorted(set(a)&set(b),key=str)
    if common: return [(a[k],b[k]) for k in common],'matched_seeds'
    if other['method'] in ['constant_velocity','velocity_rule','history_prior','linear_choice'] and len(b)==1:
        rule=next(iter(b.values()))
        return [(v,rule) for v in primary['fits']],'deterministic_comparator_reused'
    return [],'no_common_fitted_seeds'


def _pair_info(pairs,mode):
    return dict(mode=mode,seed_pairs=[[a['seed'],b['seed']] for a,b in pairs],number_of_pairs=len(pairs))


def _calibration(confidence,correct):
    edges=np.linspace(.2,1.,9)
    bins=np.searchsorted(edges,confidence,side='right').clip(1,len(edges)-1)-1
    result=[]
    for i in range(8):
        take=bins==i
        result.append(dict(n=int(take.sum()),mean_confidence=float(confidence[take].mean()) if take.any() else None,
                           accuracy=float(correct[take].mean()) if take.any() else None))
    ece=sum(v['n']*abs(v['mean_confidence']-v['accuracy']) for v in result if v['n'])/max(len(confidence),1)
    return dict(edges=edges,bins=result,ece=float(ece),forecasts=len(confidence))


def _validate(d,forecasts):
    for key in DATA_FIELDS[:7]:
        if key not in d: raise ValueError('Missing figure input '+key)
    n=len(d['mask'])
    if np.shape(d['x'])!=(n,5,30) or np.shape(d['prefix'])!=(n,5,5,11):
        raise ValueError('Figures require raw 30-feature state and raw 11-channel observed prefix')
    if np.shape(d['risk'])!=(n,2) or np.shape(d['centers'])!=(2,2): raise ValueError('Unexpected figure target schema')
    if len(d['match'])!=n or len(d['source_rows'])!=n: raise ValueError('Figure row identifiers do not align')
    if len(set(map(int,d['source_rows'])))!=n: raise ValueError('Duplicate source rows in scored subset')
    if not np.isfinite(d['x']).all() or not np.isfinite(d['prefix']).all(): raise ValueError('Nonfinite observed figure inputs')
    masks=np.asarray(d['mask'],np.int64);risk=np.asarray(d['risk'],bool)
    if not np.isin(masks,np.arange(32)).all(): raise ValueError('Invalid compatible labels')
    player=(masks[...,None]&(1<<np.arange(5)))>0
    unique=(player.sum(-1)==1)&risk;truth=player.argmax(-1)
    accepted=np.concatenate(((masks==0)[...,None],player),-1)
    accepted=np.where(risk[...,None],accepted,True)
    groups=np.asarray(d['match']).astype(str)
    methods={}
    for method in _ordered(forecasts):
        records=forecasts[method]
        if not records: raise ValueError('No fitted forecasts for '+method)
        fits=[];seen=set()
        for record in records:
            seed=_json(record['seed'])
            if str(seed) in seen: raise ValueError('Duplicate method/seed '+method)
            seen.add(str(seed))
            p=np.asarray(record['prob'],np.float64);joint=np.asarray(record['joint'],np.float64)
            if p.shape!=(n,2,6) or joint.shape!=(n,6,6): raise ValueError('Misaligned forecast '+method)
            if not np.isfinite(p).all() or not np.isfinite(joint).all() or np.any(p<0) or np.any(joint<0):
                raise ValueError('Invalid probability in '+method)
            np.testing.assert_allclose(p.sum(-1),1.,atol=1e-5,rtol=0)
            np.testing.assert_allclose(joint.sum((1,2)),1.,atol=1e-5,rtol=0)
            np.testing.assert_allclose(p,np.stack((joint.sum(2),joint.sum(1)),1),atol=1e-5,rtol=0)
            mass=p[...,1:].sum(-1,keepdims=True)
            if np.any((mass[...,0]<=0)&unique): raise ValueError('Undefined identity probability on an evaluated outcome')
            cond=p[...,1:]/np.maximum(mass,1e-30)
            selected=np.take_along_axis(cond,truth[...,None],-1)[...,0]
            loss=-np.log(selected.clip(1e-30));correct=cond.argmax(-1)==truth
            nll,per_series=_series(loss,unique,groups)
            accuracy,_=_series(correct.astype(float),unique,groups)
            jm=(joint*accepted[:,0,:,None]*accepted[:,1,None,:]).sum((1,2))
            joint_nll,_=_series(-np.log(jm.clip(1e-30))[:,None],risk.any(-1)[:,None],groups)
            brier,_=_series((1-p[...,0]-(masks!=0))**2,risk,groups)
            confidence=cond.max(-1)[unique]
            calibration=_calibration(confidence,correct[unique])
            order=np.argsort(-cond,axis=-1,kind='stable')
            rank=(order==truth[...,None]).argmax(-1)[unique]+1
            rank_counts=np.bincount(rank,minlength=6)[1:6]
            fits.append(dict(seed=seed,prob=p,joint=joint,cond=cond,loss=loss,correct=correct,
                nll=nll,accuracy=accuracy,joint_nll=joint_nll,brier=brier,nll_by_series=per_series,
                calibration=calibration,rank_counts=rank_counts))
        methods[method]=dict(method=method,fits=fits)
    if not methods: raise ValueError('No forecasts supplied')
    counts=dict(rounds=n,series=len(set(groups)),at_risk_regions=int(risk.sum()),
        excluded_preobserved_regions=int((~risk).sum()),unique_region_outcomes=int(unique.sum()),
        tied_region_outcomes=int(((player.sum(-1)>1)&risk).sum()),empty_region_outcomes=int(((masks==0)&risk).sum()),
        no_risk_rounds=int((~risk.any(-1)).sum()),nonfinite_prefix_values=0,
        stale_prefix_observations=int((np.asarray(d['prefix'])[...,10]>1e-7).sum()),
        dead_prefix_observations=int((np.asarray(d['prefix'])[...,9]<.5).sum()),
        unique_by_region=[int(unique[:,g].sum()) for g in range(2)],
        unique_series_by_region=[len(set(groups[unique[:,g]])) for g in range(2)])
    return methods,unique,truth,counts


def _table(methods,counts):
    result=[]
    for method,item in methods.items():
        fits=item['fits'];nll=[v['nll'] for v in fits if v['nll'] is not None]
        avg=lambda key:float(np.mean([v[key] for v in fits if v[key] is not None])) if any(v[key] is not None for v in fits) else None
        result.append(dict(method=method,name=_name(method),n_seeds=len(fits),seeds=[v['seed'] for v in fits],
            n_series=len(fits[0]['nll_by_series']),
            unique_outcomes=counts['unique_region_outcomes'],identity_nll=avg('nll'),
            seed_sd=float(np.std(nll,ddof=1)) if len(nll)>1 else None,
            identity_accuracy=avg('accuracy'),joint_nll=avg('joint_nll'),entry_brier=avg('brier'),
            per_seed=[dict(seed=v['seed'],identity_nll=v['nll'],identity_accuracy=v['accuracy'],
                joint_nll=v['joint_nll'],entry_brier=v['brier'],nll_by_series=v['nll_by_series']) for v in fits]))
    return result


def _support_text(methods):
    return '; '.join(_name(k)+' '+str(len(v['fits']))+' fit(s)' for k,v in methods.items())


def _subset(d):
    labels=sorted(set(str(v).strip().lower() for v in np.asarray(d.get('split',[])).reshape(-1)))
    if labels and set(labels)<= {'val','valid','validation'}: return 'validation',labels
    if labels==['test']: return 'test',labels
    return 'unspecified',labels


def _heatmap(d,methods,unique,primary,out):
    candidates=[k for k in ('compact_gru','dynamic_graph','task_transformer','catboost_prefix') if k in methods and k!=primary]
    if not candidates: candidates=[k for k in methods if k!=primary][:3]
    groups=np.asarray(d['match']).astype(str)
    series=[s for s in sorted(set(groups)) if unique[groups==s].any()]
    if not series: return None,[], 'No unique at-risk observations'
    values=[];pairing={};used=[]
    for candidate in candidates:
        pairs,mode=_pairs(methods[primary],methods[candidate])
        if not pairs: continue
        gains=[float(np.mean([b['nll_by_series'][s]-a['nll_by_series'][s] for a,b in pairs])) for s in series]
        values.append(gains);used.append(candidate);pairing[candidate]=_pair_info(pairs,mode)
    if not used: return None,[], 'No comparator with matched seeds or a reusable deterministic rule'
    matrix=np.asarray(values).T
    matrix=np.vstack((matrix,matrix.mean(0)))
    support=[int(unique[groups==s].sum()) for s in series]
    vmax=max(.005,float(np.abs(matrix).max()))
    fig,ax=plt.subplots(figsize=(8.2,max(3.4,.24*len(matrix)+1.25)),layout='constrained')
    im=ax.imshow(matrix,aspect='auto',cmap=DIVERGING,norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax))
    for i in range(len(matrix)):
        for j in range(len(used)):
            ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=8.7,
                    color='white' if abs(matrix[i,j])>.62*vmax else INK)
    ax.set_xticks(range(len(used)),[textwrap.fill(_name(k),18) for k in used])
    ax.set_yticks(range(len(matrix)),[s+f'  (n={n})' for s,n in zip(series,support)]+['Equal-series mean'])
    ax.axhline(len(series)-.5,color='white',lw=2)
    ax.set_xlabel('Comparator minus '+_name(primary)+' identity NLL')
    cb=fig.colorbar(im,ax=ax,pad=.025,shrink=.78);cb.set_label('Positive favors '+_name(primary)+' (nat)')
    caption=('Per-series identity-loss gains, comparator minus '+_name(primary)+'. '
        'Each cell first pairs the same fitted seed and series, then averages those seed differences. '
        'Only the seed intersection is used for learned comparators; a deterministic rule can be reused across primary seeds. '
        f'There are {len(series)} contributing series; n gives unique at-risk regional outcomes in that series. '
        'The last row weights series equally. Pairing: '+ '; '.join(_name(k)+' '+str(v['seed_pairs']) for k,v in pairing.items())+'.')
    evidence=dict(series=series,support=support,comparators=used,pairing=pairing,gains=matrix,caption=caption)
    return evidence,_save(fig,out,'01_series_gains'),None


def _rank(values,descending=False):
    values=-values if descending else values
    return 1+(values[...,None]>values[...,None,:]).sum(-1)


def _profiles(d,methods,unique,truth,primary,comparator,out):
    raw=np.asarray(d['x']);prefix=np.asarray(d['prefix']);centers=np.asarray(d['centers'])
    habit=np.stack((raw[:,:,17],raw[:,:,25]),1)
    previous=np.stack((raw[:,0,21],raw[:,0,29]),1)
    historical=previous>1e-7
    position=prefix[:,:,-1,:2]*4000.;velocity=prefix[:,:,-1,6:8]*250.
    relative=position[:,None]-centers[None,:,None]
    radial=-(velocity[:,None]*relative).sum(-1)/np.maximum(np.linalg.norm(relative,axis=-1),1e-9)
    hrank=_rank(habit,descending=True);vrank=_rank(radial,descending=True)
    eligible=unique&historical
    counts=np.zeros((5,5),int);observed=np.full((5,5),np.nan)
    fits_primary=methods[primary]['fits'];fits_other=None;pairing=None
    if comparator:
        pairs,mode=_pairs(methods[primary],methods[comparator])
        if pairs:
            fits_primary=[a for a,b in pairs];fits_other=[b for a,b in pairs]
            pairing=_pair_info(pairs,mode)
    probabilities={primary:np.mean([v['cond'] for v in fits_primary],0)}
    if fits_other: probabilities[comparator]=np.mean([v['cond'] for v in fits_other],0)
    biases={k:np.full((5,5),np.nan) for k in probabilities}
    candidate_truth=truth[...,None]==np.arange(5)
    for i in range(5):
        for j in range(5):
            take=eligible[...,None]&(hrank==i+1)&(vrank==j+1)
            counts[i,j]=int(take.sum())
            if take.any():
                observed[i,j]=candidate_truth[take].mean()
                for k,p in probabilities.items(): biases[k][i,j]=p[take].mean()-observed[i,j]
    if not counts.sum(): return None,[], 'No unique outcomes with earlier region-arrival history'
    keys=list(biases)
    arrays=[observed]+[biases[k] for k in keys]
    titles=['Observed first-arriver frequency']+[_name(k)+' bias' for k in keys]
    if len(arrays)==2:
        arrays.append(counts.astype(float));titles.append('Candidate opportunities')
    vmax=max(.01,max(float(np.nanmax(np.abs(v))) for v in biases.values()))
    fig,axes=plt.subplots(1,3,figsize=(8.4,3.5),layout='constrained')
    for panel,(ax,values,title) in enumerate(zip(axes,arrays,titles)):
        missing=counts==0
        if panel==0: cmap=SEQUENTIAL.copy();norm=plt.Normalize(0,1)
        elif panel<=len(keys): cmap=DIVERGING.copy();norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
        else: cmap=SEQUENTIAL.copy();norm=plt.Normalize(0,max(counts.max(),1))
        cmap.set_bad('#e5e9ec')
        im=ax.imshow(np.ma.array(values,mask=missing),cmap=cmap,norm=norm)
        ax.set_title('('+chr(97+panel)+')',loc='left',fontsize=9.3,fontweight='bold')
        for i in range(5):
            for j in range(5):
                if not counts[i,j]: text='—'
                elif panel==0: text=f'{100*values[i,j]:.0f}%\nn={counts[i,j]}'
                elif panel<=len(keys): text=f'{100*values[i,j]:+.1f}\nn={counts[i,j]}'
                else: text=str(counts[i,j])
                ax.text(j,i,text,ha='center',va='center',fontsize=8.4,
                        color='white' if counts[i,j] and ((panel==0 and values[i,j]>.65) or (0<panel<=len(keys) and abs(values[i,j])>.7*vmax)) else INK)
        ax.set_xticks(range(5),range(1,6));ax.set_yticks(range(5),range(1,6))
        ax.set_xlabel('Current approach-speed rank')
        if panel==0: ax.set_ylabel('Historical first-arrival rank')
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.12,shrink=.88)
        cb.set_label('First-arriver frequency' if panel==0 else ('Bias (percentage points)' if panel<=len(keys) else 'Opportunities'),fontsize=8.6)
        if panel==0: cb.ax.xaxis.set_major_formatter(PercentFormatter(1))
        elif panel<=len(keys): cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{100*value:.0f}'))
    caption=('Panels: '+ '; '.join('('+chr(97+i)+') '+title for i,title in enumerate(titles))+'. '
        f'History and approach-speed profiles use {int(eligible.sum())} unique at-risk region outcomes with earlier unique-arrival history, '
        f'giving {int(counts.sum())} five-player candidate opportunities. Rank 1 means highest historical count or fastest radial approach toward the fixed region center; ties share their best rank. '
        'Frequencies are conditioned on a unique regional arrival and measure which candidate arrives first; they do not measure regional-entry incidence. '
        'Bias is predicted conditional probability minus observed frequency, in percentage points in cell annotations. Counts are candidate opportunities; empty cells are gray. '
        'Probabilities average fitted conditional distributions, not an inference ensemble. Comparison pairing: '+str(pairing)+'.')
    evidence=dict(region_outcomes=int(eligible.sum()),opportunities=int(counts.sum()),counts=counts,observed=observed,
        biases=biases,pairing=pairing,excluded_no_history=int((unique&~historical).sum()),
        dead_at_cutoff_opportunities=int((eligible[...,None]&(prefix[:,:,-1,9][:,None]<.5)).sum()),caption=caption)
    return evidence,_save(fig,out,'02_history_motion_profiles'),None


def _reliability(methods,unique,out):
    if not unique.any(): return None,[], 'No unique at-risk observations'
    fig=plt.figure(figsize=(9.0,max(4.8,2.8+.32*len(methods))))
    grid=fig.add_gridspec(1,3,width_ratios=[1.15,.60,1])
    ax=fig.add_subplot(grid[0,0]);bx=fig.add_subplot(grid[0,2])
    fig.subplots_adjust(left=.09,right=.985,bottom=.31,top=.90,wspace=.03)
    ax.plot([.2,1],[.2,1],'--',color='#9aa6af',lw=1,label='Perfect calibration')
    evidence={};fractions=[]
    markers=['o','s','^','D','v','P','X','h']
    for i,(method,item) in enumerate(methods.items()):
        fits=item['fits']
        confidence=np.concatenate([v['cond'].max(-1)[unique] for v in fits])
        correct=np.concatenate([v['correct'][unique] for v in fits])
        record=_calibration(confidence,correct);evidence[method]=record
        supported=[v for v in record['bins'] if v['n']]
        ax.plot([v['mean_confidence'] for v in supported],[v['accuracy'] for v in supported],
            color=_color(method),marker=markers[i%len(markers)],ms=4,lw=1.3,label=_name(method)+f' ({record["ece"]:.3f})')
        fractions.append([v['n']/max(record['forecasts'],1) for v in record['bins']])
    ax.set(xlim=(.18,1.02),ylim=(.0,1.02),xlabel='Mean top-choice probability',ylabel='Observed top-choice accuracy')
    ax.grid(alpha=.7);ax.set_title('(a)',loc='left',fontweight='bold')
    fractions=np.asarray(fractions);masked=np.ma.masked_where(fractions==0,fractions)
    cmap=SEQUENTIAL.copy();cmap.set_bad('#e5e9ec')
    bx.imshow(masked,cmap=cmap,vmin=0,vmax=max(.01,float(fractions.max())),aspect='auto')
    for i in range(len(fractions)):
        for j in range(8):
            bx.text(j,i,'—' if fractions[i,j]==0 else f'{100*fractions[i,j]:.0f}',ha='center',va='center',fontsize=8.2,
                color='white' if fractions[i,j]>.65*fractions.max() else INK)
    bx.set_yticks(range(len(methods)),[textwrap.fill(_name(k),20) for k in methods]);bx.tick_params(axis='y',labelsize=8.3)
    bx.set_xticks(range(8),['.2–.3','.3–.4','.4–.5','.5–.6','.6–.7','.7–.8','.8–.9','.9–1'],rotation=45,ha='right')
    bx.set_xlabel('Top-choice probability bin');bx.set_title('(b)',loc='left',fontweight='bold')
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.005),fontsize=8)
    caption=('(a) Reliability and pooled ECE. (b) Forecast support by confidence bin. '
        f'All supplied methods are shown on {int(unique.sum())} unique at-risk regional outcomes. Eight bins span confidence 0.2–1.0. '
        'ECE is the forecast-count-weighted absolute confidence–accuracy gap. Each fitted seed contributes another forecast of the same outcomes, not independent matches. '
        'The right panel reports within-method forecast percentages; gray bins have zero support. Fit counts: '+_support_text(methods)+'.')
    return dict(methods=evidence,support_fractions=fractions,caption=caption),_save(fig,out,'03_reliability_support'),None


def _shortlists(methods,unique,out):
    n=int(unique.sum())
    if not n: return None,[], 'No unique at-risk observations'
    fig=plt.figure(figsize=(9.0,max(4.8,2.8+.32*len(methods))))
    grid=fig.add_gridspec(1,3,width_ratios=[1.15,.60,1])
    ax=fig.add_subplot(grid[0,0]);bx=fig.add_subplot(grid[0,2])
    fig.subplots_adjust(left=.09,right=.985,bottom=.31,top=.90,wspace=.03)
    evidence={};ranks=[]
    for method,item in methods.items():
        counts=np.asarray([v['rank_counts'] for v in item['fits']])
        distribution=counts.mean(0)/n;coverage=np.cumsum(distribution)
        evidence[method]=dict(per_seed_rank_counts=counts,rank_distribution=distribution,coverage=coverage,case_denominator=n)
        ranks.append(distribution)
        ax.plot(range(1,6),coverage,marker='o',ms=4,lw=2 if method=='foundation_hurdle' else 1.2,color=_color(method),label=_name(method))
    ax.set(xlim=(.85,5.15),ylim=(0,1.02),xticks=range(1,6),xlabel='Players retained in the shortlist',ylabel='Observed entrant included')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid();ax.set_title('(a)',loc='left',fontweight='bold')
    ranks=np.asarray(ranks);bx.imshow(ranks,cmap=SEQUENTIAL,vmin=0,vmax=max(float(ranks.max()),.01),aspect='auto')
    for i in range(len(ranks)):
        for j in range(5): bx.text(j,i,f'{100*ranks[i,j]:.1f}',ha='center',va='center',fontsize=8.5,color='white' if ranks[i,j]>.65*ranks.max() else INK)
    bx.set_yticks(range(len(methods)),[textwrap.fill(_name(k),20) for k in methods]);bx.tick_params(axis='y',labelsize=8.3)
    bx.set_xticks(range(5),range(1,6));bx.set_xlabel('Rank of the observed entrant')
    bx.set_title('(b)',loc='left',fontweight='bold')
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.005),fontsize=8)
    caption=('(a) Coverage by analyst shortlist budget. (b) Entrant-rank distribution (%). '
        f'All supplied methods rank the same {n} unique at-risk regional outcomes. Counts first pool cases within each fitted seed, '
        'then give fitted seeds equal weight. Stable roster-slot order resolves probability ties. This pooled coverage differs from the primary equal-series accuracy. '
        'Fit counts: '+_support_text(methods)+'.')
    return dict(methods=evidence,caption=caption),_save(fig,out,'04_candidate_shortlists'),None


def _corrections(d,methods,unique,primary,comparator,out):
    if not comparator: return None,[], 'No comparator forecasts'
    pairs,mode=_pairs(methods[primary],methods[comparator])
    if not pairs or not unique.any(): return None,[], 'No paired seeds or unique outcomes'
    transitions=[]
    for a,b in pairs:
        matrix=np.zeros((2,2),int)
        for before in range(2):
            for after in range(2): matrix[before,after]=int((unique&(b['correct']==bool(before))&(a['correct']==bool(after))).sum())
        transitions.append(matrix)
    transitions=np.asarray(transitions);average=transitions.mean(0)
    raw=np.asarray(d['x']);groups=np.asarray(d['match']).astype(str)
    support=np.rint(np.expm1(np.stack((raw[:,0,21],raw[:,0,29]),1))).clip(0,None)
    categories=[support==0,(support>=1)&(support<=2),(support>=3)&(support<=5),support>=6]
    names=['No prior arrivals','1–2 prior arrivals','3–5 prior arrivals','6+ prior arrivals']
    records=[]
    for label,category in zip(names,categories):
        keep=unique&category
        values=[]
        for a,b in pairs:
            mean,per_series=_series(b['loss']-a['loss'],keep,groups)
            values.append(dict(gain=mean,series=per_series))
        scores=[v['gain'] for v in values if v['gain'] is not None]
        records.append(dict(label=label,n=int(keep.sum()),series=len(set(groups[keep.any(-1)])),
                            mean_gain=float(np.mean(scores)) if scores else None,per_pair=values))
    fig,(ax,bx)=plt.subplots(1,2,figsize=(8.4,3.9),gridspec_kw={'width_ratios':[1,1.35]},layout='constrained')
    cell_colors=[['#e9edf0','#d4ece6'],['#f2dfd0','#dbe6ef']]
    cell_names=[['Both incorrect','Corrected'],['New error','Both correct']]
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=cell_colors[i][j],edgecolor='white',lw=2))
            ax.text(j,i-.16,cell_names[i][j],ha='center',va='center',fontsize=9)
            ax.text(j,i+.13,f'{average[i,j]:.1f}',ha='center',va='center',fontsize=16,fontweight='bold',color=TEAL if (i,j)==(0,1) else (AMBER if (i,j)==(1,0) else INK))
    ax.set(xlim=(-.5,1.5),ylim=(1.5,-.5),xticks=[0,1],yticks=[0,1],
           xlabel=_name(primary)+' first choice',ylabel=_name(comparator)+' first choice')
    ax.set_xticklabels(['Incorrect','Correct']);ax.set_yticklabels(['Incorrect','Correct'])
    ax.set_title('(a)',loc='left',fontweight='bold');ax.set_aspect('equal')
    bx.axvline(0,color='#9aa6af',lw=1)
    for i,record in enumerate(records):
        values=[v['gain'] for v in record['per_pair'] if v['gain'] is not None]
        if values:
            bx.plot([min(values),max(values)],[i,i],color=TEAL,lw=1.5)
            bx.scatter(values,[i]*len(values),s=22,facecolors='white',edgecolors=TEAL,zorder=3)
            bx.scatter(record['mean_gain'],i,marker='D',s=32,color=TEAL,zorder=4)
        else: bx.text(0,i,'No observations',va='center',ha='center',color='#7c8891',fontsize=8.5)
    bx.set_yticks(range(4),[v['label']+f'\nn={v["n"]}, series={v["series"]}' for v in records]);bx.invert_yaxis()
    bx.set_xlabel('Comparator minus selected-model\nidentity NLL (nat)',labelpad=7)
    bx.grid(axis='x');bx.set_title('(b)',loc='left',fontweight='bold')
    caption=('(a) Matched first-choice changes. (b) Gain by completed-history support. '
        f'{_name(primary)} versus {_name(comparator)} uses matched forecast pairs {_pair_info(pairs,mode)["seed_pairs"]}. '
        f'The left cells average case counts across fitted pairs on the same {int(unique.sum())} unique outcomes; they sum to that case denominator. '
        'The right panel computes identity-loss differences within each contributing series, then weights series and paired seeds equally. '
        'Open markers are fitted-pair scores, diamonds their means and ranges their fitted-pair spans, not confidence intervals. '
        'Historical support counts earlier unique regional arrivals. These are separately trained model comparisons, not causal interventions.')
    evidence=dict(comparator=comparator,pairing=_pair_info(pairs,mode),transition_counts=transitions,
                  mean_transition_counts=average,history_groups=records,caption=caption)
    return evidence,_save(fig,out,'05_forecast_corrections'),None


def _draw_trajectory(ax,xy,goal,centers):
    for key in ('A','B'):
        x0,x1,y0,y1=BOXES[key]
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=TEAL if key==goal else '#b5c2cd',
            alpha=.13 if key==goal else .07,edgecolor=TEAL if key==goal else '#8e9ca7',lw=1))
        ax.text((x0+x1)/2,y0+30,key+' staging',ha='center',va='bottom',fontsize=8.2)
        sx0,sx1,sy0,sy1=SITE_BOXES[key]
        ax.add_patch(Rectangle((sx0,sy0),sx1-sx0,sy1-sy0,fill=False,edgecolor='#9da7ae',ls=':',lw=.8))
    ax.scatter(centers[:,0],centers[:,1],marker='+',s=45,color=TEAL,zorder=4)
    for player,color in enumerate(PLAYER_COLORS):
        points=xy[player]
        ax.plot(points[:,0],points[:,1],'-o',ms=2.8,lw=1.1,color=color)
        ax.annotate('',xy=points[-1],xytext=points[-2],arrowprops=dict(arrowstyle='->',lw=1.2,color=color))
    all_x=np.r_[xy[...,0].ravel(),-2250,1850];all_y=np.r_[xy[...,1].ravel(),1100,3400]
    ax.set_xlim(all_x.min()-120,all_x.max()+120);ax.set_ylim(all_y.min()-150,all_y.max()+150)
    ax.set_aspect('equal',adjustable='box');ax.set_xlabel('World x');ax.set_ylabel('World y')
    # The inset is placed opposite the highlighted goal; it shows actual prefix
    # motion at a readable scale, never an imagined route through the map.
    inset=ax.inset_axes([.03 if goal=='A' else .58,.58,.39,.35])
    for player,color in enumerate(PLAYER_COLORS):
        points=xy[player]
        inset.plot(points[:,0],points[:,1],'-o',ms=2.6,lw=1.1,color=color)
        inset.annotate('',xy=points[-1],xytext=points[-2],arrowprops=dict(arrowstyle='->',color=color,lw=1.1))
        inset.text(points[-1,0]+7,points[-1,1]+7,'P'+str(player+1),fontsize=8,color=color)
    span_x=max(float(np.ptp(xy[...,0])),120.);span_y=max(float(np.ptp(xy[...,1])),120.)
    inset.set_xlim(xy[...,0].min()-.2*span_x,xy[...,0].max()+.25*span_x)
    inset.set_ylim(xy[...,1].min()-.2*span_y,xy[...,1].max()+.25*span_y)
    inset.set_xticks([]);inset.set_yticks([]);inset.set_title('0–2 s prefix zoom',fontsize=8.2)
    inset.set_facecolor('white')


def _cases(d,methods,unique,truth,primary,comparator,out):
    rows=np.asarray(d['source_rows'],np.int64);groups=np.asarray(d['match']).astype(str)
    xy=np.asarray(d['prefix'])[...,:2]*4000.;centers=np.asarray(d['centers'])
    pairs=[];mode='primary_fits_only'
    if comparator: pairs,mode=_pairs(methods[primary],methods[comparator])
    primary_fits=[a for a,b in pairs] if pairs else methods[primary]['fits']
    other_fits=[b for a,b in pairs] if pairs else []
    evidence=[];files=[];skipped=[]
    for g,goal in enumerate(('A','B')):
        available=np.flatnonzero(unique[:,g])
        chosen=sorted(available,key=lambda i:int(rows[i]))[:2]
        if not chosen: skipped.append(goal+': no unique at-risk observations');continue
        fig,axes=plt.subplots(len(chosen),2,figsize=(8.4,2.9*len(chosen)),squeeze=False,
                              gridspec_kw={'width_ratios':[1.2,1]},layout='constrained')
        for panel,row in enumerate(chosen):
            ax,bx=axes[panel]
            _draw_trajectory(ax,xy[row],goal,centers)
            winner=int(truth[row,g]);source=int(rows[row])
            title=f'{goal} · series {groups[row]} · source row {source}'
            ax.set_title(title,loc='left',fontsize=9.1,fontweight='bold')
            full=np.asarray([v['cond'][row,g] for v in primary_fits])
            other=np.asarray([v['cond'][row,g] for v in other_fits]) if other_fits else None
            y=np.arange(5);height=.32 if other is not None else .55
            bx.barh(y-.18 if other is not None else y,full.mean(0),height=height,color=TEAL,label=_name(primary))
            for values in full: bx.plot(values,y-.18 if other is not None else y,'|',color=INK,ms=6,alpha=.7)
            if other is not None:
                bx.barh(y+.18,other.mean(0),height=height,color=AMBER,alpha=.85,label=_name(comparator))
                for values in other: bx.plot(values,y+.18,'|',color='#72501e',ms=6,alpha=.7)
            bx.set_yticks(y,['P'+str(i+1)+(r' $\star$' if i==winner else '') for i in range(5)])
            bx.invert_yaxis();bx.set_xlim(0,1);bx.xaxis.set_major_formatter(PercentFormatter(1))
            bx.set_xlabel('Conditional first-arriver probability');bx.grid(axis='x')
            bx.set_title('Observed entrant: P'+str(winner+1),loc='left',fontweight='bold')
            bx.legend(loc='lower right',fontsize=8.2)
            evidence.append(dict(goal=goal,row=int(row),source_row=source,series=groups[row],winner=winner,
                prefix_xy=xy[row],primary_probabilities=full,comparator=comparator if other is not None else None,
                comparator_probabilities=other,pairing=_pair_info(pairs,mode) if pairs else dict(mode=mode,seeds=[v['seed'] for v in primary_fits])))
        files+=_save(fig,out,'06_cases_'+goal)
    if not files: return None,[], '; '.join(skipped)
    caption=('Cases are selected before inspecting forecast quality: for each region, take up to the first two unique at-risk observations in ascending original source-row order. '
        'Solid arrows show the five observed 0–2 s trajectories, including a prefix zoom; they are not future paths or navigation routes. '
        'Shaded staging rectangles and dotted site-priority boxes use the fixed Dust2 geometry, and crosses mark region centers. '
        'Bars average conditional distributions over matched fitted seeds when a comparator is available; short ticks show those fits. The star marks the observed entrant. '
        'These examples are not selected wins. Missing case groups: '+str(skipped)+'.')
    return dict(selection='first two per region by original source row; no prediction-quality filter',cases=evidence,
                geometry=dict(staging_boxes=BOXES,site_priority_boxes=SITE_BOXES),skipped=skipped,caption=caption),files,None


def _tradeoff(d,methods,unique,primary,comparator,out):
    if not unique.any() or not comparator:return None,[], 'No comparable unique outcomes'
    pairs,mode=_pairs(methods[primary],methods[comparator])
    if not pairs:return None,[], 'No paired fitted seeds'
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9.0,4.2),layout='constrained',gridspec_kw={'width_ratios':[1.1,1]})
    for name,item in methods.items():
        fits=item['fits'];x=np.array([v['nll'] for v in fits]);y=np.array([v['accuracy'] for v in fits])
        ax.scatter(x,y,color=_color(name),s=23,alpha=.5)
        ax.scatter(x.mean(),y.mean(),color=_color(name),marker='D',s=43,label=_name(name))
    ax.set_xlabel('Identity NLL (equal series; lower is better)')
    ax.set_ylabel('First-choice accuracy (equal series)')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.6)
    ax.set_title('(a)',loc='left',fontweight='bold')
    ax.legend(fontsize=7.1,loc='best')
    groups=np.asarray(d['match']).astype(str);series=sorted(set(groups[unique.any(-1)]))
    categories=[(True,True),(True,False),(False,True),(False,False)]
    labels=['Both correct','Only selected correct','Only comparator correct','Both incorrect']
    contributions=[];counts=[]
    for a,b in pairs:
        delta=b['loss']-a['loss'];cs=[];ns=[]
        for ca,cb in categories:
            keep=unique&(a['correct']==ca)&(b['correct']==cb)
            ns.append(int(keep.sum()))
            cs.append(float(np.mean([delta[(groups==s)[:,None]&keep].sum()/unique[groups==s].sum() for s in series])))
        assert ns[0]+ns[1]==int((unique&a['correct']).sum())
        assert ns[0]+ns[2]==int((unique&b['correct']).sum())
        assert sum(ns)==int(unique.sum())
        contributions.append(cs);counts.append(ns)
        expected=np.mean([b['nll_by_series'][s]-a['nll_by_series'][s] for s in series])
        np.testing.assert_allclose(sum(cs),expected,atol=1e-10)
    contributions=np.asarray(contributions);counts=np.asarray(counts)
    avg=contributions.mean(0);palette=[NAVY,TEAL,AMBER,'#77618c']
    bx.axvline(0,color='#9aa6af',lw=1)
    for i in range(4):
        bx.barh(i,avg[i],height=.52,color=palette[i],alpha=.83)
        bx.scatter(contributions[:,i],[i]*len(pairs),facecolors='white',edgecolors=palette[i],s=23,zorder=3)
    bx.set_yticks(range(4),[f'{label}\nmean n={counts[:,i].mean():.1f}' for i,label in enumerate(labels)])
    bx.invert_yaxis();bx.grid(axis='x',alpha=.6)
    bx.set_xlabel('Contribution to paired identity-NLL gain (nat)')
    bx.set_title('(b)',loc='left',fontweight='bold')
    caption=('(a) Probability quality and first-choice accuracy. Small points are fitted seeds and diamonds are their method means. '
        '(b) Exact additive decomposition of '+_name(comparator)+' minus '+_name(primary)+' identity NLL by paired first-choice outcome. '
        'Each category sums losses over its cases but divides by all unique outcomes in the series, then weights series and paired seeds equally; category contributions sum to the full paired NLL difference. '
        'Bars are means and open points are fitted-pair values. Counts are average pooled cases per pair, not additional independent observations. '
        'Positive contributions favor the selected method. Pairing: '+str(_pair_info(pairs,mode)['seed_pairs'])+'.')
    return dict(primary=primary,comparator=comparator,pairing=_pair_info(pairs,mode),categories=labels,
        per_pair_contributions=contributions,per_pair_case_counts=counts,mean_contribution=avg,
        contribution_total=float(avg.sum()),series=series,caption=caption),_save(fig,out,'07_probability_accuracy_tradeoff'),None


def _pack(d,forecasts,primary,path,requested_primary=None):
    arrays={};meta=dict(schema=1,primary=primary,requested_primary=requested_primary or primary,data_fields=[],methods=[])
    for key in DATA_FIELDS:
        if key in d:
            value=np.asarray(d[key])
            if value.dtype.kind=='O': value=value.astype(str)
            arrays['data_'+key]=value;meta['data_fields'].append(key)
    for i,method in enumerate(_ordered(forecasts)):
        entry=dict(method=method,fits=[])
        for j,record in enumerate(forecasts[method]):
            entry['fits'].append(dict(seed=_json(record['seed']),key=f'm{i:03d}_s{j:03d}'))
            arrays[f'm{i:03d}_s{j:03d}_prob']=np.asarray(record['prob'])
            arrays[f'm{i:03d}_s{j:03d}_joint']=np.asarray(record['joint'])
        meta['methods'].append(entry)
    arrays['metadata']=np.asarray(json.dumps(meta,ensure_ascii=False))
    np.savez_compressed(path,**arrays)


def _unpack(path):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']))
        if meta['schema']!=1: raise ValueError('Unsupported figure archive schema')
        d={key:z['data_'+key] for key in meta['data_fields']};forecasts={}
        for method in meta['methods']:
            forecasts[method['method']]=[dict(seed=v['seed'],prob=z[v['key']+'_prob'],joint=z[v['key']+'_joint'],diagnostics={}) for v in method['fits']]
    return d,forecasts,meta.get('requested_primary',meta['primary'])


def _generate(d,forecasts,out,primary,archive_source=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    methods,unique,truth,counts=_validate(d,forecasts)
    requested_primary=primary
    if primary not in methods: primary=next(iter(methods))
    comparator=next((k for k in ('compact_gru','foundation_hurdle','dynamic_graph','task_transformer') if k in methods and k!=primary),next((k for k in methods if k!=primary),None))
    archive=out/'replot_arrays.npz'
    if archive_source is None: _pack(d,forecasts,primary,archive,requested_primary=requested_primary)
    elif Path(archive_source).resolve()!=archive.resolve(): shutil.copyfile(archive_source,archive)
    _style()
    scored_subset,split_labels=_subset(d)
    evidence=dict(schema=1,primary=primary,requested_primary=requested_primary,counts=counts,
        scored_subset=scored_subset,split_labels=split_labels,
        main_table=_table(methods,counts),method_fit_counts={k:len(v['fits']) for k,v in methods.items()},
        array_sha256={key:_array_sha(np.asarray(d[key])) for key in DATA_FIELDS if key in d},
        replot_archive_sha256=_sha(archive),figures={})
    draws=[('01_series_gains',lambda:_heatmap(d,methods,unique,primary,out)),
           ('02_history_motion_profiles',lambda:_profiles(d,methods,unique,truth,primary,comparator,out)),
           ('03_reliability_support',lambda:_reliability(methods,unique,out)),
           ('04_candidate_shortlists',lambda:_shortlists(methods,unique,out)),
           ('05_forecast_corrections',lambda:_corrections(d,methods,unique,primary,comparator,out)),
           ('06_deterministic_cases',lambda:_cases(d,methods,unique,truth,primary,comparator,out)),
           ('07_probability_accuracy_tradeoff',lambda:_tradeoff(d,methods,unique,primary,comparator,out))]
    manifest=dict(schema=1,primary=primary,requested_primary=requested_primary,
        scored_subset=scored_subset,split_labels=split_labels,
        source='frozen same-row probability records; no model inference or fitting',
        statistical_unit='series for log loss; explicitly pooled cases for coverage, calibration, profiles and cases',
        method_fit_counts=evidence['method_fit_counts'],counts=counts,figures={})
    captions=['# Figure captions','',
        'Scored subset: '+scored_subset+'. All counts below refer to this supplied subset. Multiple fitted seeds predict the same outcomes; they are not extra independent matches.',
        '', 'Fit counts: '+_support_text(methods)+'.','']
    for name,draw in draws:
        values,files,reason=draw()
        if values is None:
            manifest['figures'][name]=dict(status='skipped',reason=reason)
            evidence['figures'][name]=dict(status='skipped',reason=reason)
            captions.extend(['## '+name,'','Scored subset: '+scored_subset+'. Skipped: '+reason,''])
        else:
            values['scored_subset']=scored_subset
            values['caption']='Scored subset: '+scored_subset+'. '+values['caption']
            evidence['figures'][name]=values
            manifest['figures'][name]=dict(status='generated',files=files,
                sha256={filename:_sha(out/filename) for filename in files})
            captions.extend(['## '+name,'',values['caption'],''])
    _write(out/'numeric_evidence.json',evidence)
    _write(out/'figure_manifest.json',manifest)
    with (out/'all_method_table.csv').open('w',encoding='utf-8',newline='') as handle:
        fields=('method','name','n_seeds','n_series','unique_outcomes','identity_nll','seed_sd','identity_accuracy','joint_nll','entry_brier')
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows({key:v[key] for key in fields} for v in evidence['main_table'])
    (out/'captions.md').write_text('\n'.join(captions),'utf-8')
    plan='''# Evidence-to-figure plan

| Family | Question and actual measurement | Unit / mask | Supporting or contrary outcome |
|---|---|---|---|
| 01 Series gains | Same-seed comparator NLL minus complete-model NLL | Unique at-risk labels, mean within series then paired seeds | Mixed or negative cells remain visible; missing seed intersections skip comparisons |
| 02 History × motion | Observed winner frequency and probability bias by historical and current radial-speed rank | History-eligible unique outcomes × five candidate opportunities | Biases may expose weaknesses; counts and empty cells are retained |
| 03 Reliability | Confidence versus correctness and forecast-bin support | Repeated fit forecasts on the same unique outcomes | Over/underconfidence shown for every supplied method; no independence CI |
| 04 Shortlists | Entrant rank and top-k coverage | Cases pooled within fit then fits equally weighted | Complete candidate budgets 1–5 and all supplied methods remain visible |
| 05 Corrections | Corrected/introduced first-choice errors and paired NLL by prior-arrival count | Same-seed cases for transitions; equal-series NLL within support groups | Both useful changes and new errors retained; retrained comparisons, not causal effects |
| 06 Cases | Observed two-second motion and conditional probability allocation | Earliest two source rows per region with unique future target | Selection is independent of whether either method wins |

Figures use observed arrays and saved probabilities only. No attention maps,
generated future paths, t-SNE separation, epoch curves, or causal-effect claims
are manufactured. The main comparison CSV always contains every supplied method
and its own fit count. Pairwise figures use the common seed intersection.

Redraw all figures on CPU without training:

    from figures import redraw
    redraw('replot_arrays.npz', 'redrawn_figures')

The NPZ is self-contained for these figures. Saved diagnostics are not consumed
by this module; if needed, they remain in the model forecast exports.
'''
    (out/'figure_plan.md').write_text(plan,'utf-8')
    return manifest


def generate(d,forecasts,out,primary='foundation_hurdle'):
    """Validate all supplied probabilities, save evidence, and generate figures."""
    return _generate(d,forecasts,out,primary)


def redraw(evidence_npz,out):
    """Redraw solely from the self-contained NPZ, without training libraries."""
    d,forecasts,primary=_unpack(Path(evidence_npz))
    return _generate(d,forecasts,out,primary,archive_source=Path(evidence_npz))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=redraw(args.evidence,args.out)
    print(json.dumps({k:v['status'] for k,v in result['figures'].items()},indent=2))
