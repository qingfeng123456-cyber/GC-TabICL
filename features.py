"""Anonymous, target-canonical table rows from the observed friendly prefix.

No estimator, normalization, labels, or file access is involved in ``build``.
The 337 columns comprise own91 + team mean/std182 + sorted peer relations64.
Observation semantics follow V5 prefix_classical, with canonical region views
instead of a region ID and explicit anonymous peer comparisons.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

WIDTH=337
OWN_WIDTH=91
PAIR_NAMES=('relative_x_1000','relative_y_1000',
    'relative_velocity_x_250','relative_velocity_y_250',
    'candidate_distance_1000','goal_distance_gap_1000','radial_velocity_gap_250',
    'peer_habit','peer_last_lead','peer_recent_lead','habit_gap','recent_lead_gap',
    'peer_goal_distance_1000','spawn_relative_x_1000','spawn_relative_y_1000','speed_gap_250')
REGION_STEMS=('distance','rank','gap','habit','team_prior_visit','last_lead','recent_lead','prior_unique_log')
COMMON_NAMES=('start_cash','start_armor','spawn_goal_x_4000','spawn_goal_y_4000','spawn_z',
    'carried_0','carried_1','carried_2','carried_3','carried_4','round_fraction',
    'own_score','opponent_score','prior_rounds_log')
PREFIX_NAMES=('goal_x_4000','goal_y_4000','z_4000','displacement_x_1000','displacement_y_1000',
    'displacement_z_1000','velocity_x_250','velocity_y_250','velocity_z_250','alive','sample_age_05')
GEOMETRY_NAMES=('freeze_distance_4000','current_distance_4000','current_radial_velocity_250',
    'distance_progress_1000','current_goal_rank_4','current_goal_gap_1000')
OWN_NAMES=COMMON_NAMES+tuple('active_'+s for s in REGION_STEMS)+tuple('other_'+s for s in REGION_STEMS)+tuple(
    f'time_{t}_{name}' for t in range(5) for name in PREFIX_NAMES)+GEOMETRY_NAMES
FEATURE_NAMES=OWN_NAMES+tuple('team_mean_'+s for s in OWN_NAMES)+tuple('team_std_'+s for s in OWN_NAMES)+tuple(
    f'peer_{rank+1}_{name}' for rank in range(4) for name in PAIR_NAMES)
assert len(OWN_NAMES)==OWN_WIDTH and len(FEATURE_NAMES)==WIDTH


def _inputs(d,rows):
    rows=np.asarray(rows,dtype=np.int64)
    if rows.ndim!=1:raise ValueError('rows must be a one-dimensional integer index')
    # This is the entire inference whitelist. In particular, no mask, risk,
    # future array, player ID, round ID, group ID, or split enters a feature.
    x=np.asarray(d['x'],dtype=np.float64)
    p=np.asarray(d['prefix'],dtype=np.float64)
    centers=np.asarray(d['centers'],dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(5,30) or p.shape!=(len(x),5,5,11):
        raise ValueError('Expected x[N,5,30] and prefix[N,5,5,11]')
    if centers.shape!=(2,2):raise ValueError('Expected centers[2,2]')
    if rows.size and (rows.min()<0 or rows.max()>=len(x)):raise IndexError('rows out of range')
    x=x[rows];p=p[rows]
    if not (np.isfinite(x).all() and np.isfinite(p).all() and np.isfinite(centers).all()):
        raise ValueError('Nonfinite observed input')
    return x,p,centers


def _own(x,p,centers):
    n=len(x)
    static=[]
    for goal in range(2):
        common=x[...,:14].copy()
        common[...,2:4]-=centers[goal]/4000.
        static.append(np.concatenate((common,x[...,14+8*goal:22+8*goal],
                                      x[...,14+8*(1-goal):22+8*(1-goal)]),-1))
    static=np.stack(static,1)
    prefix=np.broadcast_to(p[:,None],(n,2,5,5,11)).copy()
    prefix[...,:2]-=centers[None,:,None,None]/4000.
    xy0=p[:,:,0,:2]*4000.;xyt=p[:,:,-1,:2]*4000.;velocity=p[:,:,-1,6:8]*250.
    offset=centers[None,:,None]-xyt[:,None]
    current=np.linalg.norm(offset,axis=-1)
    initial=np.linalg.norm(centers[None,:,None]-xy0[:,None],axis=-1)
    radial=(offset*velocity[:,None]).sum(-1)/np.maximum(current,1e-6)
    difference=current[:,:,:,None]-current[:,:,None,:]
    equals=np.abs(difference)<=1e-5
    rank=(difference>1e-5).sum(-1)+(equals.sum(-1)-1)*.5
    geometry=np.stack((initial/4000.,current/4000.,radial/250.,(initial-current)/1000.,
                       rank/4.,(current-current.min(-1,keepdims=True))/1000.),-1)
    own=np.concatenate((static,prefix.reshape(n,2,5,55),geometry),-1)
    assert own.shape==(n,2,5,91)
    return own,static,current,radial


def _peer_relations(x,p,static,current,radial):
    n=len(x)
    # Index axes in pair arrays are [round,goal,query candidate,peer candidate].
    xy=p[:,:,-1,:2]*4.  # position channels are in world units / 4000.
    velocity=p[:,:,-1,6:8]  # normalized by 250 already.
    spawn=x[...,2:4]*4.
    delta_xy=xy[:,None,None]-xy[:,None,:,None]
    delta_velocity=velocity[:,None,None]-velocity[:,None,:,None]
    delta_spawn=spawn[:,None,None]-spawn[:,None,:,None]
    delta_xy=np.broadcast_to(delta_xy,(n,2,5,5,2))
    delta_velocity=np.broadcast_to(delta_velocity,(n,2,5,5,2))
    delta_spawn=np.broadcast_to(delta_spawn,(n,2,5,5,2))
    distance=np.linalg.norm(delta_xy,axis=-1)
    goal_gap=(current[:,:,None]-current[:,:,:,None])/1000.
    radial_gap=(radial[:,:,None]-radial[:,:,:,None])/250.
    habit=static[...,17];last=static[...,19];recent=static[...,20]
    other=lambda values:np.broadcast_to(values[:,:,None],(n,2,5,5))
    habit_other=other(habit);last_other=other(last);recent_other=other(recent)
    habit_gap=habit_other-habit[:,:,:,None]
    recent_gap=recent_other-recent[:,:,:,None]
    peer_goal_distance=other(current)/1000.
    speed=np.linalg.norm(velocity,axis=-1)
    speed_gap=np.broadcast_to(speed[:,None,None]-speed[:,None,:,None],(n,2,5,5))
    pairs=np.concatenate((delta_xy,delta_velocity,distance[...,None],goal_gap[...,None],radial_gap[...,None],
        habit_other[...,None],last_other[...,None],recent_other[...,None],habit_gap[...,None],recent_gap[...,None],
        peer_goal_distance[...,None],delta_spawn,speed_gap[...,None]),-1)
    assert pairs.shape==(n,2,5,5,16)
    peers=np.asarray([[k for k in range(5) if k!=j] for j in range(5)])
    pairs=np.take_along_axis(pairs,peers[None,None,:,:,None],axis=3)
    # Closest-to-goal competitor first; all ties use observed values only.
    # A roster-slot tie break would violate player permutation equivariance.
    priority=[12,4]+[k for k in range(16) if k not in (12,4)]
    order=np.lexsort(tuple(pairs[...,k] for k in reversed(priority)),axis=-1)
    pairs=np.take_along_axis(pairs,order[...,None],axis=3)
    return pairs.reshape(n,2,5,64)


def build(d,rows):
    """Return unnormalized float32 X[len(rows),2,5,337], without labels.

    All five observed frames are retained. Region views are canonicalized as
    active/other, and peer sorting uses observations rather than player IDs.
    No data-dependent preprocessing is fit here; a caller may normalize using
    training rows only. Array dimensions refer to two goals and five players.
    """
    x,p,centers=_inputs(d,rows)
    own,static,current,radial=_own(x,p,centers)
    # Sorting each column first gives bit-identical reduction order under any
    # player permutation, including float64 accumulation and float32 output.
    ordered=np.sort(own,axis=2)
    context=np.concatenate((ordered.mean(2),ordered.std(2)),-1)
    team=np.broadcast_to(context[:,:,None],(len(x),2,5,182))
    result=np.concatenate((own,team,_peer_relations(x,p,static,current,radial)),-1)
    if result.shape!=(len(x),2,5,WIDTH) or not np.isfinite(result).all():
        raise ValueError('Invalid constructed features')
    return np.ascontiguousarray(result,dtype=np.float32)


def labelrecords(d,rows):
    """Return (Xflat,y,roundids,goal,player,group) for unique at-risk events.

    Each eligible region event contributes exactly five candidate rows: one
    binary positive and four negatives. ``roundids`` uses immutable source_rows
    (or input row indices if source_rows is absent); ``group`` is the series ID.
    Empty, tied, and already-observed regional events do not enter this identity
    bank. Their exclusion is supervision-only, never a feature construction.
    """
    rows=np.asarray(rows,dtype=np.int64);features=build(d,rows)
    masks=np.asarray(d['mask'],dtype=np.int64)[rows]
    risk=np.asarray(d['risk'],dtype=bool)[rows]
    if masks.shape!=(len(rows),2) or risk.shape!=masks.shape:raise ValueError('Expected regional mask/risk')
    unique=(masks>0)&((masks&(masks-1))==0)&risk
    rr,gg=np.where(unique)
    player=np.tile(np.arange(5,dtype=np.int64),len(rr))
    y=((masks[rr,gg,None]&(1<<np.arange(5)))>0).astype(np.int64).reshape(-1)
    source=np.asarray(d['source_rows'],dtype=np.int64)[rows] if 'source_rows' in d else rows
    roundids=np.repeat(source[rr],5);goal=np.repeat(gg.astype(np.int64),5)
    group=np.repeat(np.asarray(d['match']).astype(str)[rows[rr]],5)
    return features[rr,gg].reshape(-1,WIDTH),y,roundids,goal,player,group


def _checks():
    here=Path(__file__).resolve().parent
    path=here.parents[1]/'v5_early_motion/portable/data/development.npz'
    keys=('x','prefix','centers','mask','risk','split','source_rows','match','prefix_grid',
          'freeze_end_tick','prefix_tick')
    with np.load(path,allow_pickle=False) as archive:d={key:archive[key] for key in keys}
    split=d['split'].astype(str)
    assert set(split)=={'train','validation'}
    rows=np.arange(len(split));x=build(d,rows)
    permutation=np.asarray([3,1,4,0,2])
    altered=dict(d,x=d['x'][:,permutation],prefix=d['prefix'][:,permutation])
    permuted=build(altered,rows)
    np.testing.assert_array_equal(permuted,x[:,:,permutation])
    swapped_x=d['x'].copy();swapped_x[...,14:22]=d['x'][...,22:30];swapped_x[...,22:30]=d['x'][...,14:22]
    swapped=build(dict(d,x=swapped_x,centers=d['centers'][[1,0]]),rows)
    np.testing.assert_array_equal(swapped,x[:,[1,0]])
    class ObservedOnly(dict):
        def __getitem__(self,key):
            if key not in {'x','prefix','centers'}:raise AssertionError('Feature construction accessed '+key)
            return super().__getitem__(key)
    np.testing.assert_array_equal(build(ObservedOnly(d),rows),x)
    assert np.max((d['prefix_tick']-d['freeze_end_tick'][:,None]))<=128
    counts={}
    for name,events in [('train',885),('validation',233)]:
        selected=np.flatnonzero(split==name)
        X,y,roundids,goal,player,group=labelrecords(d,selected)
        assert X.shape==(events*5,WIDTH) and y.sum()==events
        assert np.all(y.reshape(-1,5).sum(-1)==1)
        assert np.all(player.reshape(-1,5)==np.arange(5))
        counts[name]=dict(rounds=len(selected),series=len(set(group)),unique_events=events,
                          candidate_rows=len(X),positive_rows=int(y.sum()))
    result=dict(status='passed',feature_width=WIDTH,own_features=91,team_features=182,peer_features=64,
        shape=list(x.shape),player_permutation_max_error=float(np.max(np.abs(permuted-x[:,:,permutation]))),
        goal_exchange_max_error=float(np.max(np.abs(swapped-x[:,[1,0]]))),
        inference_input_whitelist=['x','prefix','centers'],no_normalization_fit=True,
        no_target_or_future_access_during_build=True,test_loaded=False,
        prefix_native_tick_cutoff_max=int(np.max(d['prefix_tick']-d['freeze_end_tick'][:,None])),
        training_validation=counts,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (here/'FEATURE_CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    (here/'FEATURE_NAMES.json').write_text(json.dumps(list(FEATURE_NAMES),indent=2),'utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':_checks()
