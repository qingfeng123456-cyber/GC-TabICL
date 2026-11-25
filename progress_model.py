"""Goal-relative prefix dynamics with normalized remaining first-event risks.

Known predecessors include dynamic competing-risk networks (DeepHit) and
goal-conditioned motion models. This is a bounded task-adapted prototype.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import risk_loss, group_scores, mean_metric

KINDS=('direct_goal','race_no_time','race_time')
EDGES=np.asarray([2.,4.,6.,8.,10.,13.,16.,20.,25.],np.float32)
PREFIX_NAMES=('goal_x_1000','goal_y_1000','z_1000',
              'displacement_x_1000','displacement_y_1000','displacement_z_1000',
              'velocity_x_250','velocity_y_250','velocity_z_250','alive','sample_age_05',
              'goal_distance_1000','goal_progress_1000','radial_velocity_250',
              'peer_displacement_x_1000','peer_displacement_y_1000','peer_displacement_z_1000',
              'peer_velocity_x_250','peer_velocity_y_250','peer_velocity_z_250')


def prepare_inputs(d,train):
    """Only current input channels and fixed goal centers construct features."""
    raw=np.asarray(d['x'],np.float64)
    p=np.asarray(d['prefix'],np.float64)
    centers=np.asarray(d['centers'],np.float64)
    if raw.shape[1:]!=(5,30) or p.shape[1:]!=(5,5,11) or centers.shape!=(2,2):
        raise ValueError('Unexpected development feature schema')
    names=list(map(str,d['feature_names']))
    if names[14]!='distance_A' or names[22]!='distance_B':
        raise ValueError('Unexpected original region feature ordering')
    goals=[];motions=[]
    absolute=p[...,:3]*4000.
    displacement=p[...,3:6]
    velocity=p[...,6:9]
    peer_displacement=displacement-displacement.mean(1,keepdims=True)
    peer_velocity=velocity-velocity.mean(1,keepdims=True)
    for g in range(2):
        base=raw[...,:14].copy()
        base[...,2:4]=(d['spawn_xy'].astype(np.float64)-centers[g])/1000.
        active=raw[...,14+8*g:22+8*g]
        other=raw[...,14+8*(1-g):22+8*(1-g)]
        goals.append(np.concatenate((base,active,other),-1))
        goal_position=absolute.copy()/1000.
        goal_position[...,:2]-=centers[g]/1000.
        distance=np.linalg.norm(goal_position[...,:2],axis=-1)
        progress=distance[...,:1]-distance
        radial=-(velocity[...,:2]*goal_position[...,:2]).sum(-1)/np.maximum(distance,1e-6)
        motions.append(np.concatenate((goal_position,displacement,velocity,p[...,9:11],
                         distance[...,None],progress[...,None],radial[...,None],
                         peer_displacement,peer_velocity),-1))
    x=np.stack(goals,1);prefix=np.stack(motions,1)
    assert x.shape[1:]==(2,5,30) and prefix.shape[1:]==(2,5,5,20)
    xm=x[train].mean((0,1,2));xs=x[train].std((0,1,2));xs[xs<1e-6]=1.
    pm=prefix[train].mean((0,1,2,3));ps=prefix[train].std((0,1,2,3));ps[ps<1e-6]=1.
    x=((x-xm)/xs).clip(-10,10).astype(np.float32)
    prefix=((prefix-pm)/ps).clip(-10,10).astype(np.float32)
    assert np.isfinite(x).all() and np.isfinite(prefix).all()
    return torch.from_numpy(x),torch.from_numpy(prefix),dict(x_mean=xm,x_scale=xs,
        prefix_mean=pm,prefix_scale=ps,prefix_names=np.asarray(PREFIX_NAMES),centers=centers,
        original_feature_names=d['feature_names'],time_bin_edges=EDGES)


class ProgressChoice(nn.Module):
    """Same compact goal-aware encoder for direct and first-passage heads.

    Each bin chooses no first event or one of five first-arrival causes.
    Candidate outcomes are invariant/equivariant under consistent goal/player
    reordering. No independent latent player-arrival-time assumption is used.
    """
    def __init__(self,kind):
        super().__init__()
        if kind not in KINDS: raise ValueError(kind)
        self.kind=kind
        self.motion=nn.GRU(20,24,batch_first=True)
        self.static=nn.Sequential(nn.Linear(30,24),nn.SiLU())
        self.fusion=nn.Sequential(nn.Linear(48,32),nn.LayerNorm(32),nn.SiLU())
        width=1 if kind=='direct_goal' else len(EDGES)-1
        self.player=nn.Sequential(nn.Linear(64,24),nn.SiLU(),nn.Linear(24,width))
        self.noevent=nn.Linear(32,width)
        self.register_buffer('time_bin_edges',torch.from_numpy(EDGES.copy()))
        if kind!='direct_goal':
            # Constant initial event rate gives survival 0.5 over (2,25].
            survival=np.exp(-math.log(2)*np.diff(EDGES)/(EDGES[-1]-EDGES[0]))
            with torch.no_grad():
                self.noevent.weight.zero_()
                self.noevent.bias.copy_(torch.from_numpy(np.log(5*survival/(1-survival))))

    def encode(self,x,prefix):
        if x.shape[1:]!=(2,5,30) or prefix.shape[1:]!=(2,5,5,20):
            raise ValueError('Expected goal-aware x[B,2,5,30] and prefix[B,2,5,5,20]')
        batch=len(x)
        encoded,_=self.motion(prefix.reshape(batch*2*5,5,20))
        motion=encoded[:,-1].reshape(batch,2,5,24)
        features=self.fusion(torch.cat((self.static(x),motion),-1))
        return features,motion

    def distribution(self,x,prefix):
        features,motion=self.encode(x,prefix)
        team=features.mean(2)
        scores=self.player(torch.cat((features,team[:,:,None].expand_as(features)),-1))
        noevent=self.noevent(team)
        log_time_mass=None
        if self.kind=='direct_goal':
            unary=scores.squeeze(-1);entered=noevent.squeeze(-1)
            region=torch.cat((F.logsigmoid(-entered)[...,None],
                F.logsigmoid(entered)[...,None]+unary.log_softmax(-1)),-1)
        else:
            # [round, goal, timebin, no-first-event-or-player]
            hazard=torch.cat((noevent[...,None],scores.transpose(-1,-2)),-1).log_softmax(-1)
            log_continue=hazard[...,0]
            exclusive=log_continue.cumsum(-1)-log_continue
            log_time_mass=exclusive[...,None]+hazard[...,1:]
            region=torch.cat((log_continue.sum(-1)[...,None],log_time_mass.logsumexp(-2)),-1)
        table=region[:,0,:,None]+region[:,1,None,:]
        return table,region,log_time_mass,features,motion

    def forward(self,x,prefix):
        return self.distribution(x,prefix)[0]

    def auxiliary(self,region,log_time_mass,mask,risk,arrival_seconds):
        """Time likelihood conditional on the compatible winning set.

        P(time bin=b | winner in C) = sum_{j in C} P(b,j) /
                                      sum_{j in C} P(j).
        Entry and identity likelihoods are already in the main risk loss.
        """
        if self.kind!='race_time': return region.sum()*0,{}
        eligible=risk.bool()&(mask>0)
        if not eligible.any(): return region.sum()*0,dict(time_targets=0,time_conditional_nll=0.)
        times=arrival_seconds[eligible]
        if not torch.isfinite(times).all() or not ((times>self.time_bin_edges[0])&(times<=self.time_bin_edges[-1])).all():
            raise ValueError('Nonempty at-risk first event must lie in (2,25] for auxiliary time labels')
        bins=torch.bucketize(times.contiguous(),self.time_bin_edges[1:],right=False)
        bits=1<<torch.arange(5,device=mask.device)
        accepted=(mask[eligible][:,None]&bits)>0
        masses=log_time_mass[eligible]
        selected=masses[torch.arange(len(times),device=times.device),bins]
        numerator=selected.masked_fill(~accepted,-torch.inf).logsumexp(-1)
        denominator=region[...,1:][eligible].masked_fill(~accepted,-torch.inf).logsumexp(-1)
        loss=(denominator-numerator).mean()
        return .2*loss,dict(time_targets=int(eligible.sum()),time_conditional_nll=float(loss.detach()))

    @torch.no_grad()
    def diagnostics(self,x,prefix):
        _,region,mass,features,motion=self.distribution(x,prefix)
        result=dict(player_embedding=features,motion_embedding=motion,region_prob=region.exp())
        if mass is not None: result['first_event_time_mass']=mass.exp()
        return result


def make_model(kind):
    model=ProgressChoice(kind)
    if sum(p.numel() for p in model.parameters())>20000: raise ValueError('Parameter budget exceeded')
    return model


def model_checks(model,x,prefix,mask,risk,arrival_seconds):
    model.eval()
    with torch.no_grad():
        table,region,mass,_,_=model.distribution(x,prefix)
        torch.testing.assert_close(region.logsumexp(-1),region.new_zeros(region.shape[:-1]),atol=2e-6,rtol=0)
        torch.testing.assert_close(table.flatten(1).logsumexp(-1),table.new_zeros(len(x)),atol=3e-6,rtol=0)
        p=torch.tensor([3,1,4,0,2],device=x.device)
        index=torch.cat((torch.zeros(1,dtype=torch.long,device=x.device),p+1))
        actual=model(x[:,:,p],prefix[:,:,p])
        torch.testing.assert_close(actual,table[:,index][:,:,index],atol=3e-6,rtol=0)
        swapped=model(x[:,[1,0]],prefix[:,[1,0]])
        torch.testing.assert_close(swapped,table.transpose(-1,-2),atol=3e-6,rtol=0)
        zero=risk_loss(table,mask,torch.zeros_like(risk))
        torch.testing.assert_close(zero,zero.new_zeros(()),atol=1e-8,rtol=0)
        auxiliary,_=model.auxiliary(region,mass,mask,risk,arrival_seconds)
        assert torch.isfinite(auxiliary)
        if mass is not None:
            torch.testing.assert_close(mass.exp().sum((-1,-2))+region[...,0].exp(),
                region.new_ones(region.shape[:-1]),atol=3e-6,rtol=0)
    return dict(normalized=True,player_equivariance_max_error=float((actual-table[:,index][:,:,index]).abs().max()),
                goal_swap_equivariance=True,no_risk_loss_zero=True,
                first_event_mass_normalized=mass is not None,inference_uses_arrival_seconds=False)
