"""Prefix-only responsibility predictors and training-only future auxiliaries.

Known precedents: I-JEPA/V-JEPA joint-embedding prediction and early-action
teacher/student prediction. This is a CS2 task adaptation, not a claim that
latent future prediction or EMA teachers are new algorithms.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KINDS=('flat_prefix','temporal','future_coordinates','future_latent')
MOTION=(3,4,5,6,7,8)
SIDE=(0,1,2,9,10)


def compatible(mask):
    bits=1<<torch.arange(5,device=mask.device)
    return torch.cat(((mask==0)[...,None],(mask[...,None]&bits)>0),-1)


def risk_loss(table,mask,risk):
    """Marginalize inactive regions; supervise identity only where at risk."""
    accepted=compatible(mask)
    accepted=torch.where(risk[...,None],accepted,torch.ones_like(accepted))
    allowed=accepted[:,0,:,None]&accepted[:,1,None,:]
    denom=table.flatten(1).logsumexp(-1)
    active_round=risk.any(-1)
    joint=denom-table.masked_fill(~allowed,-torch.inf).flatten(1).logsumexp(-1)
    loss=joint[active_round].mean() if active_round.any() else table.sum()*0
    marginal=torch.stack((table.logsumexp(2),table.logsumexp(1)),1)
    players=compatible(mask)[...,1:]
    unique=(players.sum(-1)==1)&risk
    if unique.any():
        conditional=marginal[...,1:][unique].log_softmax(-1)
        loss=loss-conditional.masked_select(players[unique]).mean()
    return loss


def joint_table(unary,entered):
    region=torch.cat((F.logsigmoid(-entered)[...,None],
                      F.logsigmoid(entered)[...,None]+unary.log_softmax(-1)),-1)
    return region[:,0,:,None]+region[:,1,None,:]


class PrefixChoice(nn.Module):
    """Same prefix inference graph across the three temporal variants.

    The GRU processes relative displacement and velocity (six channels).
    Absolute coordinates, alive flags, and staleness are also available through
    a shared side branch, so none of the eleven observed channels is dropped.
    The EMA future target contains motion only, without copied static features.
    """
    def __init__(self,kind,prefix_channels=11,frames=5,prefix_mean=None,prefix_scale=None):
        super().__init__()
        if kind not in KINDS or prefix_channels!=11:
            raise ValueError('Expected known model and 11-channel prefix')
        self.kind=kind;self.frames=frames;self.hidden=24
        if kind=='flat_prefix':
            width=30+frames*prefix_channels
        else:
            self.motion=nn.GRU(6,self.hidden,batch_first=True)
            width=30+frames*len(SIDE)+self.hidden
        self.fusion=nn.Sequential(nn.Linear(width,32),nn.LayerNorm(32),nn.SiLU())
        self.player=nn.Sequential(nn.Linear(64,24),nn.SiLU(),nn.Linear(24,2))
        self.presence=nn.Linear(32,2)
        if kind=='future_coordinates': self.coordinate_head=nn.Linear(32,12)
        if kind=='future_latent':
            self.predictor=nn.Sequential(nn.Linear(30+self.hidden,32),nn.SiLU(),nn.Linear(32,self.hidden))
            self.teacher=deepcopy(self.motion)
            for parameter in self.teacher.parameters(): parameter.requires_grad_(False)
        mean=torch.zeros(prefix_channels) if prefix_mean is None else torch.as_tensor(prefix_mean,dtype=torch.float32)
        scale=torch.ones(prefix_channels) if prefix_scale is None else torch.as_tensor(prefix_scale,dtype=torch.float32)
        self.register_buffer('prefix_mean',mean)
        self.register_buffer('prefix_scale',scale.clamp_min(1e-6))

    def motion_embedding(self,values,valid=None,teacher=False):
        batch,players,steps,channels=values.shape
        encoder=self.teacher if teacher else self.motion
        encoded,_=encoder(values.reshape(batch*players,steps,channels))
        encoded=encoded.reshape(batch,players,steps,self.hidden)
        if valid is None: return encoded[:,:,-1]
        weight=valid.to(encoded.dtype)
        return (encoded*weight[...,None]).sum(-2)/weight.sum(-1,keepdim=True).clamp_min(1)

    def prefix_features(self,x,prefix):
        if prefix.shape[1:]!=(5,self.frames,11) or x.shape[1:]!=(5,30):
            raise ValueError('Expected x[B,5,30], prefix[B,5,T,11]')
        if self.kind=='flat_prefix':
            return self.fusion(torch.cat((x,prefix.flatten(-2)),-1)),None,None
        motion=self.motion_embedding(prefix[...,MOTION])
        side=prefix[...,SIDE].flatten(-2)
        features=self.fusion(torch.cat((x,side,motion),-1))
        return features,motion,side

    def from_features(self,features):
        team=features.mean(1)
        unary=self.player(torch.cat((features,team[:,None].expand_as(features)),-1)).transpose(1,2)
        return joint_table(unary,self.presence(team))

    def forward(self,x,prefix):
        """Inference accepts only available state and observed prefix."""
        features,_,_=self.prefix_features(x,prefix)
        return self.from_features(features)

    @torch.no_grad()
    def update_teacher(self,momentum=.99):
        if self.kind=='future_latent':
            for target,online in zip(self.teacher.parameters(),self.motion.parameters()):
                target.mul_(momentum).add_(online,alpha=1-momentum)

    def auxiliary(self,x,prefix,future,future_valid,mask,risk,diagnostics=False):
        """All future tensors belong to training supervision, never inference."""
        features,motion,side=self.prefix_features(x,prefix)
        zero=features.sum()*0
        if self.kind not in ('future_coordinates','future_latent'): return zero,{}
        valid=future_valid.bool()
        if future.shape[-2:]!=(4,3) or valid.shape!=future.shape[:-1]:
            raise ValueError('Expected four future displacement targets')
        if self.kind=='future_coordinates':
            prediction=self.coordinate_head(features).reshape_as(future)
            squared=(prediction-future).square()*valid[...,None]
            mse=squared.sum()/(valid.sum()*3).clamp_min(1)
            return .1*mse,dict(future_coordinate_mse=float(mse.detach()))
        # Future points are at cutoff+1,+2,+3,+4 seconds. Prefix velocities are
        # in units of 250 world units/s and displacements in units of 1000.
        previous=torch.cat((torch.zeros_like(future[:,:,:1]),future[:,:,:-1]),-2)
        velocity=4*(future-previous)
        previous_valid=torch.cat((torch.ones_like(valid[:,:,:1]),valid[:,:,:-1]),-1)
        velocity=velocity*(valid&previous_valid)[...,None]
        raw=torch.cat((future,velocity),-1)
        numerical=((raw-self.prefix_mean[list(MOTION)])/self.prefix_scale[list(MOTION)]).clamp(-10,10)
        numerical=numerical*valid[...,None]
        online_future=self.motion_embedding(numerical,valid)
        future_features=self.fusion(torch.cat((x,side,online_future),-1))
        # Future-supervised role loss is active only where every candidate has
        # at least one observed target frame. Main prefix loss keeps all rows.
        future_risk=risk&valid.any(-1).all(-1)[:,None]
        future_role=risk_loss(self.from_features(future_features),mask,future_risk)
        with torch.no_grad(): target=self.motion_embedding(numerical,valid,teacher=True)
        prediction=self.predictor(torch.cat((x,motion),-1))
        # Layer normalization specifies scale; it does not prove non-collapse.
        difference=(F.layer_norm(prediction,(self.hidden,))-F.layer_norm(target,(self.hidden,))).square().mean(-1)
        available=valid.any(-1)
        latent=(difference*available).sum()/available.sum().clamp_min(1)
        stats=dict(future_role_loss=float(future_role.detach()),future_latent_mse=float(latent.detach()))
        if diagnostics:
            with torch.no_grad():
                rows=target[available]
                stats['future_players']=int(available.sum())
                if len(rows)>1:
                    std=rows.std(0,unbiased=False)
                    centered=rows-rows.mean(0)
                    eigen=torch.linalg.eigvalsh(centered.T@centered/max(len(rows),1)).clamp_min(0)
                    probability=eigen/eigen.sum().clamp_min(1e-12)
                    stats.update(teacher_mean_dimension_std=float(std.mean()),
                        teacher_effective_rank=float((-(probability*probability.clamp_min(1e-12).log()).sum()).exp()),
                        student_mean_dimension_std=float(motion.flatten(0,1).std(0,unbiased=False).mean()))
        return .3*future_role+.1*latent,stats

    @torch.no_grad()
    def diagnostics(self,x,prefix):
        features,motion,_=self.prefix_features(x,prefix)
        result=dict(player_embedding=features)
        if motion is not None: result['motion_embedding']=motion
        if self.kind=='future_latent': result['predicted_future_embedding']=self.predictor(torch.cat((x,motion),-1))
        return result


def make_model(kind,prefix_channels=11,**kwargs):
    model=PrefixChoice(kind,prefix_channels,**kwargs)
    if sum(p.numel() for p in model.parameters())>30000: raise ValueError('Parameter budget exceeded')
    return model


def group_scores(d,ids,prob,joint):
    out={}
    for match in sorted(set(d['match'][ids])):
        selected=d['match'][ids]==match;rows=ids[selected]
        p,j=prob[selected],joint[selected]
        masks,risk=d['mask'][rows],d['risk'][rows].astype(bool)
        player=(masks[...,None]&(1<<np.arange(5)))>0
        accepted=np.concatenate(((masks==0)[...,None],player),-1)
        accepted=np.where(risk[...,None],accepted,True)
        unique=(player.sum(-1)==1)&risk
        truth=player.argmax(-1)
        conditional=p[...,1:]/p[...,1:].sum(-1,keepdims=True).clip(1e-30)
        true=np.take_along_axis(conditional,truth[...,None],-1)[...,0].clip(1e-30)
        predicted=conditional.argmax(-1)
        jmass=(j*accepted[:,0,:,None]*accepted[:,1,None,:]).sum((1,2)).clip(1e-30)
        rmask=risk.any(-1)
        def mean(values,keep): return float(values[keep].mean()) if keep.any() else None
        out[str(match)]=dict(rounds=len(rows),eligible_region_rounds=int(risk.sum()),unique=int(unique.sum()),
            hits=int(((predicted==truth)&unique).sum()),identity_nll=mean(-np.log(true),unique),
            identity_accuracy=mean(predicted==truth,unique),all_joint_nll=mean(-np.log(jmass),rmask),
            entry_brier=mean((1-p[...,0]-(masks!=0))**2,risk))
    return out


def mean_metric(values,key):
    values=[v[key] for v in values if v[key] is not None]
    return float(np.mean(values)) if values else None


def model_checks(model,x,prefix):
    model.eval()
    with torch.no_grad():
        table=model(x,prefix)
        torch.testing.assert_close(table.flatten(1).logsumexp(-1),torch.zeros(len(x),device=x.device),atol=2e-6,rtol=0)
        permutation=torch.tensor([3,1,4,0,2],device=x.device)
        index=torch.cat((torch.zeros(1,dtype=torch.long,device=x.device),permutation+1))
        changed=model(x[:,permutation],prefix[:,permutation])
        torch.testing.assert_close(changed,table[:,index][:,:,index],atol=3e-6,rtol=0)
        empty=risk_loss(table,torch.ones((len(x),2),dtype=torch.long,device=x.device),torch.zeros((len(x),2),dtype=torch.bool,device=x.device))
        torch.testing.assert_close(empty,empty.new_zeros(()),atol=1e-8,rtol=0)
    return dict(normalized=True,permutation_max_error=float((changed-table[:,index][:,:,index]).abs().max()),
                no_risk_loss_zero=True,inference_uses_future=False)
