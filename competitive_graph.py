"""Dynamic competitive choice with differentiable Rank Centrality.

Known ingredients: Rank Centrality (Negahban, Oh, Shah) and learned dynamic
interaction graphs (Kipf et al.). This prototype learns ordered pair edges
from actual five-frame relative trajectories. It is not a physical time model.
"""
import torch
from torch import nn
from torch.nn import functional as F

from progress_model import ProgressChoice


def rank_centrality(unary,edge):
    """Stationary score of transition i -> j, where edge[i,j] favours i.

    With edge=0, detailed balance gives stationary=softmax(unary), apart
    from the explicit numerical clipping at extreme comparison probabilities.
    """
    n=unary.shape[-1]
    preference=unary[..., :,None]-unary[...,None,:]+edge
    off=torch.sigmoid(-preference).double().clamp(1e-6,1-1e-6)
    eye=torch.eye(n,device=unary.device,dtype=torch.float64)
    off=off*(1-eye)/n
    transition=off+torch.diag_embed(1-off.sum(-1))
    # (I-P^T + 11^T) pi = 1 is nonsingular for this positive connected chain.
    matrix=eye-transition.transpose(-1,-2)+torch.ones_like(transition)
    stationary=torch.linalg.solve(matrix,torch.ones_like(unary,dtype=torch.float64)[...,None]).squeeze(-1)
    if not torch.isfinite(stationary).all() or (stationary<=0).any():
        raise RuntimeError('Invalid stationary competitive distribution')
    stationary=stationary/stationary.sum(-1,keepdim=True)
    return stationary.to(unary.dtype),transition.to(unary.dtype)


class CompetitiveGraph(nn.Module):
    def __init__(self,kind='dynamic_graph'):
        super().__init__()
        if kind not in ('dynamic_graph','unary_control'):raise ValueError(kind)
        self.kind=kind
        self.base=ProgressChoice('direct_goal')
        if kind=='dynamic_graph':
            self.relative_motion=nn.GRU(20,12,batch_first=True)
            self.edge=nn.Sequential(nn.Linear(76,24),nn.SiLU(),nn.Linear(24,1))
            # Start from the same distribution as the existing unary model.
            nn.init.zeros_(self.edge[-1].weight);nn.init.zeros_(self.edge[-1].bias)

    def distribution(self,x,prefix):
        features,motion=self.base.encode(x,prefix)
        team=features.mean(2)
        unary=self.base.player(torch.cat((features,team[:,:,None].expand_as(features)),-1)).squeeze(-1)
        entered=self.base.noevent(team).squeeze(-1)
        edge=unary[..., :,None]*0+unary[...,None,:]*0
        if self.kind=='dynamic_graph':
            b,g,n,t,c=prefix.shape
            relative=prefix[:,:,:,None]-prefix[:,:,None,:]
            sequence,_=self.relative_motion(relative.reshape(b*g*n*n,t,c))
            pair=sequence[:,-1].reshape(b,g,n,n,12)
            left=features[:,:,:,None].expand(-1,-1,-1,n,-1)
            right=features[:,:,None,:].expand(-1,-1,n,-1,-1)
            ordered=self.edge(torch.cat((left,right,pair),-1)).squeeze(-1)
            edge=.5*(ordered-ordered.transpose(-1,-2))
        conditional,transition=rank_centrality(unary,edge)
        region=torch.cat((F.logsigmoid(-entered)[...,None],
                          F.logsigmoid(entered)[...,None]+conditional.log()),-1)
        return region[:,0,:,None]+region[:,1,None,:],dict(
            conditional=conditional,transition=transition,pair_edge=edge,
            unary=unary,player_embedding=features,motion_embedding=motion)

    def forward(self,x,prefix):return self.distribution(x,prefix)[0]

    @torch.no_grad()
    def diagnostics(self,x,prefix):return self.distribution(x,prefix)[1]


def structural_checks(model,x,prefix):
    model.eval()
    with torch.no_grad():
        table,diag=model.distribution(x,prefix)
        torch.testing.assert_close(table.flatten(1).logsumexp(-1),table.new_zeros(len(x)),atol=3e-6,rtol=0)
        p=torch.tensor([3,1,4,0,2],device=x.device)
        states=torch.cat((torch.zeros(1,dtype=torch.long,device=x.device),p+1))
        changed=model(x[:,:,p],prefix[:,:,p])
        error=float((changed-table[:,states][:,:,states]).abs().max())
        torch.testing.assert_close(changed,table[:,states][:,:,states],atol=3e-6,rtol=0)
        torch.testing.assert_close(model(x[:,[1,0]],prefix[:,[1,0]]),table.transpose(-1,-2),atol=3e-6,rtol=0)
        torch.testing.assert_close(diag['pair_edge'],-diag['pair_edge'].transpose(-1,-2),atol=1e-7,rtol=0)
        torch.testing.assert_close(diag['transition'].sum(-1),torch.ones_like(diag['unary']),atol=2e-6,rtol=0)
        stationary=torch.matmul(diag['conditional'][...,None,:],diag['transition']).squeeze(-2)
        torch.testing.assert_close(stationary,diag['conditional'],atol=2e-6,rtol=0)
        u=torch.randn(7,2,5,device=x.device)
        noedges=torch.zeros(7,2,5,5,device=x.device)
        central,_=rank_centrality(u,noedges)
        torch.testing.assert_close(central,u.softmax(-1),atol=2e-6,rtol=0)
        cyclic=noedges.clone();cyclic[...,0,1]=1.5;cyclic[...,1,0]=-1.5
        other,_=rank_centrality(u,cyclic)
        changed_by_edge=float((central-other).abs().max())
        assert changed_by_edge>1e-3
    return dict(player_equivariance_max_error=error,goal_exchange=True,stochastic_transition=True,
                stationary_equation_checked=True,antisymmetric_edges=True,
                no_edges_reduces_to_softmax=True,nontrivial_edge_change=changed_by_edge)
