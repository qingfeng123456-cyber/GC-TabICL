"""Observed-token hierarchy and region-conditioned competitive responsibility queries."""
import math
import torch
from torch import nn
from torch.nn import functional as F

KINDS=('task_transformer','plain_player_transformer','no_social_attention','no_goal_query','no_temporal_order')


class AttentionBlock(nn.Module):
    def __init__(self,width=32,heads=4,dropout=.15):
        super().__init__();self.width=width;self.heads=heads
        self.q=nn.Linear(width,width);self.k=nn.Linear(width,width);self.v=nn.Linear(width,width)
        self.out=nn.Linear(width,width)
        self.q_norm=nn.LayerNorm(width);self.m_norm=nn.LayerNorm(width);self.ff_norm=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*width,width))
        self.dropout=nn.Dropout(dropout)

    def forward(self,query,memory=None,bias=None):
        self_attention=memory is None
        q=self.q_norm(query);m=q if self_attention else self.m_norm(memory)
        batch,n,_=q.shape;k=m.shape[1];h=self.heads;d=self.width//h
        q=self.q(q).reshape(batch,n,h,d).transpose(1,2)
        key=self.k(m).reshape(batch,k,h,d).transpose(1,2)
        value=self.v(m).reshape(batch,k,h,d).transpose(1,2)
        score=torch.matmul(q,key.transpose(-1,-2))/math.sqrt(d)
        if bias is not None:score=score+bias
        attention=score.softmax(-1)
        message=torch.matmul(self.dropout(attention),value).transpose(1,2).reshape(batch,n,self.width)
        query=query+self.dropout(self.out(message))
        query=query+self.dropout(self.ff(self.ff_norm(query)))
        return query,attention


class ResponsibilityTransformer(nn.Module):
    def __init__(self,kind='task_transformer',width=32,dropout=.15):
        super().__init__()
        if kind not in KINDS:raise ValueError(kind)
        self.kind=kind;self.width=width;self.heads=4
        if kind=='plain_player_transformer':
            self.player_projection=nn.Linear(130,width)
            self.plain_layers=nn.ModuleList([AttentionBlock(width,4,dropout) for _ in range(4)])
        else:
            self.motion_projection=nn.Linear(11,width);self.static_projection=nn.Linear(30,width)
            self.goal_motion_projection=nn.Linear(20,width)
            self.time_embedding=nn.Parameter(torch.randn(5,width)*.02)
            self.token_type=nn.Parameter(torch.randn(2,width)*.02)
            self.temporal=AttentionBlock(width,4,dropout)
            if kind!='no_social_attention':
                self.social=AttentionBlock(width,4,dropout)
                self.social_bias=nn.Sequential(nn.Linear(6,8),nn.SiLU(),nn.Linear(8,4))
            if kind!='no_goal_query':
                self.cross=AttentionBlock(width,4,dropout)
                self.cross_bias=nn.Sequential(nn.Linear(10,12),nn.SiLU(),nn.Linear(12,4))
                self.competition=AttentionBlock(width,4,dropout)
            else:
                self.pool_fusion=nn.Linear(2*width,width)
        self.final_norm=nn.LayerNorm(width)
        self.player_head=nn.Sequential(nn.Linear(2*width,16),nn.SiLU(),nn.Linear(16,1))
        self.entry_head=nn.Linear(width,1)
        # These templates encode association, never a learnable roster-slot identity.
        self.register_buffer('memory_player',torch.cat((torch.arange(5).repeat_interleave(5),torch.arange(5).repeat(2))))
        self.register_buffer('query_player',torch.arange(5).repeat(2))
        self.register_buffer('memory_goal',torch.cat((torch.full((25,),-1),torch.arange(2).repeat_interleave(5))))
        self.register_buffer('query_goal',torch.arange(2).repeat_interleave(5))
        self.register_buffer('memory_age',torch.cat((torch.linspace(1,0,5).repeat(5),torch.zeros(10))))

    def social_geometry(self,xy,velocity):
        # B,T,query-player,key-player,channels; all states are observed.
        xy=xy.transpose(1,2);velocity=velocity.transpose(1,2)
        relative=xy[:,:,None]-xy[:,:,:,None]
        vrelative=velocity[:,:,None]-velocity[:,:,:,None]
        distance=relative.square().sum(-1,keepdim=True).sqrt()
        own=torch.eye(5,device=xy.device).expand(len(xy),5,5,5)[...,None]
        return torch.cat((relative,vrelative,distance,own),-1)

    def cross_geometry(self,xy,velocity,centers):
        b=len(xy)
        if self.kind=='no_temporal_order':
            last=xy.mean(2);last_velocity=velocity.mean(2)
        else:last=xy[:,:,-1];last_velocity=velocity[:,:,-1]
        memory_xy=torch.cat((xy.reshape(b,25,2),last[:,None].expand(b,2,5,2).reshape(b,10,2)),1)
        memory_v=torch.cat((velocity.reshape(b,25,2),last_velocity[:,None].expand(b,2,5,2).reshape(b,10,2)),1)
        query_xy=last[:,None].expand(b,2,5,2).reshape(b,10,2)
        query_v=last_velocity[:,None].expand(b,2,5,2).reshape(b,10,2)
        rel=memory_xy[:,None]-query_xy[:,:,None]
        rel_v=memory_v[:,None]-query_v[:,:,None]
        distance=rel.square().sum(-1,keepdim=True).sqrt()
        own=(self.query_player[:,None]==self.memory_player[None]).to(xy).expand(b,-1,-1)[...,None]
        static=(self.memory_goal>=0).to(xy).expand(b,10,-1)[...,None]
        same_goal=(self.query_goal[:,None]==self.memory_goal[None]).to(xy).expand(b,-1,-1)[...,None]
        age=(torch.zeros_like(self.memory_age) if self.kind=='no_temporal_order' else self.memory_age).to(xy).expand(b,10,-1)[...,None]
        query_goal=centers[:,:,None].expand(b,2,5,2).reshape(b,10,2)
        goal_distance=(memory_xy[:,None]-query_goal[:,:,None]).square().sum(-1,keepdim=True).sqrt()
        return torch.cat((rel,rel_v,distance,own,static,same_goal,age,goal_distance),-1)

    def encode(self,static,motion,goal_motion,xy,velocity,centers):
        b=len(static);d=self.width;diagnostics={}
        if self.kind=='plain_player_transformer':
            player=torch.cat((static,goal_motion.flatten(-2)),-1)
            query=self.player_projection(player).reshape(b*2,5,d)
            for layer in self.plain_layers:query,attention=layer(query)
            query=query.reshape(b,2,5,d)
            diagnostics['last_player_attention']=attention.reshape(b,2,4,5,5)
            return query,diagnostics
        temporal=self.motion_projection(motion)+self.token_type[0]
        if self.kind!='no_temporal_order':temporal=temporal+self.time_embedding[None,None]
        temporal,attention=self.temporal(temporal.reshape(b*5,5,d))
        temporal=temporal.reshape(b,5,5,d)
        diagnostics['temporal_attention']=attention.reshape(b,5,4,5,5)
        if self.kind!='no_social_attention':
            bias=self.social_bias(self.social_geometry(xy,velocity)).reshape(b*5,5,5,4).permute(0,3,1,2)
            social=temporal.transpose(1,2).reshape(b*5,5,d)
            social,attention=self.social(social,bias=bias)
            temporal=social.reshape(b,5,5,d).transpose(1,2)
            diagnostics['social_attention']=attention.reshape(b,5,4,5,5)
        history=self.static_projection(static)+self.token_type[1]
        if self.kind=='no_goal_query':
            pooled=temporal.mean(2)[:,None].expand(b,2,5,d)
            query=F.silu(self.pool_fusion(torch.cat((pooled,history),-1)))
        else:
            memory=torch.cat((temporal.reshape(b,25,d),history.reshape(b,10,d)),1)
            summary=goal_motion.mean(3) if self.kind=='no_temporal_order' else goal_motion[:,:,:,-1]
            query=(history+self.goal_motion_projection(summary)).reshape(b,10,d)
            bias=self.cross_bias(self.cross_geometry(xy,velocity,centers)).permute(0,3,1,2)
            query,attention=self.cross(query,memory,bias)
            diagnostics['query_memory_attention']=attention
            query,attention=self.competition(query.reshape(b*2,5,d))
            query=query.reshape(b,2,5,d)
            diagnostics['candidate_competition_attention']=attention.reshape(b,2,4,5,5)
        return query,diagnostics

    def distribution(self,static,motion,goal_motion,xy,velocity,centers):
        query,diagnostics=self.encode(static,motion,goal_motion,xy,velocity,centers)
        query=self.final_norm(query);team=query.mean(2)
        utility=self.player_head(torch.cat((query,team[:,:,None].expand_as(query)),-1)).squeeze(-1)
        entry=self.entry_head(team).squeeze(-1)
        region=torch.cat((F.logsigmoid(-entry)[...,None],F.logsigmoid(entry)[...,None]+utility.log_softmax(-1)),-1)
        joint=region[:,0,:,None]+region[:,1,None,:]
        diagnostics.update(query_embedding=query,utility=utility,region_probability=region.exp())
        return joint,utility,diagnostics

    def forward(self,static,motion,goal_motion,xy,velocity,centers):
        return self.distribution(static,motion,goal_motion,xy,velocity,centers)[0]
