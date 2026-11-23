"""Causal temporal teacher-state encoder with five cross-attention mechanism prototypes."""
import torch
from torch import nn
from torch.nn import functional as F

class PraxisTrace(nn.Module):
    def __init__(self, input_dim=11, hidden=256, prototypes=5, use_graph=True, use_prototypes=True):
        super().__init__()
        self.use_graph=use_graph
        self.use_prototypes=use_prototypes
        self.input=nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.recurrent=nn.GRU(hidden,hidden,batch_first=True) if use_graph else None
        self.prototypes=nn.Parameter(torch.randn(prototypes,hidden)*.05)
        self.q=nn.Linear(hidden,hidden)
        self.k=nn.Linear(hidden,hidden)
        self.v=nn.Linear(hidden,hidden)
        self.head=nn.Sequential(nn.Linear(hidden*2 if use_prototypes else hidden,hidden//2),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden//2,1))
        self.state_head=nn.Linear(hidden,4)
        self.mechanism_head=nn.Linear(hidden,prototypes)

    def forward(self,x):
        h=self.input(x)
        if self.use_graph:
            h,_=self.recurrent(h)
        if self.use_prototypes:
            attention=torch.softmax(self.q(h)@self.k(self.prototypes).T / h.shape[-1]**.5,dim=-1)
            context=attention@self.v(self.prototypes)
            h_out=torch.cat((h,context),dim=-1)
        else:
            attention=None
            h_out=h
        return self.head(h_out).squeeze(-1), self.state_head(h), self.mechanism_head(h), attention, h


def contrastive_loss(z, labels, temperature=.3):
    z=F.normalize(z,dim=-1)
    sim=z@z.T /temperature
    mask=torch.eye(len(z),device=z.device,dtype=torch.bool)
    same=labels[:,None].eq(labels[None,:]) & ~mask
    logp=sim.masked_fill(mask,-1e9).log_softmax(-1)
    valid=same.any(-1)
    if not valid.any():
        return z.sum()*0
    return -(logp*same).sum(-1)[valid].div(same.sum(-1).clamp(min=1)[valid]).mean()
