"""Leakage-safe teacher-level splitting, training, metrics, and saving."""
import copy, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .data import MECHANISMS
from .encoder import encode_offline, encode_anthropic
from .model import PraxisTrace, contrastive_loss

FEATURES=['belief','practice','context','reflection','pd_exposure']+MECHANISMS+['coverage']

def seed_all(seed):
    random.seed(seed); np.random.seed(seed);torch.manual_seed(seed)

def features(df, encoder='offline', model_name='claude-sonnet-4-6', cache_path=None):
    # No future state / label is used for input. Encode each row's current evidence.
    if cache_path and Path(cache_path).exists():
        cached=pd.read_csv(cache_path)
        if len(cached)==len(df) and cached.teacher_id.equals(df.teacher_id) and cached.wave.equals(df.wave):
            return cached
    values=[]
    for _,r in df.iterrows():
        scores,spans,conf=(encode_offline(r) if encoder=='offline' else encode_anthropic(r,model_name))
        values.append([*scores,float(r.pd_exposure),*[float(r[m]) for m in MECHANISMS],float((conf>=.5).mean())])
    out=df[['teacher_id','site','pathway','wave','sustained_label','mechanism']].copy()
    out[FEATURES]=np.array(values,dtype=float)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True,exist_ok=True);out.to_csv(cache_path,index=False)
    return out

def split_teachers(df,seed=42):
    teachers=df[['teacher_id','site','pathway']].drop_duplicates()
    strat=teachers.site.astype(str)+'_'+teachers.pathway.astype(str)
    trainval,test=train_test_split(teachers.teacher_id.to_numpy(),test_size=.2,random_state=seed,stratify=strat)
    tmp=teachers.set_index('teacher_id').loc[trainval]
    train,val=train_test_split(trainval,test_size=.2,random_state=seed,stratify=(tmp.site+'_'+tmp.pathway).to_numpy())
    return dict(train=train.tolist(),val=val.tolist(),test=test.tolist())

def tensors(df,ids,device='cpu',missing=0,seed=42):
    subset=df[df.teacher_id.isin(ids)].sort_values(['teacher_id','wave'])
    arr=np.stack([g[FEATURES].to_numpy(np.float32) for _,g in subset.groupby('teacher_id',sort=True)])
    target=np.stack([g.sustained_label.to_numpy(np.float32) for _,g in subset.groupby('teacher_id',sort=True)])
    mech=np.stack([g.mechanism.to_numpy(np.int64) for _,g in subset.groupby('teacher_id',sort=True)])
    if missing:
        rng=np.random.default_rng(seed)
        keep=rng.random(arr[:,:,:4].shape)>=missing
        arr[:,:,:4]*=keep
        arr[:,:,-1]=keep.mean(axis=-1)
    return torch.tensor(arr,device=device),torch.tensor(target,device=device),torch.tensor(mech,device=device)

def metrics(y,prob):
    y=np.asarray(y).astype(int);prob=np.asarray(prob)
    pred=(prob>=.5).astype(int)
    return {'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'kappa':float(cohen_kappa_score(y,pred)),
            'auroc':float(roc_auc_score(y,prob)) if len(np.unique(y))>1 else None,
            'brier':float(brier_score_loss(y,prob)), 'samples':int(len(y)),'positive_rate':float(y.mean())}

def evaluate(model,df,ids,device='cpu',missing=0,seed=42):
    model.eval();x,y,_=tensors(df,ids,device,missing,seed)
    with torch.no_grad():
        logits,*_=model(x)
    prob=torch.sigmoid(logits).cpu().numpy().ravel()
    yy=y.cpu().numpy().ravel()
    return metrics(yy,prob),pd.DataFrame({'teacher_id':np.repeat(sorted(ids),x.shape[1]),'wave':np.tile(np.arange(1,x.shape[1]+1),len(ids)),'truth':yy.astype(int),'probability':prob})

def train_model(df,train_ids,val_ids,seed=42,epochs=50,batch=32,hidden=256,patience=5,
                device='cpu',no_graph=False,no_prototypes=False,no_evidence=False,no_review=False):
    seed_all(seed)
    model=PraxisTrace(hidden=hidden,use_graph=not no_graph,use_prototypes=not no_prototypes).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=1e-4)
    x,y,m=tensors(df,train_ids,device)
    vx,vy,_=tensors(df,val_ids,device)
    best=float('inf');best_state=None;wait=0;history=[]
    for epoch in range(epochs):
        model.train(); order=torch.randperm(len(x),device=device);running=[]
        for sl in order.split(batch):
            xb=x[sl].clone();yb=y[sl];mb=m[sl]
            if no_evidence:xb[:,:,-1]=1.
            elif not no_review:
                # Low-confidence simulated evidence masked before training.
                xb[:,:,:4]*=(xb[:,:,-1:] >= .5)
            logits,state,mlogits,attention,emb=model(xb)
            # State learning works on observed channels, not future labels.
            state_loss=F.smooth_l1_loss(state,xb[:,:,:4]/5.)
            evidence_loss=((torch.sigmoid(state)-xb[:,:,:4]/5.).abs()*xb[:,:,-1:]).mean()
            align_loss=F.binary_cross_entropy_with_logits(logits,yb)
            temp_loss=(torch.sigmoid(logits[:,1:])-torch.sigmoid(logits[:,:-1])).square().mean()
            proto_loss=F.cross_entropy(mlogits.reshape(-1,5),mb.reshape(-1)) if not no_prototypes else align_loss*0
            # Embedding contrast uses available mechanism labels (no future outcome).
            con=contrastive_loss(emb[:,-1,:],mb[:,-1]) if len(sl)>1 and not no_prototypes else align_loss*0
            loss=1.*state_loss+(.8*evidence_loss if not no_evidence else 0)+.6*(proto_loss+con)/2+.5*temp_loss+align_loss
            optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
            running.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_logits,*_=model(vx)
            val_loss=float(F.binary_cross_entropy_with_logits(val_logits,vy))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(running)),'val_bce':val_loss})
        if val_loss<best-1e-5:
            best=val_loss;best_state=copy.deepcopy(model.state_dict());wait=0
        else:
            wait+=1
            if wait>=patience:break
    model.load_state_dict(best_state)
    return model,pd.DataFrame(history)

def baseline(df,splits):
    train=df[df.teacher_id.isin(splits['train'])];test=df[df.teacher_id.isin(splits['test'])]
    # Static per-wave logistic regression. No temporal history in the baseline.
    baseline=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced'))
    baseline.fit(train[FEATURES].to_numpy(),train.sustained_label)
    return metrics(test.sustained_label,baseline.predict_proba(test[FEATURES].to_numpy())[:,1])
