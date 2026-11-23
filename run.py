#!/usr/bin/env python3
"""Reproduction runner: python run.py --help"""
import argparse,json,sys
from pathlib import Path
import pandas as pd
import torch
from praxistrace.data import generate
from praxistrace.train import features, split_teachers, train_model, evaluate, baseline, seed_all


def main():
    p=argparse.ArgumentParser(description='PraxisTrace-LLM methodological reproduction (not exact paper reproduction)')
    p.add_argument('--mode',choices=['main','full','generate'],default='main')
    p.add_argument('--out',default='outputs')
    p.add_argument('--data',default=None,help='Optional pre-existing CSV compatible with generator schema')
    p.add_argument('--encoder',choices=['offline','anthropic'],default='offline')
    p.add_argument('--model-name',default='claude-sonnet-4-6')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    p.add_argument('--n',type=int,default=360)
    args=p.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    seed_all(args.seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    df=pd.read_csv(args.data) if args.data else generate(n=args.n,seed=args.seed)
    df.to_csv(out/'synthetic_benchmark.csv',index=False)
    print(f'Dataset: {len(df)} rows, {df.teacher_id.nunique()} teachers, prevalence={df.sustained_label.mean():.3f}',flush=True)
    if args.mode=='generate':return
    # Hosted API is optional and can incur costs. Save encoded inputs for inspection.
    feat=features(df,encoder=args.encoder,model_name=args.model_name,
                  cache_path=str(out/f'encoded_{args.encoder}.csv'))
    splits=split_teachers(feat,args.seed)
    (out/'split_ids.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    model,hist=train_model(feat,splits['train'],splits['val'],seed=args.seed,epochs=args.epochs,hidden=args.hidden,device=device)
    result,preds=evaluate(model,feat,splits['test'],device=device)
    hist.to_csv(out/'training_history.csv',index=False)
    preds.to_csv(out/'test_predictions.csv',index=False)
    torch.save({'state_dict':model.state_dict(),'hidden':args.hidden,'features':list(feat.columns)},out/'model.pt')
    report={'description':'Fresh results on independently generated synthetic data; NOT the original paper results',
            'mode':args.mode,'seed':args.seed,'encoder':args.encoder,'device':device,'test':result,'baseline_static_lr':baseline(feat,splits)}
    print('TEST',json.dumps(result,indent=2),flush=True)
    if args.mode=='full':
        report['ablations']={}
        for label,kw in [('without_graph',{'no_graph':True}),('without_prototypes',{'no_prototypes':True}),
                         ('without_evidence_consistency',{'no_evidence':True}),('without_review_gate',{'no_review':True})]:
            sub,h=train_model(feat,splits['train'],splits['val'],seed=args.seed,epochs=args.epochs,hidden=args.hidden,device=device,**kw)
            r,_=evaluate(sub,feat,splits['test'],device=device)
            report['ablations'][label]=r
            print('ABLATION',label,r,flush=True)
        report['missing_evidence']={}
        for frac in [0,.1,.2,.3,.4,.5]:
            report['missing_evidence'][str(frac)]=evaluate(model,feat,splits['test'],device,frac,args.seed)[0]
        report['cross_site']={}
        for sites_train,site_test in [(['A'],'B'),(['A'],'C'),(['B'],'C'),(['A','B'],'C')]:
            members=feat[feat.site.isin(sites_train)][['teacher_id','site','pathway']].drop_duplicates()
            ids=members.teacher_id.tolist()
            from sklearn.model_selection import train_test_split
            tr,va=train_test_split(ids,test_size=.2,random_state=args.seed)
            sub,h=train_model(feat,tr,va,seed=args.seed,epochs=args.epochs,hidden=args.hidden,device=device)
            target=feat.loc[feat.site==site_test,'teacher_id'].unique().tolist()
            key=''.join(sites_train)+'_to_'+site_test
            report['cross_site'][key]=evaluate(sub,feat,target,device)[0]
            print('TRANSFER',key,report['cross_site'][key],flush=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    try:
        from plots import create_plots
        create_plots(df,hist,report,out)
    except ImportError:
        print('matplotlib unavailable: skipping plots')
    print('Saved to',out.resolve(),flush=True)

if __name__=='__main__':main()
