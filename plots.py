from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_plots(df,history,report,out):
    out=Path(out)
    fig,ax=plt.subplots(figsize=(7,4))
    g=df.groupby('wave')[['belief','practice','alignment']].mean()
    for key in g.columns:ax.plot(g.index,g[key],marker='o',label=key)
    ax.set(xlabel='Wave',ylabel='Mean value',title='Generated longitudinal trajectories')
    ax.legend();fig.tight_layout();fig.savefig(out/'longitudinal.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(history.epoch,history.train_loss,label='Training joint loss')
    ax.plot(history.epoch,history.val_bce,label='Validation BCE')
    ax.set(xlabel='Epoch',ylabel='Loss',title='Model training')
    ax.legend();fig.tight_layout();fig.savefig(out/'training.png',dpi=180);plt.close(fig)
    if 'missing_evidence' in report:
        fig,ax=plt.subplots(figsize=(7,4))
        d=report['missing_evidence'];ax.plot([float(k)*100 for k in d],[v['auroc'] for v in d.values()],marker='o')
        ax.set(xlabel='Evidence missing (%)',ylabel='AUROC',title='Missing evidence sensitivity');fig.tight_layout()
        fig.savefig(out/'robustness.png',dpi=180);plt.close(fig)
    if 'ablations' in report:
        fig,ax=plt.subplots(figsize=(9,4))
        names=['full']+list(report['ablations'])
        vals=[report['test']['auroc']]+[v['auroc'] for v in report['ablations'].values()]
        ax.bar(range(len(names)),vals);ax.set_xticks(range(len(names)),names,rotation=25,ha='right')
        ax.set(ylabel='AUROC',title='Ablation results');fig.tight_layout();fig.savefig(out/'ablations.png',dpi=180);plt.close(fig)
