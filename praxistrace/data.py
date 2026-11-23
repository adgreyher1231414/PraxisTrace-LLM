"""Synthetic six-wave teacher trajectories. All non-paper parameters are explicit assumptions."""
import numpy as np
import pandas as pd

MECHANISMS = ['agency', 'peer_collaboration', 'mentoring', 'resource_support', 'reflective_regulation']
SITES = ['A', 'B', 'C']
STATE_NAMES = ['belief', 'practice', 'context', 'reflection']


def generate(n=360, waves=6, seed=42, noise=0.15, gamma=0.60):
    if n % 3:
        raise ValueError('n must be divisible by 3 to balance sites')
    if waves < 3:
        raise ValueError('waves must be at least 3')
    rng = np.random.default_rng(seed)
    sites = np.repeat(SITES, n // 3)
    pathways = np.array(['progression'] * int(n * .6) + ['regression'] * int(n * .2) + ['plateau'] * (n - int(n*.6)-int(n*.2)))
    rng.shuffle(pathways)
    # Listed b-p and b-c correlations are reproduced in the initial Gaussian draws.
    corr = np.array([[1, .38, .29, .25], [.38, 1, .28, .18], [.29, .28, 1, .24], [.25, .18, .24, 1]])
    initial = rng.multivariate_normal([3.41, 2.65, 2.6, 2.5],
                                      np.diag([.62, .7, .68, .65]) @ corr @ np.diag([.62, .7, .68, .65]), n)
    bias = {'A': .42, 'B': 0, 'C': -.42}
    records = []
    for i in range(n):
        state = initial[i].copy()
        state[2] += bias[sites[i]]
        mechanism = int(rng.integers(0, 5))
        effect = rng.uniform(.04, .13, size=5)
        effect[mechanism] += .16
        shock_t = int(rng.choice([1,2,3])) if pathways[i]=='regression' and rng.random()<.2 else -1
        for t in range(waves):
            if t:
                prev = state.copy()
                drift = rng.normal(0, noise, 4)
                state[0] += .075 + drift[0]
                if pathways[i] == 'progression':
                    # Gradual convergence of practice towards pedagogical belief.
                    state[1] += max(0, rng.normal(.20, .07)) + .04*(state[0]-state[1])
                elif pathways[i] == 'regression':
                    state[1] += (.14 if t < 3 else -.22) + .02*(state[0]-state[1])
                else:
                    state[1] += rng.normal(0,.025)
                state[1] += drift[1]
                state[2] += drift[2] + .06 * (bias[sites[i]] - state[2] + 2.6)
                state[3] += .055 + drift[3]
                if t == shock_t:
                    state[2] -= .68
                    state[1] -= .70
                if t == shock_t+1 and shock_t >= 0:
                    state[2] += .30
                    state[1] += .28
                state = np.clip(state, 1, 5)
            # The paper defines cosine(W_b*b, W_p*p) but does not release the W matrices.
            # This bounded proxy is deliberately NOT claimed to be the paper's exact alignment.
            alignment = float(np.clip(1 - abs(state[0]-state[1])/2.5, 0, 1))
            mechanisms = np.clip(effect + rng.normal(0,.06,5) + (state[2]-2.5)*.035, 0, 1)
            pd_exposure = float(np.clip(.45 + bias[sites[i]]*.30 + rng.normal(0,.15),0,1))
            evidence = {
                'interview': f'I believe child-centered learning is important; reported belief {state[0]:.2f} out of five.',
                'observation': f'Classroom observation indicates child-centered practice at {state[1]:.2f} out of five.',
                'reflection_text': f'I reflect on my teaching routines; reflection score {state[3]:.2f} out of five.',
                'pd_record': f'Professional learning participation {pd_exposure:.2f}; contextual support {state[2]:.2f} out of five.'
            }
            records.append(dict(teacher_id=i, site=sites[i], pathway=pathways[i], wave=t+1,
                                belief=state[0], practice=state[1], context=state[2], reflection=state[3],
                                pd_exposure=pd_exposure, alignment=alignment, mechanism=mechanism,
                                shock=int(t==shock_t), **{k:float(v) for k,v in zip(MECHANISMS,mechanisms)},
                                **evidence))
    df = pd.DataFrame(records)
    # y_{i,t}: now or later, two consecutive aligned waves. Future outcomes are labels only,
    # never passed as input to a prefix predictor.
    labels = np.zeros(len(df), dtype=np.int64)
    for _, g in df.groupby('teacher_id', sort=False):
        aligned = g.alignment.to_numpy() >= gamma
        for pos, rowid in enumerate(g.index):
            labels[rowid] = int(any(aligned[j] and aligned[j+1] for j in range(pos,len(aligned)-1)))
    df['sustained_label'] = labels
    return df


def save(df, path):
    df.to_csv(path, index=False)
