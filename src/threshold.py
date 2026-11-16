import numpy as np
import pandas as pd

def smooth_errors(y_true, y_pred, span=30):
    return pd.Series(np.abs(y_true - y_pred)).ewm(span=span).mean().values

def dynamic_threshold(e, zs=np.arange(2.5, 10.5, 0.5)):
    """Paper's non-parametric dynamic threshold. Uses ONLY errors, no labels."""
    best, best_score = e.mean() + zs[-1]*e.std(), -np.inf
    for z in zs:
        eps = e.mean() + z*e.std()
        below, above = e[e < eps], e > eps
        if not above.any(): continue
        n_seq = np.sum(np.diff(above.astype(int)) == 1) + above[0]
        score = ((e.mean()-below.mean())/e.mean() + (e.std()-below.std())/e.std()) / (above.sum() + n_seq**2)
        if score > best_score:
            best, best_score = eps, score
    return best

def to_sequences(flags, offset=250):
    idx = np.flatnonzero(flags)
    if len(idx)==0: return []
    splits = np.split(idx, np.flatnonzero(np.diff(idx)>1)+1)
    return [[int(s[0]+offset), int(s[-1]+offset)] for s in splits]

def prune_sequences(seqs, errors, offset=250, pct=0.13):
    """Sort by max error desc. If drop <13%, smaller ones are normal."""
    if len(seqs) <= 1: return seqs
    def max_err(s):
        a, b = s[0]-offset, s[1]-offset
        return errors[max(0,a):max(b+1, a+1)].max() if b>a else 0
    scored = sorted(seqs, key=max_err, reverse=True)
    keep = [scored[0]]
    for i in range(1, len(scored)):
        prev = max_err(scored[i-1]); cur = max_err(scored[i])
        if prev>0 and (prev-cur)/prev < pct:
            break
        keep.append(scored[i])
    return keep

def score_events(pred_seqs, true_seqs):
    hit = lambda a,b: a[0]<=b[1] and b[0]<=a[1]
    tp = sum(any(hit(p,t) for p in pred_seqs) for t in true_seqs)
    fp = sum(not any(hit(p,t) for t in true_seqs) for p in pred_seqs)
    return tp, fp, len(true_seqs)-tp
