"""Evaluate all channels: LSTM + dynamic threshold + pruning vs z-score baseline."""
import argparse, ast, os, json, numpy as np, pandas as pd, torch
from model import TelemetryLSTM
from dataset import make_windows, Scaler
from threshold import smooth_errors, dynamic_threshold, to_sequences, prune_sequences, score_events
from config import *

def predict_channel(chan_id, data_dir, ckpt_dir):
    ckpt = torch.load(f"{ckpt_dir}/{chan_id}.pth", map_location="cpu", weights_only=False)
    train = np.load(f"{data_dir}/train/{chan_id}.npy")
    test = np.load(f"{data_dir}/test/{chan_id}.npy")
    scaler = Scaler(); scaler.mean=ckpt["scaler_mean"]; scaler.std=ckpt["scaler_std"]
    test_s = scaler.transform(test)
    Xte, yte = make_windows(test_s, WINDOW)
    model = TelemetryLSTM(input_dim=train.shape[1]); model.load_state_dict(ckpt["model"]); model.eval()
    preds=[]
    with torch.no_grad():
        for i in range(0, len(Xte), 1024):
            preds.append(model(torch.from_numpy(Xte[i:i+1024])).numpy())
    pred = np.concatenate(preds)
    return yte, pred, test

def zscore_baseline(train, test, true_seqs):
    mean, std = train[:,0].mean(), train[:,0].std()
    flags = np.abs(test[:,0]-mean) > 3*std
    seqs = to_sequences(flags, offset=0)
    return score_events(seqs, true_seqs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    labels = pd.read_csv(f"{args.data_dir}/labeled_anomalies.csv")
    rows=[]; total_tp=total_fp=total_fn=0; base_tp=base_fp=base_fn=0
    for _, row in labels.iterrows():
        cid = row["chan_id"]
        true_seqs = ast.literal_eval(row["anomaly_sequences"])
        # SKIP if no checkpoint - short train file channels like D-12, T-10
        if not os.path.exists(f"{args.ckpt_dir}/{cid}.pth"):
            print(f"SKIP {cid} - no checkpoint (too short train file)")
            continue
        yte, pred, test_raw = predict_channel(cid, args.data_dir, args.ckpt_dir)
        err = smooth_errors(yte, pred, span=EWMA_SPAN)
        thr = dynamic_threshold(err)
        flags = err > thr
        seqs = to_sequences(flags, offset=WINDOW)
        seqs = prune_sequences(seqs, err, offset=WINDOW)
        tp, fp, fn = score_events(seqs, true_seqs)
        train = np.load(f"{args.data_dir}/train/{cid}.npy")
        btp, bfp, bfn = zscore_baseline(train, test_raw, true_seqs)
        rows.append({"chan_id":cid, "spacecraft":row["spacecraft"], "class":row["class"],
                     "lstm_tp":tp, "lstm_fp":fp, "lstm_fn":fn, "threshold":float(thr), "n_pred_seqs":len(seqs),
                     "base_tp":btp, "base_fp":bfp, "base_fn":bfn})
        total_tp+=tp; total_fp+=fp; total_fn+=fn; base_tp+=btp; base_fp+=bfp; base_fn+=bfn
        print(f"{cid} [{row['spacecraft']}] LSTM TP={tp} FP={fp} FN={fn} | Base TP={btp} FP={bfp} FN={bfn}")
    df=pd.DataFrame(rows); df.to_csv(f"{args.results_dir}/per_channel.csv", index=False)
    def pr(tp,fp,fn):
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f=2*p*r/(p+r) if p+r>0 else 0; return p,r,f
    for name, (tp,fp,fn) in {"LSTM":(total_tp,total_fp,total_fn), "Z-score":(base_tp,base_fp,base_fn)}.items():
        p,r,f=pr(tp,fp,fn); print(f"{name}: Precision={p:.3f} Recall={r:.3f} F1={f:.3f} TP={tp} FP={fp} FN={fn}")
    for col in ["spacecraft","class"]:
        for val, g in df.groupby(col):
            p,r,f=pr(g["lstm_tp"].sum(), g["lstm_fp"].sum(), g["lstm_fn"].sum())
            print(f"LSTM {col}={val}: P={p:.3f} R={r:.3f} TP={g['lstm_tp'].sum()} FP={g['lstm_fp'].sum()} FN={g['lstm_fn'].sum()}")
    print(f"Saved -> {args.results_dir}/per_channel.csv")

if __name__=="__main__": main()
