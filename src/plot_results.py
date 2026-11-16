"""Plot: telemetry + prediction + smoothed error + threshold, true anomaly shaded."""
import argparse, ast, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from model import TelemetryLSTM
from dataset import make_windows, Scaler
from threshold import smooth_errors, dynamic_threshold, to_sequences
from config import *

ap=argparse.ArgumentParser(); ap.add_argument("--chan_id", default="P-1"); ap.add_argument("--data_dir", default="data"); ap.add_argument("--ckpt_dir", default="checkpoints"); args=ap.parse_args()
labels=pd.read_csv(f"{args.data_dir}/labeled_anomalies.csv"); row=labels[labels.chan_id==args.chan_id].iloc[0]
true_seqs=ast.literal_eval(row["anomaly_sequences"])
ckpt=torch.load(f"{args.ckpt_dir}/{args.chan_id}.pth", map_location="cpu", weights_only=False)
test=np.load(f"{args.data_dir}/test/{args.chan_id}.npy")
scaler=Scaler(); scaler.mean=ckpt["scaler_mean"]; scaler.std=ckpt["scaler_std"]
Xte, yte = make_windows(scaler.transform(test), WINDOW)
model=TelemetryLSTM(input_dim=test.shape[1]); model.load_state_dict(ckpt["model"]); model.eval()
with torch.no_grad(): pred=np.concatenate([model(torch.from_numpy(Xte[i:i+1024])).numpy() for i in range(0,len(Xte),1024)])
err=smooth_errors(yte, pred, span=EWMA_SPAN); thr=dynamic_threshold(err)
fig, axes=plt.subplots(2,1, figsize=(14,7), sharex=True)
axes[0].plot(test[:,0], label="Telemetry", lw=0.8)
axes[0].plot(np.arange(WINDOW, len(test)), scaler.inverse_col0(pred), label="LSTM Prediction", lw=0.8)
for s,e in true_seqs: axes[0].axvspan(s,e, color="red", alpha=0.25)
axes[0].set_title(f"{args.chan_id} [{row['spacecraft']}] - Telemetry vs Prediction"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(np.arange(WINDOW, len(test)), err, label="Smoothed Error", lw=0.8)
axes[1].axhline(thr, color="red", ls="--", label=f"Dynamic Threshold={thr:.3f}")
for s,e in true_seqs: axes[1].axvspan(s,e, color="red", alpha=0.25)
axes[1].set_title("Smoothed Error + Threshold"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"results_{args.chan_id}.png", dpi=150); print(f"Saved results_{args.chan_id}.png")
