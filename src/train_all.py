"""Train all 82 channels one by one. Run overnight / in background on Colab."""
import subprocess, sys, os, glob
import pandas as pd

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    labels = pd.read_csv("data/labeled_anomalies.csv")
    chan_ids = labels["chan_id"].tolist()
    # Also include train-only channels if any
    for f in sorted(glob.glob("data/train/*.npy")):
        cid = os.path.basename(f).replace(".npy","")
        if cid not in chan_ids: chan_ids.append(cid)
    print(f"Total channels: {len(chan_ids)}")
    for i, cid in enumerate(chan_ids):
        print(f"\n=== [{i+1}/{len(chan_ids)}] {cid} ===")
        subprocess.run([sys.executable, "src/train.py", "--chan_id", cid, "--epochs", str(args.epochs)], check=False)
    print("All done!")
