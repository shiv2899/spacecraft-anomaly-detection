"""Train 1 channel: python src/train.py --chan_id P-1 --epochs 30"""
import argparse, os, json, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from model import TelemetryLSTM
from dataset import make_windows, Scaler
from config import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chan_id", default="P-1")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="checkpoints")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Channel: {args.chan_id}")

    train = np.load(f"{args.data_dir}/train/{args.chan_id}.npy")
    scaler = Scaler().fit(train)
    train_s = scaler.transform(train)

    # Last 15% of train for validation + early stopping
    split = int(len(train_s)*(1-VAL_SPLIT))
    tr, va = train_s[:split], train_s[split:]
    Xtr, ytr = make_windows(tr, WINDOW)
    Xva, yva = make_windows(va, WINDOW)

    train_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)), batch_size=BATCH_SIZE)

    model = TelemetryLSTM(input_dim=train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best = float("inf"); patience = 0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        model.eval(); vloss=0
        with torch.no_grad():
            for xb, yb in val_dl:
                vloss += loss_fn(model(xb.to(device)), yb.to(device)).item()*len(xb)
        vloss/=len(val_dl.dataset)
        print(f"Epoch {epoch+1}/{args.epochs} val_loss={vloss:.5f}")
        if vloss < best:
            best=vloss; patience=0
            torch.save({"model": model.state_dict(), "scaler_mean": scaler.mean, "scaler_std": scaler.std}, f"{args.out_dir}/{args.chan_id}.pth")
        else:
            patience+=1
            if patience>=PATIENCE:
                print("Early stopping"); break
    print(f"Saved -> {args.out_dir}/{args.chan_id}.pth | best val {best:.5f}")

if __name__=="__main__": main()
