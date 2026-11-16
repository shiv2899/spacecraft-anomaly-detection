# Spacecraft Anomaly Detection

Train an LSTM to predict normal spacecraft telemetry, then flag moments where reality stops matching the prediction.

Built on real NASA telemetry from SMAP satellite and Curiosity rover (MSL), released by NASA JPL.
82 channels (55 SMAP, 27 MSL) with 105 anomalies labeled by mission experts.

## Results

Evaluated on **66 channels**. 17 channels with very short train files were skipped (less than 250 steps for validation, e.g. D-12, T-10, M-7).

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Z-score Baseline (3-sigma) | 0.068 | 0.523 | 0.120 |
| LSTM + Dynamic Threshold + Pruning | **0.689** | 0.488 | **0.571** |

**Breakdown - LSTM:**
- SMAP: Precision **0.733**, Recall 0.524
- MSL: Precision 0.562, Recall 0.391

> Paper (Hundman et al., NASA JPL) reports: **87.5% precision, 80.0% recall** on all 82 channels.
> Gaussian threshold: 91.7% precision but only 62.9% recall.
> MSL is harder: 69.4% recall vs 85.5% on SMAP.

**What this means:** Our LSTM is **10x more precise** than the Z-score baseline (0.689 vs 0.068). Recall is lower than the paper because 17 short channels were skipped and we trained for 50 epochs on CPU. The LSTM still catches real anomalies with very few false alarms, which is what matters for spacecraft - a false alarm can waste a whole downlink pass.

[P-1 Anomaly Detection](results_P-1.png)
*P-1 channel: Blue = true telemetry, Orange = LSTM prediction, Red bands = true anomalies. Where prediction diverges, anomaly is flagged.*

Full per-channel results in `results/per_channel.csv`.

## How it works

1. Model sees last **250 steps** of all columns (telemetry + commands) and predicts next telemetry value
2. Error = | true - predicted |, smoothed with EWMA (span=30)
3. **Dynamic Threshold** (non-parametric, no labels used): try `mean + z*std` for z in 2.5 to 10.0, pick one that removes most extreme errors while flagging fewest points/sequences
4. **Pruning:** sort flagged sequences by max error, if drop from one to next < 13%, treat smaller ones as normal
5. **Event-level scoring:** anomaly counts as found if ANY flagged sequence overlaps it

## Quickstart

```bash
# 1. Clone
git clone https://github.com/shiv2899/spacecraft-anomaly-detection.git
cd spacecraft-anomaly-detection

# 2. Env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Data - download from Kaggle (see data/README.md)
# Put .npy files in data/train/ and data/test/, labeled_anomalies.csv in data/
# Dataset: NASA Anomaly Detection Dataset SMAP & MSL

# 4. Train on 1 channel (fast test)
PYTHONPATH=src python src/train.py --chan_id P-1 --epochs 30

# 5. Train all 82 channels
PYTHONPATH=src python src/train_all.py --epochs 50

# 6. Evaluate
PYTHONPATH=src python src/evaluate.py --results_dir results/

# 7. Plot
PYTHONPATH=src python src/plot_results.py --chan_id P-1
```

## Project Structure
```
spacecraft-anomaly-detection/
├── data/               # ignored by git, .npy + labeled_anomalies.csv
├── results/            # per_channel.csv - per-channel Precision / Recall
├── src/
│   ├── config.py       # all hyperparams
│   ├── dataset.py      # windows, scaling
│   ├── model.py        # LSTM
│   ├── train.py        # train 1 channel
│   ├── train_all.py    # train all 82
│   ├── threshold.py    # smoothing + dynamic threshold + pruning
│   ├── evaluate.py     # event-level scoring
│   └── plot_results.py # telemetry + prediction + threshold plot
├── LICENSE.md
├── README.md
└── requirements.txt
```

## Limitations
- Telemetry from only 2 missions (SMAP + MSL), may not generalize to other spacecraft
- One small LSTM per channel, no cross-channel correlations used
- Threshold is unsupervised, so some contextual anomalies are still missed
- 17 channels with very short train files were skipped in evaluation
