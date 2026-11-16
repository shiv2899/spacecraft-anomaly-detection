# All hyperparameters in one place - from Hundman et al. (NASA JPL) paper
WINDOW = 250          # model sees last 250 steps
LSTM_LAYERS = 2
HIDDEN_SIZE = 80
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 50
PATIENCE = 10         # early stopping
VAL_SPLIT = 0.15      # last 15% of train file for validation
EWMA_SPAN = 30        # error smoothing
ZS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
PRUNE_PCT = 0.13      # 13% drop rule
SEED = 42
DEVICE = "cuda"  # auto fallback in train.py if no GPU
