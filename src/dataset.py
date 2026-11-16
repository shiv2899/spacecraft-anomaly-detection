import numpy as np

def make_windows(x, L=250):
    """x: (T, D) -> X: (T-L, L, D), y: (T-L,) = next telemetry value (col 0)"""
    X = np.stack([x[i:i+L] for i in range(len(x)-L)])
    y = x[L:, 0]
    return X.astype(np.float32), y.astype(np.float32)

class Scaler:
    """Fit ONLY on train file. Never fit on test - common mistake in guide."""
    def __init__(self):
        self.mean = None; self.std = None
    def fit(self, x):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + 1e-8
        return self
    def transform(self, x):
        return (x - self.mean) / self.std
    def inverse_col0(self, v):
        return v * self.std[0, 0] + self.mean[0, 0]
