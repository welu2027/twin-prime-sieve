"""
Train an LSTM to predict the next twin prime gap from a sequence of previous gaps.

This treats twin prime gaps as a time series: given the last `seq_len` gaps,
predict the next gap.

Saves the trained model to data/model_lstm.pt.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import DATA_DIR, MODELS_DIR


class GapLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(gaps: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(gaps) - seq_len):
        X.append(gaps[i : i + seq_len])
        y.append(gaps[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default=str(DATA_DIR / "twin_primes_features.parquet"))
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_parquet(args.inp)
    gaps = df["gap_after"].values[:-1].astype(np.float32)

    # normalize gaps
    gap_mean, gap_std = gaps.mean(), gaps.std()
    gaps_norm = (gaps - gap_mean) / (gap_std + 1e-8)

    X, y = make_sequences(gaps_norm, args.seq_len)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split, :, None], X[split:, :, None]
    y_train, y_test = y[:split], y[split:]

    train_dl = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                          batch_size=args.batch, shuffle=True)
    test_dl  = DataLoader(TensorDataset(torch.tensor(X_test),  torch.tensor(y_test)),
                          batch_size=args.batch)

    model = GapLSTM(1, args.hidden, args.layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.HuberLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                test_loss += criterion(pred, yb).item() * len(xb)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss/len(X_train):.4f} | "
              f"test_loss={test_loss/len(X_test):.4f}")

    out = MODELS_DIR / "model_lstm.pt"
    torch.save({"model_state": model.state_dict(),
                "gap_mean": gap_mean, "gap_std": gap_std,
                "seq_len": args.seq_len, "hidden": args.hidden,
                "layers": args.layers, "dropout": args.dropout}, out)
    print(f"Saved LSTM model to {out}")


if __name__ == "__main__":
    main()
