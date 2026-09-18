from pathlib import Path
import time
import copy
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset


class Config:
    SEED = 42
    SAMPLING_RATE = 100
    PAST_STEPS = 128
    FUTURE_START_STEP = 2
    FUTURE_STEPS = 60

    FREE_ACC_PATH = "data/raw/Sensor Free Acceleration.csv"
    GYRO_PATH = "data/raw/segment_gyro.csv"
    ORIENT_EULER_PATH = "data/raw/Sensor Orientation - Euler.csv"
    TARGET_PATH = "data/raw/Joint Angles XZY.csv"

    LOWER_LIMB_SEGMENTS = ["Right Upper Leg", "Right Lower Leg", "Right Foot"]
    AXES = ["x", "y", "z"]
    TARGET_COLUMNS = ["Right Knee Flexion/Extension"]
    PHASE_CLASSES = ["Swing", "Initial Contact / Loading Response", "Stance", "Pre Swing"]

    D_MODEL = 96
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 2
    NUM_HEADS = 4
    FFN_DIM = 192
    DROPOUT = 0.1

    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 5e-4
    WEIGHT_DECAY = 1e-5
    CLIP_GRAD_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 12

    PHASE_INPUT_DIM = 2
    PHASE_HIDDEN_DIM = 64
    PHASE_NUM_CLASSES = len(PHASE_CLASSES)
    PHASE_BATCH_SIZE = 64
    PHASE_EPOCHS = 20
    PHASE_LR = 1e-3
    PHASE_WEIGHT_DECAY = 1e-5
    PHASE_PATIENCE = 6

    DEVICE = torch.device("cpu")

    @staticmethod
    def set_seed(seed=None):
        seed = Config.SEED if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


class TimeSeriesDataset(Dataset):
    def __init__(self, X_windows, y_windows):
        self.X = torch.tensor(np.asarray(X_windows), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y_windows), dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloader(X_windows, y_windows, batch_size=32, shuffle=True):
    return DataLoader(TimeSeriesDataset(X_windows, y_windows), batch_size=batch_size, shuffle=shuffle)


def split_windows(X_windows, y_windows, train_ratio=0.7, val_ratio=0.15):
    n = len(X_windows)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return (
        (X_windows[:train_end], y_windows[:train_end]),
        (X_windows[train_end:val_end], y_windows[train_end:val_end]),
        (X_windows[val_end:], y_windows[val_end:]),
    )


def create_sliding_windows(X, y, past_steps, future_steps, future_start_step=0, stride=1):
    Xw, yw = [], []
    last_start = len(X) - past_steps - future_start_step - future_steps + 1

    for start in range(0, last_start, stride):
        past_end = start + past_steps
        future_start = past_end + future_start_step
        future_end = future_start + future_steps
        Xw.append(X[start:past_end])
        yw.append(y[future_start:future_end])

    return np.asarray(Xw, dtype=np.float32), np.asarray(yw, dtype=np.float32)


def _select_segment_axes(df, segments):
    cols = ["Frame"]
    for segment in segments:
        for axis in Config.AXES:
            col = f"{segment} {axis}"
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
            cols.append(col)
    return df[cols].copy()


def _load_feature_file(path, segments):
    return _select_segment_axes(pd.read_csv(path), segments)


def _load_target_file(path, target_cols):
    df = pd.read_csv(path)
    return df[["Frame"] + target_cols].copy()


def _load_and_preprocess_data():
    free_acc = _load_feature_file(Config.FREE_ACC_PATH, Config.LOWER_LIMB_SEGMENTS)
    gyro = _load_feature_file(Config.GYRO_PATH, Config.LOWER_LIMB_SEGMENTS)
    orient = _load_feature_file(Config.ORIENT_EULER_PATH, Config.LOWER_LIMB_SEGMENTS)
    target = _load_target_file(Config.TARGET_PATH, Config.TARGET_COLUMNS)

    df = free_acc.merge(gyro, on="Frame", suffixes=("_acc", "_gyro"))
    df = df.merge(orient, on="Frame")
    df = df.merge(target, on="Frame")
    df = df.sort_values("Frame").reset_index(drop=True)

    y = df[Config.TARGET_COLUMNS].to_numpy(dtype=np.float32)
    X = df.drop(columns=["Frame"] + Config.TARGET_COLUMNS).to_numpy(dtype=np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    X = (X - X.mean(axis=0)) / std

    return X.astype(np.float32), y.astype(np.float32)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        if context is None:
            context = x

        B, Tq, D = x.shape
        Tk = context.shape[1]

        q = self.q_proj(x).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        return self.out_proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory):
        x = self.norm1(x + self.drop(self.self_attn(x)))
        x = self.norm2(x + self.drop(self.cross_attn(x, context=memory)))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, d_model, past_steps, future_steps, output_dim,
                 num_encoder_blocks=3, num_decoder_layers=2, num_heads=4, ffn_dim=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.future_steps = future_steps
        self.input_proj = nn.Linear(input_dim, d_model)
        self.decoder_input = nn.Parameter(torch.zeros(1, future_steps, d_model))
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, ffn_dim, dropout) for _ in range(num_encoder_blocks)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, ffn_dim, dropout) for _ in range(num_decoder_layers)]
        )
        self.output_proj = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _positional_encoding(self, length, d_model, device):
        position = torch.arange(length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(length, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self._positional_encoding(x.size(1), self.d_model, x.device)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)

        decoder_x = self.decoder_input.expand(x.size(0), -1, -1)
        decoder_x = decoder_x + self._positional_encoding(decoder_x.size(1), self.d_model, x.device)

        for layer in self.decoder_layers:
            decoder_x = layer(decoder_x, x)

        return self.output_proj(decoder_x)


class GaitPhaseClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, num_classes=4, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(self.norm(out))


class HorizonWeightedMAELoss(nn.Module):
    def __init__(self, future_steps, end_weight=1.3):
        super().__init__()
        weights = torch.linspace(1.0, end_weight, future_steps, dtype=torch.float32)
        self.register_buffer("weights", weights.view(1, future_steps, 1))

    def forward(self, preds, target):
        return (torch.abs(preds - target) * self.weights.to(preds.device)).mean()


def train_model(X_train, y_train, X_val, y_val):
    train_loader = get_dataloader(X_train, y_train, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(X_val, y_val, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = Seq2SeqModel(
        input_dim=X_train.shape[2],
        d_model=Config.D_MODEL,
        past_steps=X_train.shape[1],
        future_steps=y_train.shape[1],
        output_dim=y_train.shape[2],
        num_encoder_blocks=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        num_heads=Config.NUM_HEADS,
        ffn_dim=Config.FFN_DIM,
        dropout=Config.DROPOUT,
    ).to(Config.DEVICE)

    criterion = HorizonWeightedMAELoss(y_train.shape[1], 1.3)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

    best_state, best_val, best_epoch = None, float("inf"), 0
    patience = 0

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
                val_losses.append(criterion(model(xb), yb).item())

        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val, best_epoch, patience = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= Config.EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_loss": best_val}


def evaluate_model(model, X, y):
    model.eval()
    test_start = time.perf_counter()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32, device=Config.DEVICE)).cpu().numpy()
    test_time_sec = time.perf_counter() - test_start

    return preds, y, {
        "mae": float(np.mean(np.abs(y - preds))),
        "rmse": float(np.sqrt(np.mean((y - preds) ** 2))),
        "test_time_sec": test_time_sec,
    }


def measure_latency_ms(model, X_test, n_samples=5, repeat_runs=20, warmup_runs=5):
    model.eval()
    latencies_ms = []

    with torch.no_grad():
        for idx in range(min(warmup_runs, len(X_test))):
            sample = torch.from_numpy(X_test[idx:idx + 1]).float().to(Config.DEVICE)
            _ = model(sample)

        for idx in range(min(n_samples, len(X_test))):
            sample = torch.from_numpy(X_test[idx:idx + 1]).float().to(Config.DEVICE)
            t0 = time.perf_counter()
            for _ in range(repeat_runs):
                _ = model(sample)
            lat_ms = ((time.perf_counter() - t0) * 1000.0) / repeat_runs
            latencies_ms.append(lat_ms)

    return latencies_ms


def plot_predictions(preds, actual, sample_index=0):
    pred_seq = preds[sample_index, :, 0]
    actual_seq = actual[sample_index, :, 0]
    horizon_ms = (np.arange(len(pred_seq)) + Config.FUTURE_START_STEP) * (1000 / Config.SAMPLING_RATE)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(horizon_ms, actual_seq, label="Actual", linewidth=2)
    ax.plot(horizon_ms, pred_seq, label="Predicted", linestyle="--", linewidth=2)
    ax.set_title("Actual vs Predicted Right Knee Flexion/Extension")
    ax.set_xlabel("Future Horizon (ms)")
    ax.set_ylabel("Angle (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def angle_to_gait_phase(angle_deg, delta_deg=0.0):
    if angle_deg < 15:
        return "Swing"
    elif angle_deg < 30:
        return "Initial Contact / Loading Response"
    elif angle_deg < 60:
        return "Pre Swing" if delta_deg > 1.0 else "Stance"
    else:
        return "Pre Swing"


def infer_gait_phases(angle_series_deg):
    flat = np.asarray(angle_series_deg, dtype=np.float32).reshape(-1)
    deltas = np.gradient(flat) if len(flat) > 1 else np.zeros_like(flat)
    return [angle_to_gait_phase(float(a), float(d)) for a, d in zip(flat, deltas)]


def phase_maps():
    p2i = {name: idx for idx, name in enumerate(Config.PHASE_CLASSES)}
    i2p = {idx: name for idx, name in enumerate(Config.PHASE_CLASSES)}
    return p2i, i2p


def angles_to_phase_ids(angle_windows):
    p2i, _ = phase_maps()
    labels = np.zeros((angle_windows.shape[0], angle_windows.shape[1]), dtype=np.int64)

    for i in range(angle_windows.shape[0]):
        phases = infer_gait_phases(angle_windows[i, :, 0])
        labels[i] = [p2i[p] for p in phases]

    return labels


def prepare_phase_inputs(angle_windows):
    angle_windows = angle_windows.astype(np.float32)
    delta_windows = np.gradient(angle_windows[:, :, 0], axis=1).astype(np.float32)

    angle_mean = float(angle_windows.mean())
    angle_std = float(max(angle_windows.std(), 1e-8))
    delta_mean = float(delta_windows.mean())
    delta_std = float(max(delta_windows.std(), 1e-8))

    angle_norm = (angle_windows[:, :, 0] - angle_mean) / angle_std
    delta_norm = (delta_windows - delta_mean) / delta_std

    X_phase = np.stack([angle_norm, delta_norm], axis=-1).astype(np.float32)
    stats = {
        "angle_mean": angle_mean,
        "angle_std": angle_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
    }
    return X_phase, stats


def apply_phase_input_norm(angle_windows, stats):
    angle_windows = angle_windows.astype(np.float32)
    delta_windows = np.gradient(angle_windows[:, :, 0], axis=1).astype(np.float32)

    angle_norm = (angle_windows[:, :, 0] - stats["angle_mean"]) / max(stats["angle_std"], 1e-8)
    delta_norm = (delta_windows - stats["delta_mean"]) / max(stats["delta_std"], 1e-8)

    return np.stack([angle_norm, delta_norm], axis=-1).astype(np.float32)


def train_phase_classifier(X_train_phase, y_train_phase, X_val_phase, y_val_phase):
    train_ds = TensorDataset(
        torch.tensor(X_train_phase, dtype=torch.float32),
        torch.tensor(y_train_phase, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_phase, dtype=torch.float32),
        torch.tensor(y_val_phase, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=Config.PHASE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.PHASE_BATCH_SIZE, shuffle=False)

    model = GaitPhaseClassifier(
        input_dim=Config.PHASE_INPUT_DIM,
        hidden_dim=Config.PHASE_HIDDEN_DIM,
        num_classes=Config.PHASE_NUM_CLASSES,
    ).to(Config.DEVICE)

    flat_labels = y_train_phase.reshape(-1)
    counts = np.bincount(flat_labels, minlength=Config.PHASE_NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    weights = torch.tensor(weights, dtype=torch.float32, device=Config.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=Config.PHASE_LR, weight_decay=Config.PHASE_WEIGHT_DECAY)

    best_state = None
    best_val = float("inf")
    patience = 0

    for epoch in range(Config.PHASE_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits.reshape(-1, Config.PHASE_NUM_CLASSES), yb.reshape(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
                logits = model(xb)
                loss = criterion(logits.reshape(-1, Config.PHASE_NUM_CLASSES), yb.reshape(-1))
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= Config.PHASE_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


def predict_phase_ids(classifier, X_phase):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(torch.tensor(X_phase, dtype=torch.float32, device=Config.DEVICE))
        return torch.argmax(logits, dim=-1).cpu().numpy()


def phase_accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def print_phase_confusion(y_true, y_pred):
    cm = np.zeros((Config.PHASE_NUM_CLASSES, Config.PHASE_NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true.reshape(-1), y_pred.reshape(-1)):
        cm[int(t), int(p)] += 1

    print("\nPhase Confusion Matrix (rows=true, cols=pred):")
    header = " " * 18 + " | ".join(f"{i:>5d}" for i in range(Config.PHASE_NUM_CLASSES))
    print(header)

    for i, phase_name in enumerate(Config.PHASE_CLASSES):
        row = " | ".join(f"{cm[i, j]:>5d}" for j in range(Config.PHASE_NUM_CLASSES))
        print(f"{phase_name[:18]:18s} {row}")

    print("\nPhase Index Map:")
    for i, phase_name in enumerate(Config.PHASE_CLASSES):
        print(f"{i}: {phase_name}")


def print_gait_mismatches(pred_phase_ids, true_phase_ids, preds, sample_index=0, max_steps=60):
    _, i2p = phase_maps()
    horizon = min(max_steps, pred_phase_ids.shape[1])
    mismatch_count = 0

    print("\nGait-phase mismatches (true vs predicted):")
    for h in range(horizon):
        angle = float(preds[sample_index, h, 0])
        true_phase = i2p[int(true_phase_ids[sample_index, h])]
        pred_phase = i2p[int(pred_phase_ids[sample_index, h])]

        if true_phase != pred_phase:
            mismatch_count += 1
            t_ms = (Config.FUTURE_START_STEP + h) * (1000 / Config.SAMPLING_RATE)
            print(
                f"At {t_ms:.0f} ms: flex/ext angle {angle:.2f}°, "
                f"true phase = {true_phase}, predicted phase = {pred_phase} [MISMATCH]"
            )

    print(f"\nDisplayed-sample mismatches: {mismatch_count}/{horizon}")


PROCESSED_DIR = Path("data/processed")


def print_pipeline_info():
    print(f"Sampling Rate: {Config.SAMPLING_RATE} Hz")
    print(f"Past Window: {(Config.PAST_STEPS / Config.SAMPLING_RATE) * 1000:.2f} ms")
    print(f"Future Start: {(Config.FUTURE_START_STEP / Config.SAMPLING_RATE) * 1000:.2f} ms")
    print(f"Future Horizon: {(Config.FUTURE_STEPS / Config.SAMPLING_RATE) * 1000:.2f} ms")
    print("Device:", Config.DEVICE)


def save_predictions_csv(preds, actual, path):
    df = pd.DataFrame({
        "actual_angle_deg": actual[0, :, 0],
        "predicted_angle_deg": preds[0, :, 0],
    })
    df.to_csv(path, index=False)
    print(f"Predictions saved to {path}")


def run_experiment():
    Config.set_seed()
    print_pipeline_info()

    X, y = _load_and_preprocess_data()

    Xw, yw = create_sliding_windows(
        X,
        y,
        past_steps=Config.PAST_STEPS,
        future_steps=Config.FUTURE_STEPS,
        future_start_step=Config.FUTURE_START_STEP,
        stride=1,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X.npy", Xw)
    np.save(PROCESSED_DIR / "y.npy", yw)
    print("Saved windows: data/processed/X.npy, data/processed/y.npy")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_windows(Xw, yw)

    print("\nExperiment: IMU")
    print(f"Train/Val/Test samples: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"Input features used: {X_train.shape[2]}")
    print(f"Output targets: {Config.TARGET_COLUMNS}")

    train_start = time.perf_counter()
    model, history = train_model(X_train, y_train, X_val, y_val)
    train_time_sec = time.perf_counter() - train_start
    print(f"Training Time: {train_time_sec:.2f} s")

    preds, actual, metrics = evaluate_model(model, X_test, y_test)
    print(f"Testing Time: {metrics['test_time_sec']:.2f} s")

    baseline = y_test[:, 0:1, :]
    baseline = np.repeat(baseline, preds.shape[1], axis=1)

    baseline_mae = float(np.mean(np.abs(actual - baseline)))
    print(f"Baseline MAE: {baseline_mae:.6f}")
    print(f"Model MAE:    {metrics['mae']:.6f}")
    print(f"Model RMSE:   {metrics['rmse']:.6f}")

    angle_range = actual.max() - actual.min()
    error_percent = (metrics["mae"] / max(angle_range, 1e-8)) * 100
    accuracy = 100 - error_percent
    print(f"Error Percentage: {error_percent:.2f}%")
    print(f"Approx Accuracy: {accuracy:.2f}%")

    latencies = measure_latency_ms(model, X_test, n_samples=5, repeat_runs=20, warmup_runs=5)
    print(f"Average latency: {np.mean(latencies):.2f} ms")
    print(f"Median latency:  {np.median(latencies):.2f} ms")
    print(f"Max latency:     {np.max(latencies):.2f} ms")

    y_train_phase = angles_to_phase_ids(y_train)
    y_val_phase = angles_to_phase_ids(y_val)
    y_test_phase = angles_to_phase_ids(y_test)

    X_train_phase, phase_stats = prepare_phase_inputs(y_train)
    X_val_phase = apply_phase_input_norm(y_val, phase_stats)
    X_pred_phase = apply_phase_input_norm(preds, phase_stats)

    phase_clf = train_phase_classifier(X_train_phase, y_train_phase, X_val_phase, y_val_phase)
    pred_phase_ids = predict_phase_ids(phase_clf, X_pred_phase)
    phase_acc = phase_accuracy(y_test_phase, pred_phase_ids)

    print(f"\nPseudo-label Gait Phase Accuracy: {phase_acc * 100:.2f}%")
    print_phase_confusion(y_test_phase, pred_phase_ids)
    print_gait_mismatches(pred_phase_ids, y_test_phase, preds, sample_index=0, max_steps=60)

    save_predictions_csv(preds, actual, PROCESSED_DIR / "predictions_sample0.csv")

    fig = plot_predictions(preds, actual, sample_index=0)
    fig.savefig(PROCESSED_DIR / "prediction_sample_0_flex_ext.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure saved to data/processed/prediction_sample_0_flex_ext.png")

    return {
        "device": str(Config.DEVICE),
        "train_time_sec": train_time_sec,
        "test_time_sec": metrics["test_time_sec"],
        "baseline_mae": baseline_mae,
        "model_mae": metrics["mae"],
        "model_rmse": metrics["rmse"],
        "phase_accuracy": phase_acc,
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
        "avg_latency_ms": float(np.mean(latencies)),
    }


if __name__ == "__main__":
    Config.set_seed()
    print("Device:", Config.DEVICE)
    results = run_experiment()
    print("\nExperiment Summary")
    print(pd.DataFrame([results]).to_string(index=False))
