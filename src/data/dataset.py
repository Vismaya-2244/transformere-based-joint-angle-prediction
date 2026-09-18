import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


class TimeSeriesDataset(Dataset):
    """
    Dataset for seq-to-seq time-series windows.
    X_windows: (N, past_steps, input_feats)
    y_windows: (N, future_steps, output_feats)
    """

    def __init__(self, X_windows, y_windows):
        if len(X_windows) != len(y_windows):
            raise ValueError(
                f"Mismatch between X and y samples: {len(X_windows)} vs {len(y_windows)}"
            )

        X_windows = np.asarray(X_windows)
        y_windows = np.asarray(y_windows)

        if X_windows.ndim != 3:
            raise ValueError(f"X_windows must be 3D (N, T_in, F_in), got {X_windows.shape}")
        if y_windows.ndim != 3:
            raise ValueError(f"y_windows must be 3D (N, T_out, F_out), got {y_windows.shape}")

        self.X = torch.tensor(X_windows, dtype=torch.float32)
        self.y = torch.tensor(y_windows, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def select_sensor_features(
    X_windows,
    use_imu=True,
    use_emg=True,
    imu_feats=Config.IMU_FEATS,
    emg_feats=Config.EMG_FEATS,
):
    """
    Select feature subsets from X_windows of shape (N, T, F).

    Assumes feature order:
      [IMU features | EMG features]
    """
    X_windows = np.asarray(X_windows)

    if X_windows.ndim != 3:
        raise ValueError(f"X_windows must be 3D (N, T, F), got {X_windows.shape}")

    if not use_imu and not use_emg:
        raise ValueError("At least one of use_imu or use_emg must be True.")

    expected_feats = imu_feats + emg_feats
    if X_windows.shape[2] < expected_feats:
        raise ValueError(
            f"Expected at least {expected_feats} input features, got {X_windows.shape[2]}"
        )

    parts = []
    if use_imu:
        parts.append(X_windows[:, :, :imu_feats])
    if use_emg:
        parts.append(X_windows[:, :, imu_feats:imu_feats + emg_feats])

    return np.concatenate(parts, axis=2).astype(np.float32)


def split_windows(
    X_windows,
    y_windows,
    train_ratio=0.7,
    val_ratio=0.15,
):
    """
    Chronological split (no shuffle) to avoid temporal leakage.
    """
    n = len(X_windows)
    if n == 0:
        raise ValueError("No samples available for split.")

    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be in [0,1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    X_train, y_train = X_windows[:train_end], y_windows[:train_end]
    X_val, y_val = X_windows[train_end:val_end], y_windows[train_end:val_end]
    X_test, y_test = X_windows[val_end:], y_windows[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_dataloader(
    X_windows,
    y_windows,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    pin_memory=False,
):
    dataset = TimeSeriesDataset(X_windows, y_windows)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
