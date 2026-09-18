# src/evaluate.py
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import Config


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def to_angle_deg(x):
    return (x + 1.0) * 45.0


def per_step_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=(0, 2))


def per_step_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 2)))


def evaluate_model(model, X, y, device):
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    metrics = {
        "mae": _mae(y, preds),
        "rmse": _rmse(y, preds),
        "per_step_mae": per_step_mae(y, preds),
        "per_step_rmse": per_step_rmse(y, preds),
    }
    return preds, y, metrics


def plot_predictions(preds, actual, sample_index=0, title="", show=True):
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract sample
    pred_seq = preds[sample_index, :, 0]
    actual_seq = actual[sample_index, :, 0]

    # Time axis (ms)
    horizon_ms = np.arange(len(pred_seq)) * (1000 / 100)  # assuming 100Hz

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(horizon_ms, actual_seq, label="Actual", linewidth=2)
    ax.plot(horizon_ms, pred_seq, label="Predicted", linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Future Horizon (ms)")
    ax.set_ylabel("Knee Angle (deg)")
    ax.legend()
    ax.grid(True)

    if show:
        plt.show()

    return fig


def plot_horizon_error(metrics, title="Prediction Error Across Future Horizon", show=False):
    per_step_mae_vals = metrics["per_step_mae"]
    per_step_rmse_vals = metrics["per_step_rmse"]

    horizon_steps = np.arange(
        Config.FUTURE_START_STEP,
        Config.FUTURE_START_STEP + len(per_step_mae_vals),
    )
    horizon_ms = horizon_steps * (1000 / Config.SAMPLING_RATE)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(horizon_ms, per_step_mae_vals, label="Per-step MAE", linewidth=2)
    ax.plot(horizon_ms, per_step_rmse_vals, label="Per-step RMSE", linewidth=2, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Future Horizon (ms)")
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig
