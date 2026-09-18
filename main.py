from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from config import Config
from src.data.dataset import split_windows, select_sensor_features
from src.data.preprocessing import synchronize_and_normalize
from src.data.sliding_window import create_sliding_windows
from src.evaluate import evaluate_model, plot_predictions
from src.train import train_model
from src.utils.metrics import mae, rmse
from src.models.seq2seq_model import Seq2SeqModel


RAW_DATA_PATH = Path("data/raw/sensor_data.csv")
PROCESSED_DIR = Path("data/processed")
X_OUT_PATH = PROCESSED_DIR / "X.npy"
Y_OUT_PATH = PROCESSED_DIR / "y.npy"


def _print_pipeline_info() -> None:
    print(f"Sampling Rate: {Config.SAMPLING_RATE} Hz")
    print(f"Past Window: {Config.PAST_WINDOW_MS:.2f} ms")
    print(f"Future Start: {Config.FUTURE_START_MS:.2f} ms")
    print(f"Future Horizon: {Config.FUTURE_WINDOW_MS:.2f} ms")


def _measure_latency_ms(model: torch.nn.Module, X_test: np.ndarray, n_samples: int = None) -> list[float]:
    model.eval()
    device = Config.DEVICE
    latencies_ms = []

    if n_samples is None:
        n_samples = Config.LATENCY_SAMPLES

    with torch.no_grad():
        warmup_runs = min(Config.LATENCY_WARMUP_RUNS, len(X_test))
        for idx in range(warmup_runs):
            sample = torch.from_numpy(X_test[idx:idx + 1]).float().to(device)
            _ = model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        for idx in range(min(n_samples, len(X_test))):
            sample = torch.from_numpy(X_test[idx:idx + 1]).float().to(device)

            t0 = time.perf_counter()
            for _ in range(Config.LATENCY_REPEAT_RUNS):
                _ = model(sample)

            if device.type == "cuda":
                torch.cuda.synchronize()

            lat_ms = ((time.perf_counter() - t0) * 1000.0) / Config.LATENCY_REPEAT_RUNS
            latencies_ms.append(lat_ms)
            print(f"Latency sample {idx}: {lat_ms:.2f} ms")

    if latencies_ms:
        print(f"Average latency: {np.mean(latencies_ms):.2f} ms")
        print(f"Median latency:  {np.median(latencies_ms):.2f} ms")
        print(f"Max latency:     {np.max(latencies_ms):.2f} ms")

    return latencies_ms


def angle_to_gait_phase(angle_deg: float, delta_deg: float = 0.0) -> str:
    if angle_deg < 15:
        return "Terminal Swing"
    elif angle_deg < 30:
        return "Initial Contact / Loading Response"
    elif angle_deg < 45:
        return "Terminal Stance" if delta_deg > 1.0 else "Mid Stance"
    elif angle_deg < 60:
        return "Pre Swing" if delta_deg > 1.0 else "Terminal Stance"
    else:
        return "Pre Swing"


def infer_gait_phases(angle_series_deg: np.ndarray) -> list[str]:
    flat = np.asarray(angle_series_deg, dtype=np.float32).reshape(-1)
    deltas = np.gradient(flat) if len(flat) > 1 else np.zeros_like(flat)
    return [angle_to_gait_phase(float(angle), float(delta)) for angle, delta in zip(flat, deltas)]


def _print_gait_phase_output(preds_deg: np.ndarray, sample_index: int = 0, max_steps: int = None) -> None:
    print("\nGait-phase interpretation (predicted):")

    if max_steps is None:
        max_steps = Config.FUTURE_STEPS if Config.PRINT_FULL_GAIT_HORIZON else 20

    horizon = min(max_steps, preds_deg.shape[1])
    phases = infer_gait_phases(preds_deg[sample_index, :horizon, 0])

    for h in range(horizon):
        angle = float(preds_deg[sample_index, h, 0])
        phase = phases[h]
        t_ms = (Config.FUTURE_START_STEP + h) * (1000 / Config.SAMPLING_RATE)
        print(f"At {t_ms:.0f} ms: for knee joint angle {angle:.2f}°, gait phase is {phase}")


def _save_predictions_csv(preds: np.ndarray, actual: np.ndarray, out_path: Path) -> None:
    n_samples, horizon, out_dim = preds.shape
    if out_dim != 1:
        raise ValueError(f"Expected output_dim=1, got {out_dim}")

    preds_deg = preds
    actual_deg = actual

    horizon_ms = np.arange(
        Config.FUTURE_START_STEP,
        Config.FUTURE_START_STEP + horizon,
        dtype=np.int32,
    ) * (1000 / Config.SAMPLING_RATE)

    rows = []

    for i in range(n_samples):
        actual_phases = infer_gait_phases(actual_deg[i, :, 0])
        predicted_phases = infer_gait_phases(preds_deg[i, :, 0])

        for h in range(horizon):
            a_deg = float(actual_deg[i, h, 0])
            p_deg = float(preds_deg[i, h, 0])

            rows.append(
                {
                    "sample_index": i,
                    "horizon_step": h,
                    "horizon_ms": float(horizon_ms[h]),
                    "actual_angle_deg": a_deg,
                    "actual_gait_phase": actual_phases[h],
                    "predicted_angle_deg": p_deg,
                    "predicted_gait_phase": predicted_phases[h],
                }
            )

    pd.DataFrame(rows).round(2).to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


def _print_horizon_metrics(actual: np.ndarray, preds: np.ndarray) -> None:
    horizon = actual.shape[1]
    third = horizon // 3

    ranges = [
        ("Early Horizon", slice(0, third)),
        ("Mid Horizon", slice(third, 2 * third)),
        ("Late Horizon", slice(2 * third, horizon)),
    ]

    print("\nPer-horizon metrics:")
    for name, sl in ranges:
        part_actual = actual[:, sl, :]
        part_preds = preds[:, sl, :]
        print(
            f"{name}: "
            f"MAE={mae(part_actual, part_preds):.6f} | "
            f"RMSE={rmse(part_actual, part_preds):.6f}"
        )


def _load_and_preprocess_data(csv_path: Path):
    df = pd.read_csv(csv_path)

    if "joint_angle" not in df.columns:
        raise ValueError("Expected a 'joint_angle' column in sensor_data.csv")

    feature_df = df.drop(columns=["joint_angle"])
    total_expected = Config.IMU_FEATS + Config.EMG_FEATS
    if feature_df.shape[1] < total_expected:
        raise ValueError(
            f"Expected at least {total_expected} feature columns "
            f"({Config.IMU_FEATS} IMU + {Config.EMG_FEATS} EMG), "
            f"but found {feature_df.shape[1]}"
        )

    feature_values = feature_df.to_numpy(dtype=np.float32)
    imu = feature_values[:, :Config.IMU_FEATS]
    emg = feature_values[:, Config.IMU_FEATS:Config.IMU_FEATS + Config.EMG_FEATS]
    labels = df["joint_angle"].to_numpy(dtype=np.float32)

    X, y, _stats = synchronize_and_normalize(
        imu,
        emg,
        labels,
        emg_rectify=Config.EMG_RECTIFY,
        emg_smooth_window=Config.EMG_SMOOTH_WINDOW,
        imu_smooth_window=Config.IMU_SMOOTH_WINDOW,
        label_smooth_window=Config.LABEL_SMOOTH_WINDOW,
        imu_clip_z=Config.IMU_CLIP_Z,
        emg_clip_z=Config.EMG_CLIP_Z,
    )

    return X, y.reshape(-1, 1)


def _build_model_class(use_positional_encoding=True, use_temporal_attention=True, use_feature_attention=True):
    class ExperimentSeq2SeqModel(Seq2SeqModel):
        def __init__(self, **kwargs):
            super().__init__(
                **kwargs,
                use_positional_encoding=use_positional_encoding,
                use_temporal_attention=use_temporal_attention,
                use_feature_attention=use_feature_attention,
            )
    return ExperimentSeq2SeqModel


def _save_figure(fig, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")




def _run_experiment(name, Xw, yw, *, use_imu, use_emg, use_positional_encoding, use_temporal_attention, use_feature_attention):
    print(f"\nExperiment: {name}")

    Xw_selected = select_sensor_features(
        Xw,
        use_imu=use_imu,
        use_emg=use_emg,
        imu_feats=Config.IMU_FEATS,
        emg_feats=Config.EMG_FEATS,
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_windows(
        Xw_selected,
        yw,
        train_ratio=0.7,
        val_ratio=0.15,
    )

    print(f"Train/Val/Test samples: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"Input features used: {X_train.shape[2]}")

    model_class = _build_model_class(
        use_positional_encoding=use_positional_encoding,
        use_temporal_attention=use_temporal_attention,
        use_feature_attention=use_feature_attention,
    )

    model, history = train_model(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        model_class=model_class,
        verbose=Config.VERBOSE_TRAINING,
    )

    preds, actual, metrics = evaluate_model(model, X_test, y_test, Config.DEVICE)

    baseline = y_test[:, 0:1, :]
    baseline = np.repeat(baseline, preds.shape[1], axis=1)

    print(f"Baseline MAE: {mae(actual, baseline):.6f}")
    print(f"Model MAE:    {metrics['mae']:.6f}")
    print(f"Model RMSE:   {metrics['rmse']:.6f}")

    # --- Accuracy & Error ---
    angle_range = actual.max() - actual.min()
    error_percent = (metrics['mae'] / angle_range) * 100
    accuracy = 100 - error_percent

    print(f"Error Percentage: {error_percent:.2f}%")
    print(f"Approx Accuracy: {accuracy:.2f}%")
    print(f"Model RMSE: {metrics['rmse']:.6f}")

    _print_horizon_metrics(actual, preds)
    _measure_latency_ms(model, X_test, n_samples=Config.LATENCY_SAMPLES)

    safe_name = name.lower().replace(" ", "_").replace("+", "plus").replace("/", "_")
    pred_path = PROCESSED_DIR / f"predictions_{safe_name}.csv"
    _save_predictions_csv(preds, actual, pred_path)

    _print_gait_phase_output(preds, sample_index=0)

    output_dir = PROCESSED_DIR / "plots" / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min(Config.PLOT_NUM_SAMPLES, len(preds))):
        fig = plot_predictions(
            preds,
            actual,
            sample_index=idx,
            title=f"{name}: Actual vs Predicted Future Knee Angle",
            show=True,
        )
        _save_figure(fig, output_dir / f"prediction_sample_{idx}.png")


    return {
        "name": name,
        "baseline_mae": mae(actual, baseline),
        "model_mae": metrics["mae"],
        "model_rmse": metrics["rmse"],
        "best_epoch": history.get("best_epoch"),
        "best_val_loss": history.get("best_val_loss"),
    }


def main():
    Config.set_seed()
    _print_pipeline_info()

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {RAW_DATA_PATH}")

    X, y = _load_and_preprocess_data(RAW_DATA_PATH)

    Xw, yw = create_sliding_windows(
        X,
        y,
        past_steps=Config.PAST_STEPS,
        future_steps=Config.FUTURE_STEPS,
        future_start_step=Config.FUTURE_START_STEP,
        stride=1,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(X_OUT_PATH, Xw)
    np.save(Y_OUT_PATH, yw)
    print(f"Saved windows: {X_OUT_PATH}, {Y_OUT_PATH}")

    experiments = [
        {
            "name": "IMU + EMG",
            "use_imu": True,
            "use_emg": True,
            "use_positional_encoding": True,
            "use_temporal_attention": True,
            "use_feature_attention": True,
        },
    ]

    results = []
    for exp in experiments:
        result = _run_experiment(
            exp["name"],
            Xw,
            yw,
            use_imu=exp["use_imu"],
            use_emg=exp["use_emg"],
            use_positional_encoding=exp["use_positional_encoding"],
            use_temporal_attention=exp["use_temporal_attention"],
            use_feature_attention=exp["use_feature_attention"],
        )
        results.append(result)

    print("\nExperiment Summary")
    summary_df = pd.DataFrame(results).sort_values(by="model_mae", ascending=True)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
