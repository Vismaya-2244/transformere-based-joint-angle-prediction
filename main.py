import numpy as np
import torch
import pandas as pd
import os
import time

from config import Config
from src.data.windowing import create_sliding_windows
from src.train import train_model
from src.evaluate import evaluate_model, plot_predictions

def normalize(X):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

def to_angle(y):
    # Map approx [-1,1] → [0,90] degrees
    return (y + 1) * 45

def main():
    print("RUNNING RESEARCH-LEVEL PIPELINE")

    # Time Info
    print(f"Sampling Rate: {Config.SAMPLING_RATE} Hz")
    print(f"Past Window: {Config.PAST_WINDOW_MS:.2f} ms")
    print(f"Future Prediction Horizon: {Config.FUTURE_WINDOW_MS:.2f} ms\n")

    # Load raw data
    df = pd.read_csv("data/raw/sensor_data.csv")

    X = df.drop(columns=["joint_angle"]).values
    y = df["joint_angle"].values.reshape(-1, 1)

    # Normalize data 
    X = normalize(X)

    # Windowing
    Xw, yw = create_sliding_windows(
        X,
        y,
        Config.PAST_STEPS,
        Config.FUTURE_STEPS
    )

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/X.npy", Xw)
    np.save("data/processed/y.npy", yw)

    print("Processed data saved to data/processed/")

    # Train-Test Split 
    split = int(0.8 * len(Xw))

    X_train, X_test = Xw[:split], Xw[split:]
    y_train, y_test = yw[:split], yw[split:]

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Train model
    model = train_model(
        X_train,
        y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )

    # Evaluate on UNSEEN DATA (important)
    device = Config.DEVICE
    preds, actual = evaluate_model(model, X_test, y_test, device)

    # Real-Time Inference Simulation
    print("\n--- Real-Time Inference Simulation ---")

    latencies = []

    for i in range(5):
        sample = torch.tensor(X_test[i:i+1], dtype=torch.float32).to(device)

        start_time = time.time()
        _ = model(sample)
        latency = time.time() - start_time

        latencies.append(latency)
        print(f"Sample {i}: {latency*1000:.2f} ms")

    print(f"Average Latency: {np.mean(latencies)*1000:.2f} ms\n")

    # Convert to angle (degrees)
    preds_angle = to_angle(preds)
    actual_angle = to_angle(actual)

    # Create time steps
    time_steps = [i * (1000 / Config.SAMPLING_RATE) for i in range(len(preds_angle.flatten()))]

    # Save predictions with angle representation
    pred_df = pd.DataFrame({
        "Time (ms)": time_steps,
        "Actual Angle (deg)": actual_angle.flatten(),
        "Predicted Angle (deg)": preds_angle.flatten()
    })

    # Round values for clean display
    pred_df = pred_df.round(2)

    pred_df.to_csv("data/processed/predictions.csv", index=False)

    print("Predictions with angles saved to data/processed/predictions.csv")

    # Baseline Comparison 
    # Baseline: predict last observed value
    baseline = X_test[:, -1, 0].reshape(-1, 1)

    # Repeat baseline across future steps
    baseline = np.repeat(baseline, preds.shape[1], axis=1)
    baseline = baseline.reshape(baseline.shape[0], baseline.shape[1], 1)

    from src.utils.metrics import mae

    print("Baseline MAE:", mae(actual, baseline))

    # Metrics
    from src.utils.metrics import mae, rmse

    print("MAE:", mae(actual, preds))
    print("RMSE:", rmse(actual, preds))

    # Plot
    for i in range(3):
        plot_predictions(preds, actual, sample_index=i)

    from utils.visualization import plot_correct_prediction

    print("\n--- Correct Future Prediction Visualization ---")
    plot_correct_prediction(model)

if __name__ == "__main__":
    main()


