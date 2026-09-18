import os
import numpy as np
import pandas as pd


def main():
    np.random.seed(42)

    # Time axis
    t = np.linspace(0, 50, 1000)

    # Base signals
    data = {
        "imu_x": np.sin(t),
        "imu_y": np.cos(t),
        "imu_z": np.sin(2 * t),
        "gyro_x": np.cos(2 * t),
        "gyro_y": np.sin(0.5 * t),
        "gyro_z": np.cos(0.5 * t),
        "emg_1": np.sin(3 * t),
        "emg_2": np.cos(3 * t),
        "emg_3": np.sin(0.2 * t),
        "emg_4": np.cos(0.2 * t),
    }

    df = pd.DataFrame(data)

    # Add Gaussian noise to sensor channels
    noise_level = 0.05
    sensor_cols = df.columns.tolist()
    noise = np.random.normal(0, noise_level, size=df[sensor_cols].shape)
    df[sensor_cols] = df[sensor_cols] + noise

    # Simulate short IMU dropout
    imu_cols = ["imu_x", "imu_y", "imu_z"]
    df.loc[200:219, imu_cols] = 0.0

    # Create target (joint angle)
    df["joint_angle"] = (
        0.5 * df["imu_x"] +
        0.3 * df["imu_z"] +
        0.2 * df["gyro_y"]
    ).astype(np.float32)

    # Save
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/sensor_data.csv", index=False)
    print("Noisy sensor data saved to data/raw/sensor_data.csv")


if __name__ == "__main__":
    main()