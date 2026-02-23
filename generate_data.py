import numpy as np
import pandas as pd

# Time axis
t = np.linspace(0, 50, 1000)

# Base signals
data = {
    "imu_x": np.sin(t),
    "imu_y": np.cos(t),
    "imu_z": np.sin(2*t),
    "gyro_x": np.cos(2*t),
    "gyro_y": np.sin(0.5*t),
    "gyro_z": np.cos(0.5*t),
    "emg_1": np.sin(3*t),
    "emg_2": np.cos(3*t),
    "emg_3": np.sin(0.2*t),
    "emg_4": np.cos(0.2*t),
}

df = pd.DataFrame(data)

# Add Gaussian noise (simulate real sensors)
noise_level = 0.05
noise = np.random.normal(0, noise_level, df.shape)

# Add noise only to input signals (not target)
# Simulate sensor dropout
df.iloc[200:220, 0:3] = 0  # IMU failure for short duration

# Create target (joint angle)
df["joint_angle"] = (
    0.5 * df["imu_x"] +
    0.3 * df["imu_z"] +
    0.2 * df["gyro_y"]
)

# Save raw data
df.to_csv("data/raw/sensor_data.csv", index=False)

print("Noisy sensor data saved to data/raw/sensor_data.csv")