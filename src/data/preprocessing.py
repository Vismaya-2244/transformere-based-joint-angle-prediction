import numpy as np

def synchronize_and_normalize(imu, emg, labels):
    """
    Synchronizes signals to a common timeline[cite: 6].
    Normalizes to zero mean and unit variance.
    """
    # Simple Z-score normalization
    imu_norm = (imu - np.mean(imu, axis=0)) / (np.std(imu, axis=0) + 1e-8)
    emg_norm = (emg - np.mean(emg, axis=0)) / (np.std(emg, axis=0) + 1e-8)
    
    # Concatenate features: Shape (Total_Time, IMU+EMG)
    X = np.hstack([imu_norm, emg_norm])
    return X, labels