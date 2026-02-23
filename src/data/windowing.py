import numpy as np

def create_sliding_windows(X, y, past_steps=128, future_steps=20):
    if len(X) < past_steps + future_steps:
        raise ValueError("Time series too short for given window sizes")

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    num_samples = len(X) - past_steps - future_steps

    X_windows = np.zeros((num_samples, past_steps, X.shape[1]))
    y_windows = np.zeros((num_samples, future_steps, y.shape[1]))

    for i in range(num_samples):
        X_windows[i] = X[i:i+past_steps]
        y_windows[i] = y[i+past_steps:i+past_steps+future_steps]

    return X_windows, y_windows