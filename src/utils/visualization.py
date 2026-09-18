import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_correct_prediction(model, sample=0):
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    model.eval()
    with torch.no_grad():
        inp = torch.tensor(X[sample:sample + 1], dtype=torch.float32)
        future_pred = model(inp).cpu().numpy()[0].squeeze(-1)

    past_input = X[sample, :, 0].astype(float)   # one channel shown for context
    future_actual = y[sample].squeeze(-1).astype(float)

    input_len = len(past_input)
    future_len = len(future_actual)

    t_past = np.arange(0, input_len)
    t_future = np.arange(input_len, input_len + future_len)

    plt.figure(figsize=(10, 5))
    plt.plot(t_past, past_input, linestyle="dashed", label="Past Input (ch0)")
    plt.plot(t_future, future_actual, label="Actual Future")
    plt.plot(t_future, future_pred, label="Predicted Future")
    plt.axvline(x=input_len, linestyle="--", label="Prediction Start")

    plt.legend()
    plt.title("Future Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Angle")
    plt.tight_layout()
    plt.show()