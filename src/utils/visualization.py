import matplotlib.pyplot as plt
import numpy as np

def plot_correct_prediction(model):
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    sample = 0

    past_input = X[sample, :, 0]
    future_actual = y[sample]
    import torch

    model.eval()

    with torch.no_grad():
        inp = torch.tensor(X[sample:sample+1], dtype=torch.float32)
        future_pred = model(inp).cpu().numpy()[0]

    input_len = len(past_input)
    future_len = len(future_actual)

    t_past = np.arange(0, input_len)
    t_future = np.arange(input_len, input_len + future_len)

    plt.figure(figsize=(10,5))

    plt.plot(t_past, past_input, linestyle='dashed', label='Past Input')
    plt.plot(t_future, future_actual, label='Actual Future')
    plt.plot(t_future, future_pred, label='Predicted Future')

    plt.axvline(x=input_len, linestyle='--', label='Prediction Start')

    plt.legend()
    plt.title("Future Prediction (Correct Visualization)")
    plt.xlabel("Time Step")
    plt.ylabel("Angle")

    plt.show()