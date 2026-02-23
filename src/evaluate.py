import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.seq2seq_model import Seq2SeqModel

def evaluate_model(model, X, y, device):
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    return preds, y


def plot_predictions(preds, y, sample_index=0):
    """
    preds: (num_samples, future_steps, 1)
    y: same shape
    """

    pred = preds[sample_index].flatten()
    actual = y[sample_index].flatten()

    plt.figure()
    plt.plot(actual, label="Actual")
    plt.plot(pred, label="Predicted")
    plt.title("Actual vs Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()