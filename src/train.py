import torch
from torch import nn, optim
from src.data.dataset import get_dataloader
from src.models.seq2seq_model import Seq2SeqModel


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(X, y, epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader(X, y, batch_size=batch_size)

    model = Seq2SeqModel(
        input_dim=X.shape[2],
        d_model=64,
        past_steps=X.shape[1],
        future_steps=y.shape[1]
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    return model