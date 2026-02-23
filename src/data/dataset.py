import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X_windows, y_windows):
        assert len(X_windows) == len(y_windows), "Mismatch between X and y samples"

        self.X = torch.tensor(X_windows, dtype=torch.float32)
        self.y = torch.tensor(y_windows, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloader(X_windows, y_windows, batch_size=32, shuffle=True):
    dataset = TimeSeriesDataset(X_windows, y_windows)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)