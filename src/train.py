import copy
import torch
from torch import nn, optim

from config import Config
from src.data.dataset import get_dataloader
from src.models.seq2seq_model import Seq2SeqModel


class HorizonWeightedMAELoss(nn.Module):
    def __init__(self, future_steps: int, end_weight: float = 1.5):
        super().__init__()
        weights = torch.linspace(1.0, end_weight, future_steps, dtype=torch.float32)
        self.register_buffer("weights", weights.view(1, future_steps, 1))

    def forward(self, preds, target):
        weights = self.weights.to(preds.device)
        return (torch.abs(preds - target) * weights).mean()


def run_epoch(model, dataloader, criterion, device, optimizer=None, grad_clip=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if is_train:
                optimizer.zero_grad()

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            if is_train:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    model_class=Seq2SeqModel,
    verbose=True,
):
    Config.set_seed()
    device = Config.DEVICE

    train_loader = get_dataloader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        val_loader = get_dataloader(
            X_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

    model = model_class(
        input_dim=X_train.shape[2],
        d_model=Config.D_MODEL,
        past_steps=X_train.shape[1],
        future_steps=y_train.shape[1],
        output_dim=y_train.shape[2],
        num_encoder_blocks=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        num_heads=Config.NUM_HEADS,
        ffn_dim=Config.FFN_DIM,
        dropout=Config.DROPOUT,
    ).to(device)

    if Config.USE_HORIZON_WEIGHTED_LOSS:
        criterion = HorizonWeightedMAELoss(
            future_steps=y_train.shape[1],
            end_weight=Config.HORIZON_LOSS_END_WEIGHT,
        )
    else:
        criterion = nn.L1Loss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = None
    if Config.USE_SCHEDULER and val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "best_val_loss": None,
        "best_epoch": None,
    }

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=Config.CLIP_GRAD_NORM,
        )
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = run_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
            )
            history["val_loss"].append(val_loss)

            if scheduler is not None:
                scheduler.step(val_loss)

            if verbose:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}, Best val loss: {best_val:.6f}")
                    break
        else:
            if verbose:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_loss"] = best_val if best_state is not None else None
    history["best_epoch"] = best_epoch if best_state is not None else None

    return model, history
