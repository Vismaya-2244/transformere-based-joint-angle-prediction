import torch

class Config:

    # Data Parameters
    PAST_STEPS = 128
    FUTURE_STEPS = 60

    # Sampling rate
    SAMPLING_RATE = 100  # Hz → 1 step = 10 ms

    PAST_WINDOW_MS = (PAST_STEPS / SAMPLING_RATE) * 1000   # ≈ 1280 ms
    FUTURE_WINDOW_MS = (FUTURE_STEPS / SAMPLING_RATE) * 1000  # ≈ 600 ms

    IMU_FEATS = 6
    EMG_FEATS = 4
    NUM_INPUT_FEATS = IMU_FEATS + EMG_FEATS

    NUM_OUTPUT = 1  # knee angle

    # Model Parameters
    D_MODEL = 64
    NUM_ENCODER_LAYERS = 2
    NUM_HEADS = 4
    FFN_DIM = 128
    DROPOUT = 0.1

    # Training Parameters
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 20

    # Noise Simulation
    NOISE_LEVEL = 0.05  # used in data generation

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")