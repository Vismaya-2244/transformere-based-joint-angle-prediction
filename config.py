import random
import numpy as np
import torch


class Config:
    SEED = 42

    # Data
    SAMPLING_RATE = 100
    PAST_STEPS = 128
    FUTURE_START_STEP = 2
    FUTURE_STEPS = 60

    PAST_WINDOW_MS = (PAST_STEPS / SAMPLING_RATE) * 1000.0
    FUTURE_WINDOW_MS = (FUTURE_STEPS / SAMPLING_RATE) * 1000.0
    FUTURE_START_MS = (FUTURE_START_STEP / SAMPLING_RATE) * 1000.0

    IMU_FEATS = 24
    EMG_FEATS = 0
    NUM_INPUT_FEATS = IMU_FEATS + EMG_FEATS
    NUM_OUTPUT = 1

    # Sensor selection for experiments
    USE_IMU = True
    USE_EMG = True

    # Preprocessing
    IMU_SMOOTH_WINDOW = 3
    EMG_SMOOTH_WINDOW = 5
    LABEL_SMOOTH_WINDOW = 1
    EMG_RECTIFY = True
    IMU_CLIP_Z = 4.0
    EMG_CLIP_Z = 4.0

    # Model
    D_MODEL = 64
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 2
    NUM_HEADS = 4
    FFN_DIM = 128
    DROPOUT = 0.1

    # Training
    BATCH_SIZE = 128
    EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    CLIP_GRAD_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 10
    USE_HORIZON_WEIGHTED_LOSS = True
    HORIZON_LOSS_END_WEIGHT = 1.5
    VERBOSE_TRAINING = False

    # Scheduler
    USE_SCHEDULER = True
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    MIN_LR = 1e-6

    # Evaluation / Reporting
    PLOT_NUM_SAMPLES = 1
    LATENCY_SAMPLES = 5
    LATENCY_REPEAT_RUNS = 20
    LATENCY_WARMUP_RUNS = 5
    PRINT_FULL_GAIT_HORIZON = True

    # Runtime
    DEVICE = torch.device("cpu")
    NUM_WORKERS = 0
    PIN_MEMORY = torch.cuda.is_available()

    @staticmethod
    def set_seed(seed: int = None):
        if seed is None:
            seed = Config.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
