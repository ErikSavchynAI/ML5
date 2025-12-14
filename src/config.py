import os
import torch


class Config:
    TRAIN_PATH = "./input/train.csv"
    TEST_PATH = "./input/to_answer.csv"
    OUTPUT_DIR = "./output"

    MODEL_NAME = "microsoft/mdeberta-v3-base"
    MAX_LEN = 512

    SEED = 42
    N_FOLDS = 5
    EPOCHS = 4
    BATCH_SIZE = 16
    GRAD_ACCUM = 1

    LR = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.05

    NUM_WORKERS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PRECISION = "bf16-mixed"


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
