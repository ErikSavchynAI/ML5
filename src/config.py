import os
import torch


class Config:
    TRAIN_PATH = "./input/train.csv"
    TEST_PATH = "./input/to_answer.csv"
    OUTPUT_DIR = "./output"

    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

    MAX_LEN = 1024

    SEED = 42
    N_FOLDS = 3  # Зменшили з 5
    EPOCHS = 2  # Зменшили з 4 (вона вчиться дуже швидко)

    BATCH_SIZE = 4
    GRAD_ACCUM = 4  # Ефективний батч 16

    LR = 2e-4  # Для LoRA потрібен більший LR, ніж для повного тюнінгу
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.05

    NUM_WORKERS = 8
    DEVICE = "cuda"
    PRECISION = "bf16-true"  # H100 це любить


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
