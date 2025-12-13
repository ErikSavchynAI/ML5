import os
import torch


class Config:
    # Шляхи
    TRAIN_PATH = "./input/train.csv"
    TEST_PATH = "./input/to_answer.csv"
    OUTPUT_DIR = "./output"

    # Модель
    # mdeberta-v3-large - найкращий вибір для mixed language
    MODEL_NAME = "microsoft/mdeberta-v3-large"
    MAX_LEN = 512  # Не ріжемо контекст, пам'ять дозволяє

    # Тренування
    SEED = 42
    N_FOLDS = 5
    EPOCHS = 4  # Мало епох, щоб не перенавчитись на малому датасеті
    BATCH_SIZE = 16  # Для H100 можна і 32, але 16 стабільніше для збіжності
    GRAD_ACCUM = 1

    # Оптимізація
    LR = 1e-5  # Дуже малий LR для файн-тюнінгу
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1

    # Технічні
    NUM_WORKERS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Використовуємо Bfloat16 для H100 (швидше і стабільніше за FP16)
    PRECISION = "bf16-mixed"


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
