import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from .config import Config
from .dataset import get_dataloader
from .model import MilitaryModel
import os


def run_inference():
    test_df = pd.read_csv(Config.TEST_PATH)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    test_loader = get_dataloader(test_df, tokenizer, is_train=False, shuffle=False)

    # Читаємо кращий поріг
    try:
        with open(f"{Config.OUTPUT_DIR}/best_threshold.txt", "r") as f:
            best_th = float(f.read().strip())
    except:
        best_th = 0.5
    print(f"Using threshold: {best_th}")

    fold_preds = []

    # Проганяємо через усі 5 фолдів
    for fold in range(Config.N_FOLDS):
        # Знаходимо файл моделі
        fold_dir = f"{Config.OUTPUT_DIR}/fold_{fold}"
        # Шукаємо .ckpt файл
        ckpt_files = [f for f in os.listdir(fold_dir) if f.endswith('.ckpt')]
        if not ckpt_files:
            continue

        model_path = os.path.join(fold_dir, ckpt_files[0])
        print(f"Predicting with model: {model_path}")

        model = MilitaryModel.load_from_checkpoint(model_path)
        model.to(Config.DEVICE)
        model.eval()

        current_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                logits = model(input_ids, attention_mask)
                preds = torch.sigmoid(logits).cpu().numpy()
                current_preds.extend(preds)

        fold_preds.append(np.array(current_preds))

        del model
        torch.cuda.empty_cache()

    # Ансамбль (середнє арифметичне)
    avg_preds = np.mean(fold_preds, axis=0)
    final_labels = (avg_preds > best_th).astype(int)

    # Збереження
    submission = pd.read_csv("./input/sample_submission.csv")
    submission['new_label'] = final_labels
    submission.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")


if __name__ == "__main__":
    run_inference()
