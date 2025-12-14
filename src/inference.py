import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from .config import Config
from .dataset import get_dataloader
from .model import MilitaryModel
import os
from peft import PeftModel, PeftConfig


def run_inference():
    test_df = pd.read_csv(Config.TEST_PATH)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    test_loader = get_dataloader(test_df, tokenizer, is_train=False, shuffle=False)

    try:
        with open(f"{Config.OUTPUT_DIR}/best_threshold.txt", "r") as f:
            best_th = float(f.read().strip())
    except:
        best_th = 0.5
    print(f"Using threshold: {best_th}")

    fold_preds = []

    for fold in range(Config.N_FOLDS):
        fold_dir = f"{Config.OUTPUT_DIR}/fold_{fold}"
        ckpt_files = [f for f in os.listdir(fold_dir) if f.endswith('.ckpt')]
        if not ckpt_files:
            continue

        base_model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map=Config.DEVICE,
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id

        # 2. Накладаємо навчені адаптери
        model = PeftModel.from_pretrained(base_model, fold_dir)
        model.eval()

        current_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch["input_ids"].to(Config.DEVICE)
                attention_mask = batch["attention_mask"].to(Config.DEVICE)
                logits = model(input_ids, attention_mask)
                preds = torch.sigmoid(logits).float().cpu().numpy()
                current_preds.extend(preds)

        fold_preds.append(np.array(current_preds))

        del model
        torch.cuda.empty_cache()

    avg_preds = np.mean(fold_preds, axis=0)
    final_labels = (avg_preds > best_th).astype(int)

    submission = pd.read_csv("./input/sample_submission.csv")
    submission["new_label"] = final_labels
    submission.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")


if __name__ == "__main__":
    run_inference()
