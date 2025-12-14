import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from transformers import AutoTokenizer
import torch
import gc
import os

from .config import Config
from .dataset import get_dataloader
from .model import MilitaryModel


# --- CUSTOM CALLBACK ДЛЯ LORA ---
class SavePeftModelCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Ми не хочемо зберігати весь величезний checkpoint
        # Ми зберігаємо тільки LoRA ваги
        fold_dir = trainer.default_root_dir
        path = os.path.join(fold_dir, "adapter_model")

        # Це збереже adapter_model.bin (100 MB) і adapter_config.json
        pl_module.model.save_pretrained(path)
        print(f"\nSaved LoRA adapters to {path}")

        # Очищуємо checkpoint, щоб Lightning не писав 40GB на диск
        # Залишаємо тільки метадані, якщо треба
        return {}


def run_training():
    df = pd.read_csv(Config.TRAIN_PATH)

    # ФІКС ТОКЕНІЗЕРА
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["new_label"])):
        print(f"\n{'=' * 20} FOLD {fold + 1}/{Config.N_FOLDS} {'=' * 20}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_loader = get_dataloader(train_df, tokenizer, is_train=True, shuffle=True)
        val_loader = get_dataloader(val_df, tokenizer, is_train=True, shuffle=False)

        steps_per_epoch = len(train_loader)
        model = MilitaryModel(steps_per_epoch=steps_per_epoch)

        # ВИКОРИСТОВУЄМО НАШ CALLBACK
        peft_saver = SavePeftModelCallback()

        trainer = pl.Trainer(
            max_epochs=Config.EPOCHS,
            accelerator="gpu",
            devices=1,
            precision=Config.PRECISION,
            accumulate_grad_batches=Config.GRAD_ACCUM,
            callbacks=[peft_saver],  # Ніяких ModelCheckpoint, тільки наш saver
            enable_progress_bar=True,
            val_check_interval=0.5,
            default_root_dir=f"{Config.OUTPUT_DIR}/fold_{fold}",  # Сюди впадуть адаптери
        )

        trainer.fit(model, train_loader, val_loader)

        # --- ІНФЕРЕНС OOF ---
        # Оскільки ми не зберігали .ckpt, модель в пам'яті - це остання версія.
        # Для LLM це ок (вона не перенавчається так жорстко за 2 епохи).

        model.eval()
        model.to(Config.DEVICE)

        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(Config.DEVICE)
                attention_mask = batch["attention_mask"].to(Config.DEVICE)
                logits = model(input_ids, attention_mask)
                preds = torch.sigmoid(logits).float().cpu().numpy()
                val_preds.extend(preds)

        oof_preds[val_idx] = np.array(val_preds)

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    # Підрахунок метрик
    best_f1 = 0
    best_th = 0.5
    for th in np.arange(0.1, 0.9, 0.01):
        score = f1_score(df["new_label"], (oof_preds > th).astype(int))
        if score > best_f1:
            best_f1 = score
            best_th = th

    print(f"CV F1: {best_f1:.4f} at threshold {best_th}")

    with open(f"{Config.OUTPUT_DIR}/best_threshold.txt", "w") as f:
        f.write(str(best_th))


if __name__ == "__main__":
    run_training()
