import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer
import torch
import gc

from .config import Config
from .dataset import get_dataloader
from .model import MilitaryModel


def run_training():
    df = pd.read_csv(Config.TRAIN_PATH)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    oof_preds = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['new_label'])):
        print(f"\n{'=' * 20} FOLD {fold + 1}/{Config.N_FOLDS} {'=' * 20}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_loader = get_dataloader(train_df, tokenizer, is_train=True, shuffle=True)
        val_loader = get_dataloader(val_df, tokenizer, is_train=True, shuffle=False)

        steps_per_epoch = len(train_loader)
        model = MilitaryModel(steps_per_epoch=steps_per_epoch)

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{Config.OUTPUT_DIR}/fold_{fold}",
            filename='best_model',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )

        trainer = pl.Trainer(
            max_epochs=Config.EPOCHS,
            accelerator='gpu',
            devices=1,
            precision=Config.PRECISION,
            accumulate_grad_batches=Config.GRAD_ACCUM,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
            val_check_interval=0.5
        )

        trainer.fit(model, train_loader, val_loader)

        best_model_path = checkpoint_callback.best_model_path
        print(f"Loading best model from {best_model_path}")
        model = MilitaryModel.load_from_checkpoint(best_model_path)
        model.to(Config.DEVICE)
        model.eval()

        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                logits = model(input_ids, attention_mask)
                preds = torch.sigmoid(logits).float().cpu().numpy()
                val_preds.extend(preds)

        oof_preds[val_idx] = np.array(val_preds)

        del model, trainer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    best_f1 = 0
    best_th = 0.5
    for th in np.arange(0.3, 0.7, 0.01):
        score = f1_score(df['new_label'], (oof_preds > th).astype(int))
        if score > best_f1:
            best_f1 = score
            best_th = th

    acc = accuracy_score(df['new_label'], (oof_preds > best_th).astype(int))
    print(f"\n{'=' * 40}")
    print(f"CV F1 Score: {best_f1:.4f} at threshold {best_th:.2f}")
    print(f"CV Accuracy: {acc:.4f}")
    print(f"{'=' * 40}")

    with open(f"{Config.OUTPUT_DIR}/best_threshold.txt", "w") as f:
        f.write(str(best_th))


if __name__ == "__main__":
    run_training()
