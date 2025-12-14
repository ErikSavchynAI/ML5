import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from .config import Config


class MilitaryDataset(Dataset):
    def __init__(self, df, tokenizer, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.is_train = is_train

        # Для LLM важливо форматувати промпт правильно, але
        # для SequenceClassification ми просто даємо текст.
        # Можна спробувати додати інструкцію, але поки залишимо "сирий" текст з каналом.
        self.texts = [
            f"Channel: {row['channel_name']} | {row['cleaned_message']}"
            for _, row in df.iterrows()
        ]

        if self.is_train:
            self.labels = df["new_label"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        sample = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.is_train:
            sample["labels"] = torch.tensor(self.labels[item], dtype=torch.float)

        return sample


def get_dataloader(df, tokenizer, is_train=True, shuffle=False):
    # ВАЖЛИВО ДЛЯ LLM: встановлюємо pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = MilitaryDataset(df, tokenizer, is_train)

    return DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=True,
    )
