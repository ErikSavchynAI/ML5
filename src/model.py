import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from .config import Config


class MilitaryModel(pl.LightningModule):
    def __init__(self, steps_per_epoch=None):
        super().__init__()
        self.save_hyperparameters()

        # Конфігурація моделі для класифікації (1 мітка)
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.config.num_labels = 1
        # Важливо для Qwen/Llama: вказати pad_token_id, інакше впаде
        # Qwen не має дефолтного pad_token, беремо eos_token
        self.config.pad_token_id = 151643  # Це EOS токен Qwen2.5

        # Завантажуємо монстра
        # trust_remote_code=True іноді треба, але для Qwen2.5 вже є підтримка в transformers
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            config=self.config,
            torch_dtype=torch.bfloat16,  # Вантажимо одразу в bf16 для економії
            attn_implementation="sdpa",
        )

        # Налаштування LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,  # Ранг адаптерів (чим більше, тим розумніше, але довше)
            lora_alpha=32,  # r * 2
            lora_dropout=0.05,
            # Цільові модулі для Qwen (Attention шари)
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        # Загортаємо модель в LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()  # Подивись в логах, скільки % тренується

        self.criterion = nn.BCEWithLogitsLoss()
        self.steps_per_epoch = steps_per_epoch

    def forward(self, input_ids, attention_mask):
        # LLM повертає об'єкт SequenceClassifierOutput
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits.squeeze(-1)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        preds = torch.sigmoid(logits).float().cpu().numpy()
        labels = labels.float().cpu().numpy()

        return {"val_loss": loss, "preds": preds, "labels": labels}

    def configure_optimizers(self):
        # Для LoRA тренуємо тільки адаптери, тому беремо parameters() від self.model
        optimizer = AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                self.steps_per_epoch * Config.EPOCHS * Config.WARMUP_RATIO
            ),
            num_training_steps=self.steps_per_epoch * Config.EPOCHS,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
