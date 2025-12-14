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
        self.config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1)
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

    def feature_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask)
        embeddings = self.feature_pooling(outputs.last_hidden_state, attention_mask)
        logits = self.fc(embeddings)
        return logits.squeeze(-1)

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

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        preds = torch.sigmoid(logits).float().cpu().numpy()
        labels = labels.cpu().numpy()

        preds = torch.sigmoid(logits).float().cpu().numpy()
        labels = labels.float().cpu().numpy()

    def on_validation_epoch_end(self):
        pass

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
