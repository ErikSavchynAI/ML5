import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
from .config import Config


class MilitaryModel(pl.LightningModule):
    def __init__(self, steps_per_epoch=None):
        super().__init__()
        self.save_hyperparameters()

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
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        preds = torch.sigmoid(logits).float().cpu().numpy()
        labels = labels.cpu().numpy()

        return {'val_loss': loss, 'preds': preds, 'labels': labels}

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.steps_per_epoch * Config.EPOCHS * Config.WARMUP_RATIO),
            num_training_steps=self.steps_per_epoch * Config.EPOCHS
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
