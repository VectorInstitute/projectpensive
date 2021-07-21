import pytorch_lightning as pl
import torch
from transformers import DistilBertForSequenceClassification


class SaveCallback(pl.Callback):
    """
    Custom callback to save with transformer library, not pytorch lightning, on_train_epoch_end
    """

    def on_epoch_end(self, trainer, pl_module):
        if trainer.training:
            pl_module.bert.save_pretrained(f"results/checkpoints/epoch-{trainer.current_epoch}")


class CivilCommentsModel(pl.LightningModule):
    """
    Builds a Civil Comments classifier, on top of PyTorch Lightning.
    """

    def __init__(self):
        super().__init__()

        # Bert model
        self.bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

        # Model mods
        self.bert.dropout.p = 0
        self.bert.add_module(module=torch.nn.Sigmoid(), name="sigmoid")
        for param in self.bert.base_model.parameters():
            param.requires_grad = False

        # Model checkpoint callback
        self.checkpoint_callback = SaveCallback()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.bert.parameters(), lr=1e-3)

    def forward(self, batch):
        outputs = self.bert(**batch)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss

        return loss

    def validation_step(self, batch, batch_idx):
        # Run step and get loss
        outputs = self(batch)
        loss = outputs.loss

        # Compute val metrics
        labels = batch["labels"]
        accuracy = torch.mean(torch.eq(outputs.logits.transpose(0, 1) > 0.5, labels > 0.5).float())

        # Log metrics
        self.log("val_loss_on_epoch_start", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_accuracy_on_epoch_start", accuracy, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss, accuracy

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        return loss
