from datasets import load_dataset
import os
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

from dataset import CivilCommentsDataset
from model import CivilCommentsModel


if __name__ == "__main__":

    # Suppresses output warning
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Load tokenizer and dataset
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    dataset = load_dataset("civil_comments")

    # Process dataset and split
    train_dataset = CivilCommentsDataset("train", 100, dataset["train"], tokenizer)
    val_dataset = CivilCommentsDataset("validation", 100, dataset["validation"], tokenizer)

    # PyTorch dataloaders
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=10, num_workers=0)
    val_data_loader = DataLoader(val_dataset, batch_size=10, num_workers=0)

    # Model and trainer
    model = CivilCommentsModel()
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            gpus=torch.cuda.device_count(),
            max_epochs=20,
            callbacks=[model.checkpoint_callback],
            plugins=DDPPlugin(find_unused_parameters=False)
        )
    else:
        trainer = pl.Trainer(
            max_epochs=20,
            callbacks=[model.checkpoint_callback],
            plugins = DDPPlugin(find_unused_parameters=False)
        )

    # Train and save final model
    trainer.fit(model, train_dataloader=train_data_loader, val_dataloaders=val_data_loader)
    model.bert.save_pretrained("results/final_model")

    print("Training program complete")
