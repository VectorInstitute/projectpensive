import numpy as np
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

from dataset import CivilCommentsDataset


class CivilCommentsModel:
    """
    Trains distil-bert Hugging Face model on the `civil_comments` dataset
    """

    def __init__(self, num_train_epochs, steps_per_eval, num_training_points="all"):
        # Loading tokenizer and dataset
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.dataset = load_dataset("civil_comments")

        # Build dataset splits
        if num_training_points != "all":
            assert num_training_points <= len(self.dataset["train"]["text"])
        encodings, labels = self.build_data_split("train", num_training_points)
        self.train_dataset = CivilCommentsDataset(encodings, labels)
        encodings, labels = self.build_data_split("validation")
        self.val_dataset = CivilCommentsDataset(encodings, labels)
        encodings, labels = self.build_data_split("validation")
        self.test_dataset = CivilCommentsDataset(encodings, labels)

        # Building model and freezing layers of base
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=1
        )
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        # Building trainer
        self.training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="steps",
            eval_steps=steps_per_eval
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
        )

    def build_data_split(self, split, num_data_points):
        print(f"Generating {num_data_points} data points for {split} split...", end="", flush=True)
        if num_data_points == "all":
            encodings = self.tokenizer(self.dataset[split]["text"], truncation=True, padding=True)
            labels = self.dataset["validation"]["toxicity"]
        else:
            encodings = self.tokenizer(self.dataset[split][0:num_data_points]["text"], truncation=True, padding=True)
            labels = self.dataset["validation"][0:num_data_points]["toxicity"]
        print("done")
        return encodings, labels

    @staticmethod
    def compute_metrics(model_output):
        pred, label = model_output
        mse = np.mean(np.square(pred - label))
        return {"MSE": mse}
