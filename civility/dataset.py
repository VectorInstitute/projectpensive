import pathlib
import tensorflow as tf

from datasets import load_dataset
from transformers import DistilBertTokenizerFast


class CivilCommentsDataset:
    """
    Loads and processes the `civil_comments` dataset: https://huggingface.co/datasets/civil_comments.
    """

    def __init__(self):

        print("Building dataset...")
        # Load dataset and tokenizer
        self.dataset = load_dataset("civil_comments")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        self.data_path = pathlib.Path("data")

        # Build data sets
        self.train_data = self.load_or_generate_tf_dataset("train")
        self.val_data = self.load_or_generate_tf_dataset("validation")
        self.test_data = self.load_or_generate_tf_dataset("test")

    def load_or_generate_tf_dataset(self, split):
        """
        Build dataset if not already done so, otherwise load it from disk
        """

        if not pathlib.Path.exists(self.data_path / split):

            print(f"Building {split} data...")

            # Generate features and labels
            encodings = self.tokenizer(self.dataset[split]["text"], truncation=True, padding=True)
            features = {x: encodings[x] for x in self.tokenizer.model_input_names}
            labels = self.dataset[split].remove_columns(
                ["text", "identity_attack", "insult", "obscene", "severe_toxicity", "sexual_explicit", "threat"]
            ).to_pandas().to_numpy()

            # Build dataset and save to disk
            data = tf.data.Dataset.from_tensor_slices((
                features,
                labels
            ))
            tf.data.experimental.save(data, str(self.data_path / split))

            return data
        else:
            return tf.data.experimental.load(str(self.data_path / split))
