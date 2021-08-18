import random
import torch


class CivilCommentsDataset(torch.utils.data.Dataset):
    """
    Builds split instance of the `civil_comments` dataset: https://huggingface.co/datasets/civil_comments.
    """

    def __init__(self, split, num_data_points, dataset, tokenizer):
        self.split = split
        self.num_data_points = num_data_points
        self.dataset = dataset
        self.tokenizer = tokenizer

        self.encodings, self.labels = self.build_data_split()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def build_data_split(self):
        print(f"Generating {self.num_data_points} data points for {self.split} split...", end="", flush=True)

        civil_idx = []
        uncivil_idx = []
        num_civil = self.num_data_points // 2
        num_uncivil = self.num_data_points // 2

        # Balancing the dataset by grabbing self.num_data_points // 2 civil and uncivil data points
        for i, data in enumerate(self.dataset):
            # If civil data point and civil_idx is not full
            if data["toxicity"] < 0.5 and num_civil > 0:
                civil_idx.append(i)
                num_civil -= 1
            # If uncivil and uncivil_idx is not full
            elif data["toxicity"] > 0.5 and num_uncivil > 0:
                uncivil_idx.append(i)
                num_uncivil -= 1

            # If both lists are full, exit
            if num_civil == 0 and num_uncivil == 0:
                break

        indexes = civil_idx + uncivil_idx
        random.shuffle(indexes)
        encoding = self.tokenizer(self.dataset[indexes]["text"], truncation=True, padding=True)
        label = self.dataset[indexes]["toxicity"]

        print("done")
        return encoding, label
