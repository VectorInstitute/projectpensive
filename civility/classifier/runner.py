import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


class CivilCommentsRunner:
    """
    Runs data through model to get predictions.
    """

    def __init__(self, path_to_model):
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(path_to_model)
        self.model.dropout.p = 0
        self.model.add_module(module=torch.nn.Sigmoid(), name="sigmoid")

    def run_model(self, string):
        encoded_input = self.tokenizer(string, truncation=True, padding=True, return_tensors='pt')
        output = self.model(**encoded_input)
        return float(output.logits)


if __name__ == "__main__":
    model = CivilCommentsRunner("results/final_model")
    response = model.run_model("i really do not like you")
    print(response)
