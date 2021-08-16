import time

import sklearn.metrics
from datasets import load_dataset
import googleapiclient
from googleapiclient import discovery
from runner import CivilCommentsRunner


class JigsawAPI:

    def __init__(self):
        API_KEY = 'AIzaSyAQfy2kSqkRo7O_j7Zh7jT783OTEREV2m0'
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            cache_discovery=False
        )

    def call(self, request):
        analyze_request = {
            "comment": {"text": f"{request}"},
            "requestedAttributes": {"TOXICITY": {}}
        }
        response = self.client.comments().analyze(body=analyze_request).execute()
        return response


if __name__ == "__main__":
    # Models
    api = JigsawAPI()
    model = CivilCommentsRunner("results/final_model")

    # Dataset
    dataset = load_dataset("civil_comments")

    dataset_is_toxic = []
    jigsaw_is_toxic = []
    model_is_toxic = []

    # Evaluation loop
    for i, comment in enumerate(dataset["test"]):

        if i > 1000:
            break

        if i % 50 == 0:
            print(f"On iteration {i}...")

        text = comment["text"]
        toxicity_score = comment["toxicity"]

        dataset_score = toxicity_score > 0.5
        try:
            jigsaw_score = api.call(text)["attributeScores"]["TOXICITY"]["summaryScore"]["value"] > 0.5
        except googleapiclient.errors.HttpError:
            continue
        model_score = model.run_model(text) > 0.5

        dataset_is_toxic.append(dataset_score)
        jigsaw_is_toxic.append(jigsaw_score)
        model_is_toxic.append(model_score)

        # There is a rate limit of one query/s
        time.sleep(1)

    # Compute metrics
    print("Computing metrics...")
    jigsaw_precision, jigsaw_recall, jigsaw_f1_score, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true=dataset_is_toxic,
        y_pred=jigsaw_is_toxic,
        average="weighted"
    )
    model_precision, model_recall, model_f1_score, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true=dataset_is_toxic,
        y_pred=model_is_toxic,
        average="weighted"
    )

    print("Results of evaluation...")

    print("\nJigsaw:")
    print(f"Precision: {jigsaw_precision:.3f}, Recall: {jigsaw_recall:.3f}, F1: {jigsaw_f1_score:.3f}")

    print("\nModel:")
    print(f"Precision: {model_precision:.3f}, Recall: {model_recall:.3f}, F1: {model_f1_score:.3f}")
