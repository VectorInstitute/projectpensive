import torch

from .model import RecommenderEngine


class RecommenderEngineRunner:
    """
    Runs data through model to get recommendations.
    """

    def __init__(self, path_to_model, data, n_factors):
        # Load model and weights
        self.model = RecommenderEngine(data, n_factors, torch.device("cpu"))
        self.model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))

    def run_model(self, query, data_frame):

        # Add match_score column to data_frame
        data_frame = data_frame.assign(match_score=0.0)

        for comment in data_frame.comment.items():
            # Compute match score
            match_score = self.model.forward(
                [query["author"]],
                [query["subreddit"]],
                [comment[1]]
            )
            data_frame.at[comment[0], "match_score"] = round(float(match_score), 3)

        # Reorder data_frame based on match_score
        data_frame = data_frame.sort_values("match_score", ascending=False)

        # Remove match_score
        data_frame = data_frame[["comment", "parent_comment", "author", "subreddit"]]

        return data_frame

