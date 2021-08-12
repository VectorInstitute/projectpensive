import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import torch

from data import SarcasticDataset
from model import RecommenderEngine


if __name__ == "__main__":
    """
    Trains Recommender Engine on reddit dataset.
    Dataset: https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit
    """

    # Load Data
    data = pd.read_csv("train-balanced-sarcasm-processed.csv")

    # Note that batch_size is essentially doubled to include a negative pair
    batch_size = 15
    data_set = SarcasticDataset(data)
    data_loader = torch.utils.data.DataLoader(data_set, shuffle=True, batch_size=batch_size, num_workers=0)

    # Setup gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define model
    model = RecommenderEngine(data, n_factors=300, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 10
    model.to(device)
    for epoch in range(num_epochs):
        losses = []
        f1s = []
        precisions = []
        recalls = []

        data_loader_iter = iter(data_loader)
        for b in range(len(data_loader_iter)):
            try:
                batch = next(data_loader_iter)
            except TypeError:
                # May be faulty data point
                continue

            # Merge positive and negative cases in batch
            for key in batch.keys():
                if type(batch[key][0]) != torch.Tensor:
                    batch[key] = batch[key][0] + batch[key][1]
                else:
                    batch[key] = torch.cat((batch[key][0], batch[key][1]))

            batch["match_score"] = batch["match_score"].type(torch.float32)

            # Compute loss
            prediction = model(batch["author"], batch["subreddit"], batch["comment"])
            loss = loss_fn(prediction, batch["match_score"].to(device))

            # Compute and track metrics
            metrics = precision_recall_fscore_support(
                prediction.data.cpu() > 0.5, batch["match_score"].cpu(), warn_for=tuple()
            )

            losses.append(loss)
            f1s.append(metrics[2][0])
            precisions.append(metrics[0][0])
            recalls.append(metrics[1][0])

            # Back propagate
            loss.backward()
            optimizer.step()

            if b % 50 == 0:
                loss_avg = torch.mean(torch.Tensor(losses))
                f1_avg = torch.mean(torch.Tensor(f1s))
                precision_avg = torch.mean(torch.Tensor(precisions))
                recall_avg = torch.mean(torch.Tensor(recalls))

                print(
                    f"Batch {b: <9} | Loss: {loss_avg:.3f}, F1: {f1_avg:.3f}, Precision: {precision_avg:.3f}, "
                    f"Recall: {recall_avg:.3f}"
                )

        print(
            f"\nAfter Epoch {epoch + 1: <3} | Loss: {torch.mean(torch.Tensor(losses)):.3f}, "
            f"F1: {torch.mean(torch.Tensor(f1s)):.3f}, Precision: {torch.mean(torch.Tensor(precisions)):.3f},"
            f"Recall: {torch.mean(torch.Tensor(recalls)):.3f}"
        )

    torch.save(model.state_dict(), "final_model")
    print("Training Program Complete")
