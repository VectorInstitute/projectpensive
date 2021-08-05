import torch
import pandas as pd

from data import SarcasticDataset
from model import RecommenderEngine


if __name__ == "__main__":

    # Load Data
    data = pd.read_csv("train-balanced-sarcasm-small.csv")
    data = data.drop(["label", "date", "created_utc", "parent_comment"], 1)

    # Note that batch_size is essentially doubled to include a negative pair
    batch_size = 10
    data_set = SarcasticDataset(data)
    data_loader = torch.utils.data.DataLoader(data_set, shuffle=True, batch_size=batch_size, num_workers=0)

    # Setup gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define model
    model = RecommenderEngine(data, len(data), n_factors=64, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # TODO: implement subreddit embeddings

    # Training loop
    num_epochs = 5
    model.to(device)
    for epoch in range(num_epochs):
        losses = []
        for b, batch in enumerate(data_loader):

            # Merge positive and negative cases in batch
            for key in batch.keys():
                if type(batch[key][0]) != torch.Tensor:
                    batch[key] = batch[key][0] + batch[key][1]
                else:
                    batch[key] = torch.cat((batch[key][0], batch[key][1]))

            batch["match_score"] = batch["match_score"].type(torch.float32)

            # Compute loss
            prediction = model(batch["author"], batch["comment"])
            loss = loss_fn(prediction, batch["match_score"])
            losses.append(loss)

            # Back propagate
            loss.backward()
            optimizer.step()

            if b % 50 == 0:
                print(f"Batch {b} || Loss: {torch.mean(torch.Tensor(losses))}")
        print(f"Epoch {epoch} || Loss {torch.mean(torch.Tensor(losses))}")

    torch.save(model.state_dict(), "final_model")
    print("Training Program Complete")
