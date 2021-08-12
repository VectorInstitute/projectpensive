import pandas as pd


if __name__ == "__main__":

    # Open data
    data = pd.read_csv("train-balanced-sarcasm.csv")

    # Drop unnecessary data points and reorder columns
    data = data.drop(["label", "score", "ups", "downs", "date", "created_utc"], 1)
    data = data[["comment", "parent_comment", "author", "subreddit"]]

    # Remove data points where the comment is 3 words or less
    ids_to_drop = []
    for comment in data.comment.items():
        len_comment = len(str(comment[1]).split())
        if len_comment <= 3:
            ids_to_drop.append(comment[0])
    data = data.drop(ids_to_drop)
    data = data.reset_index(drop=True)

    # Remove data points where the author makes less than 2 comments in the dataset
    ids_to_drop = []
    author_value_counts = data.author.value_counts() > 2
    for author, condition in author_value_counts.items():
        if not condition:
            ids_to_drop += data.index[data["author"] == author].tolist()
    data = data.drop(ids_to_drop)
    data = data.reset_index(drop=True)

    data.to_csv("train-balanced-sarcasm-processed.csv")
