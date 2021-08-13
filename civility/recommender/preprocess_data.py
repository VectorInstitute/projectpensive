import pandas as pd


if __name__ == "__main__":

    # Open data
    data = pd.read_csv("train-balanced-sarcasm.csv")

    # Drop unnecessary data points and reorder columns
    data = data.drop(["label", "score", "ups", "downs", "date", "created_utc"], 1)
    data = data[["comment", "parent_comment", "author", "subreddit"]]

    # Remove data points where the comment is 9 words or less
    data["comment"] = data["comment"].astype(str)
    strings = data.comment.values
    string_cond = [len(s.split()) <= 9 for s in strings]
    ids_to_drop = [index for index, cond in enumerate(string_cond) if cond]
    data = data.drop(ids_to_drop)
    data = data.reset_index(drop=True)

    # Remove data points where the author makes less than 4 comments in the dataset
    good_authors = []
    author_value_counts = data.author.value_counts() > 7
    for author, condition in author_value_counts.items():
        if condition:
            good_authors.append(author)
        else:
            # Since author_value_counts is given in descending order, once one author does not meed the condition
            # all following authors will not meed the condition
            break
    data = data.loc[data["author"].isin(good_authors)]

    data.to_csv("train-balanced-sarcasm-processed.csv")
