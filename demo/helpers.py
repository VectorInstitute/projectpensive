from datasets import load_dataset
import pandas as pd
import streamlit as st

from civility.classifier.runner import CivilCommentsRunner


@st.cache(show_spinner=False)
def load_recommender_data():
    data = pd.read_csv("../civility/recommender/train-balanced-sarcasm.csv")
    data = data.drop(["label", "score", "ups", "downs", "date", "created_utc"], 1)
    data = data[["comment", "parent_comment", "author", "subreddit"]]
    return data


@st.cache(show_spinner=False)
def generate_feed(data, query, civility_filter, diversity_filter, civility_threshold):

    # unaltered_feed = get_recommendations(query)
    unaltered_feed = data.head(n=query["num_posts"])
    unaltered_feed = unaltered_feed.assign(toxicity_score=0.0)

    if civility_filter and diversity_filter:
        raise NotImplementedError("Done by mike and sheen")
    elif civility_filter:
        civil_filter = CivilCommentsRunner("../civility/classifier/results/final_model")
        feed = unaltered_feed
        removed_from_feed = pd.DataFrame(columns=feed.columns)

        for i, comment in unaltered_feed.comment.items():
            score = max(0, civil_filter.run_model(comment))
            if score > civility_threshold:
                removed_from_feed = removed_from_feed.append(feed.loc[i])
                removed_from_feed.at[i, "toxicity_score"] = round(score, 3)
                feed = feed.drop(i)
            else:
                feed.at[i, "toxicity_score"] = round(score, 3)

        return feed, removed_from_feed
    elif diversity_filter:
        raise NotImplementedError("Done by sheen")
    # No filter
    else:
        return unaltered_feed


@st.cache(show_spinner=False)
def run_classifier(text_input):
    classifier = CivilCommentsRunner("../civility/classifier/results/final_model")
    return classifier.run_model(text_input)


@st.cache(show_spinner=False)
def load_civility_data():
    data = load_dataset("civil_comments")
    return data["test"].to_pandas().text[:1000]
