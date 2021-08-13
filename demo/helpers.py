from datasets import load_dataset
import pandas as pd
import streamlit as st
import time
from civility.classifier.runner import CivilCommentsRunner
from diversity_methods import *

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_data(data):
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sarcasm_embeddings = torch.load("data/sarcasm_embeddings.pt", map_location=torch.device('cpu'))
    dataset = pd.read_csv("../civility/recommender/train-balanced-sarcasm.csv")
    dataset = dataset.drop(["label", "score", "ups", "downs", "date", "created_utc"], 1)
    dataset = dataset[["comment", "parent_comment", "author", "subreddit"]]
    corpus = dataset['comment'].to_list()
  
    # Add vector embeddings as column in df
    vectors = []
    for vector in sarcasm_embeddings:
        vectors.append(list(vector.numpy()))
    dataset['vector'] = vectors
    return embedder, dataset, corpus, sarcasm_embeddings

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_recommender_data():
    data = pd.read_csv("../civility/recommender/train-balanced-sarcasm.csv")
    data = data.drop(["label", "score", "ups", "downs", "date", "created_utc"], 1)
    data = data[["comment", "parent_comment", "author", "subreddit"]]
    return data

@st.cache(show_spinner=False, suppress_st_warning=True)
def generate_feed(data, query, civility_filter, diversity_filter, civility_threshold=None, selected_algo=None, selected_comment=None, embedder=None, dataset=None, corpus=None, sarcasm_embeddings=None):

#     unaltered_feed = get_recommendations(query)
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
        with st.spinner("Computing..."):
            n = query["num_posts"]
            normal_recommendations = get_similar_comments(embedder, dataset, corpus, sarcasm_embeddings, selected_comment, n)
            avg_dissim_control = compute_diversity(normal_recommendations[1], n)

            if selected_algo == "None":
                feed = unaltered_feed
                percent_change = 0
            elif selected_algo == "Bounded Greedy Selection":
                st.write("Recommendations computed with Bounded Greedy Selection:")
                recommendations = greedy_selection(embedder, dataset, corpus, sarcasm_embeddings, selected_comment, n)
                avg_dissim_algo = compute_diversity(recommendations[1], n)
                feed = recommendations[0]
                percent_change = compare_diversity(avg_dissim_algo, avg_dissim_control)
            else:
                st.write("Recommendations computed with Topic Diversification:")
                recommendations = topic_diversification(embedder, dataset, corpus, sarcasm_embeddings, selected_comment, n)
                avg_dissim_algo = compute_diversity(recommendations[1], n)
                feed = recommendations[0]
                percent_change = compare_diversity(avg_dissim_algo, avg_dissim_control)
            
            return feed, percent_change
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