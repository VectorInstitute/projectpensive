from datasets import load_dataset
import pandas as pd
import streamlit as st

from civility.classifier.runner import CivilCommentsRunner
from diversity_methods import get_similar_comments, greedy_selection, topic_diversification, compute_diversity, compare_diversity


@st.cache(show_spinner=False)
def load_recommender_data():
    data = pd.read_csv("../civility/recommender/train-balanced-sarcasm.csv")
    data = data.drop(["label", "score", "ups", "downs", "date", "created_utc"], 1)
    data = data[["comment", "parent_comment", "author", "subreddit"]]
    return data


@st.cache(show_spinner=False, suppress_st_warning=True)
def generate_feed(data, query, civility_filter, diversity_filter, civility_threshold=None, selected_algo=None, query_comment=None):

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
        n = int(query["num_posts"])
        avg_dissim_control = compute_diversity(get_similar_comments(dataset, corpus, sarcasm_embeddings, query_comment, n)[1], n)
        
        with st.spinner("Getting feed..."):
            if selected_algo == None or selected_algo == "None":
                feed = unaltered_feed
            else:
                if selected_algo == "Bounded Greedy Selection":
                    recommendations = greedy_selection(dataset, corpus, sarcasm_embeddings, query_comment, n)[0]
                    avg_dissim_algo = compute_diversity(greedy_selection(dataset, corpus, sarcasm_embeddings, query_comment, n)[1], n)
                    percent_change = compare_diversity(avg_dissim_algo, avg_dissim_control)
                else: 
                    recommendations = topic_diversification(dataset, corpus, sarcasm_embeddings, query_comment, n)[0]
                    avg_dissim_algo = compute_diversity(topic_diversification(dataset, corpus, sarcasm_embeddings, query_comment, n)[1], n)
                    percent_change = compare_diversity(avg_dissim_algo, avg_dissim_control)
                
                feed = recommendations
                st.text("Compared to a normal recommender, this algorithm increased diversity by " + str(percent_change) + "%")
                
        return feed
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


@st.cache(show_spinner=False, suppress_st_warning=True)
def get_greedy_comments(dataset, corpus, sarcasm_embeddings, query, avg_dissim_control):
    st.write("Recommendations computed with Bounded Greedy Selection:")
    recommendations = greedy_selection(dataset, corpus, sarcasm_embeddings, query, 10)[0]
    st.write(recommendations)
    avg_dissim_algo = compute_diversity(greedy_selection(dataset, corpus, sarcasm_embeddings, query, 10)[1], 10)
    percent_change = compare_diversity(avg_dissim_algo, avg_dissim_control)
    st.text("Compared to a normal recommender, this algorithm increased diversity by " + 
             str(percent_change) + "%")
    
    
@st.cache(show_spinner=False, suppress_st_warning=True)
def get_topic_diversification_comments(dataset, corpus, sarcasm_embeddings, query, avg_dissim_control):
    st.write("Recommendations computed with Topic Diversification:")
    recommendations = topic_diversification(dataset, corpus, sarcasm_embeddings, query, 10)[0]
    st.write(recommendations)
    avg_dissim_algo = compute_diversity(topic_diversification(dataset, corpus, sarcasm_embeddings, query, 10)[1], 10)
    percent_change = compare_diversity(avg_dissim_algo, avg_dissim_control)
    st.text("Compared to a normal recommender, this algorithm increased diversity by " + 
             str(percent_change) + "%")
    
@st.cache(show_spinner=False)
def get_control_diversity(dataset, corpus, sarcasm_embeddings, query):
    avg_dissim_control = compute_diversity(get_similar_comments(dataset, corpus, sarcasm_embeddings, query, 10)[1], 10)
    return avg_dissim_control
    