import streamlit as st
import pandas as pd
import torch
from helpers import load_recommender_data, generate_feed, run_classifier, load_civility_data, load_data
from diversity_methods import *


def demo():
    st.header("Demo")

    st.write(
        "As previously noted, we are working with the `Sarcastic Comments - REDDIT` dataset. These comments are "
        "provided as options to the Recommender as it generates a social media feed. Lets take a look at some "
        "preprocessed data..."
    )
    
    with st.spinner("Loading data..."):
        data = load_recommender_data()
        st.table(data.head(n=3))

    # Civility Filter
    st.subheader("Civility Filter")
    st.write(
        "We leverage the `Hugging Face transformer` library to train transformer based NLP models on the "
        "`civil_comments` dataset. A score is assigned to convey the level of civility present in a post."
    )
    st.write(
        "To try out the civility classifier, write your own comments, or select from some examples from the dataset."
    )

    text_input = st.text_input(label="Provide a comment to compute its toxicity score...")
    if text_input not in ["Provide a comment to compute its toxicity score...", ""]:
        with st.spinner("Computing..."):
            output = run_classifier(text_input)
            if output > 0.5:
                st.write(f"This comment is considered **uncivil**, with a toxicity score of {output:.3f}.")
            else:
                st.write(f"This comment is considered **civil**, with a toxicity score of {output:.3f}.")

    civil_dataset_options = load_civility_data()
    select_text = st.selectbox("Select a phrase to compute its toxicity score...", civil_dataset_options)
    if select_text:
        with st.spinner("Computing..."):
            output = run_classifier(text_input)
            if output > 0.5:
                st.write(f"This comment is considered **uncivil**, with a toxicity score of {output:.3f}.")
            else:
                st.write(f"This comment is considered **civil**, with a toxicity score of {output:.3f}.")

    # Diversity Filter
    st.subheader("Diversity Filter")
    diversity_algo_options = ("None", "Bounded Greedy Selection", "Topic Diversification")
    st.markdown("Using the HuggingFace Sentence Transformers Library, we generated embeddings for each comment. We then implemented two diverity algorithms described below. Try out both and see how your recommendations change!")
    with st.expander("1. Bounded Greedy Algorithm"):
        st.markdown("""This algorithm seeks to provide a more principled approach to improving diversity by using a “quality” metric to construct the recommendation set, R, in an incremental fashion. The quality of an item is proportional to its similarity to the target query and its relative diversity to the items so far selected. The first item to be selected is the one with the highest similarity to the target query. During each subsequent iteration, the item selected is the one with the highest quality with respect to the set of items selected during the previous iteration.  
    To reduce the complexity, we implemented a bounded version in which we first select the top k items according to their similarity to the target query and apply the Greedy Selection method to these.""")
        col1, col2, col3 = st.columns([1,1,1])
        col2.image("images/greedy_pseudo.png")
    
    with st.expander("2. Topic Diversification Algorithm"):
        st.markdown("""
        This algorithm seeks to tackle the problem of diversifying the topics of books recommended to users; here, we apply it to Reddit comments. We first generate recommended items for the target query (at least 5N for the top-N final recommendations). For each N+1 position item, we calculate the ILS (intralist similarity) if this item was part of the top-N list. Then we sort the remaining items in reverse (according to the ILS rank) to get their dissimilarity rank. We calculate the new rank for each item as r = a ∗ P + b ∗ Pd, with P being the original rank, Pd being the dissimilarity rank and a, b being constants in range [0, 1]. Lastly, we select the top-N items according to the newly calculated rank.
        """)
        col1, col2, col3 = st.columns([1,1,1])
        col2.image("images/topic_pseudo.png")
        
    embedder, dataset, corpus, sarcasm_embeddings = load_data(data)
    
    query_comment = st.text_input(label="Provide a comment to get diverse recommendations")
    algorithm = st.selectbox("Choose a diversity algorithm", diversity_algo_options)
    
    normal_recommendations = get_similar_comments(embedder, dataset, corpus, sarcasm_embeddings, query_comment, 10)
    avg_dissim_control = compute_diversity(normal_recommendations[1], 10)
    
    if query_comment not in ["Provide a comment to get diverse recommendations", ""]:
        if algorithm == diversity_algo_options[0]:
            pass
        elif algorithm == diversity_algo_options[1]:
            with st.spinner("Computing..."):
                st.write("Recommendations computed with Bounded Greedy Selection:")
                recommendations = greedy_selection(embedder, dataset, corpus, sarcasm_embeddings, query_comment, 10)
                st.table(recommendations[0])
                avg_dissim_algo = compute_diversity(recommendations[1], 10)
                percent_change = compare_diversity(avg_dissim_algo, avg_dissim_control)
                st.text("Compared to a normal recommender, this algorithm increased diversity by " + 
                         str(percent_change) + "%")
                
        else:
            with st.spinner("Computing..."):
                st.write("Recommendations computed with Topic Diversification:")
                recommendations = topic_diversification(embedder, dataset, corpus, sarcasm_embeddings, query_comment, 10)
                st.table(recommendations[0])
                avg_dissim_algo = compute_diversity(recommendations[1], 10)
                percent_change = compare_diversity(avg_dissim_algo, avg_dissim_control)
                st.text("Compared to a normal recommender, this algorithm increased diversity by " + 
                         str(percent_change) + "%")


    # Applying filters to feed
    st.subheader("Putting It All Together")
    st.write("To simulate the experience of Reddit user, we ask you to sign in as a user from the dataset and select a subreddit you want to     explore.")
    st.write("In addition, we ask you to apply your filters and provide the number of posts you wish to see.")
    
    show_feed = False
    
    # Feed settings
    popular_users = list(data.author.value_counts().keys())[:100]
    user_name = st.selectbox("Username", popular_users)

    popular_reddits = list(data.subreddit.value_counts().keys())[:100]
    subreddit = st.selectbox("Subreddit", popular_reddits)
    
    num_posts = st.slider("How many posts do you want to see?", 5, 100, value=10)
    
    civility_filter = st.checkbox("Apply civility filter")
    diversity_filter = st.checkbox("Apply diversity filter")

    if civility_filter:
        st.write(
            "We envision online platforms where users have more control over what they see. Use the slider to change "
            "the tolerance level of toxicity"
        )
        civility_threshold = st.slider("Set your tolerance level", 0.0, 1.0, step=0.01, value=0.5)
        query_comment = None
    if diversity_filter:
        selected_algo = st.radio("Select a Diversity Algorithm", diversity_algo_options, index=0)
        options = data['comment'].to_list()[:num_posts]
        query_comment = st.selectbox("Choose a query comment", options)
    
    if st.button('Generate Feed'):
            show_feed = True
    
    feed = None
    removed_from_feed = None
    
    query = {
        "user": user_name,
        "subreddit": subreddit,
        "num_posts": num_posts
    }

    # Get feed
    if show_feed == True:
        if civility_filter and diversity_filter:
            raise NotImplementedError("Done by mike and sheen")
        elif civility_filter:
            feed, removed_from_feed = generate_feed(
                data,
                query,
                civility_filter,
                diversity_filter,
                civility_threshold
            )
        elif diversity_filter:
            raise NotImplementedError("Done by sheen")
        else:
            feed = generate_feed(
                data,
                query,
                civility_filter,
                diversity_filter
            )

        st.write("")  # Blank space
        st.write("Here is your recommended feed:")
        with st.spinner("Getting feed..."):
            st.table(feed)
            
    if removed_from_feed is not None:
        st.write("What was filtered:")
        st.table(removed_from_feed)