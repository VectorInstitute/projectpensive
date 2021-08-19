import streamlit as st

from helpers import load_recommender_data, load_recommender_feed, generate_feed, run_classifier, load_civility_data, \
    load_data
from diversity_methods import compare_diversity, compute_diversity, get_similar_comments, greedy_selection, \
    topic_diversification, get_similar_subreddits, calculate_subreddit_quality, subreddit_greedy_selection


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
            type_output = run_classifier(text_input)
            type_output = max(0, type_output)
            type_output = min(type_output, 1)
            if type_output > 0.5:
                st.write(f"This comment is considered **uncivil**, with a toxicity score of {type_output:.3f}.")
            else:
                st.write(f"This comment is considered **civil**, with a toxicity score of {type_output:.3f}.")

    civil_dataset_options = load_civility_data()
    civil_dataset_options = civil_dataset_options.to_list()
    civil_dataset_options.insert(0, None)
    select_text = st.selectbox("Select a phrase to compute its toxicity score...", civil_dataset_options, )
    if select_text is not None:
        with st.spinner("Computing..."):
            select_output = run_classifier(select_text)
            select_output = max(0, select_output)
            select_output = min(select_output, 1)
            if select_output > 0.5:
                st.write(f"This comment is considered **uncivil**, with a toxicity score of {select_output:.3f}.")
            else:
                st.write(f"This comment is considered **civil**, with a toxicity score of {select_output:.3f}.")
                
    # Diversity Filter
    st.subheader("Diversity Filter")
    
    # Comment Recommender
    st.markdown("#### Comment Recommender")
    diversity_algo_options = ("None", "Bounded Greedy Selection", "Topic Diversification")
    st.markdown(
        "Using the `Hugging Face Sentence Transformers` Library, we generated embeddings for each comment. We then "
        "implemented two diverity algorithms described below. Try out both and see how your recommendations change!"
    )
    with st.expander("1. Bounded Greedy Algorithm"):
        st.write(
            """This algorithm seeks to provide a more principled approach to improving diversity by using a “quality” 
            metric to construct the recommendation set, R, in an incremental fashion. The quality of an item is 
            proportional to its similarity to the target query and its relative diversity to the items so far selected. 
            The first item to be selected is the one with the highest similarity to the target query. During each 
            subsequent iteration, the item selected is the one with the highest quality with respect to the set of 
            items selected during the previous iteration.  
            To reduce the complexity, we implemented a bounded version in which we first select the top k items 
            according to their similarity to the target query and apply the Greedy Selection method to these. 
            Reference: [Improving Recommendation Diversity](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.5232&rep=rep1&type=pdf) (2001, Keith Bradley and Barry Smyth)"""
        )
        st.latex(r''' 
        Quality(t,c,R) = Similarity(t,c) * RelDiversity(c, R)
        ''')
        st.latex(r''' 
        RelDiversity(c,R) =
        \left\{
            \begin{array}{ll}
                1 \quad  if \quad  R = \{\}; \\
                \frac{\sum_{i=1..m}(1 - Similarity(c, r_i))} {m} , otherwise
            \end{array}
        \right.
        ''')
        col1, col2, col3 = st.columns([0.75,1,0.75])
        col2.image("images/greedy_pseudo.png")
    
    with st.expander("2. Topic Diversification Algorithm"):
        st.write(
            """This algorithm seeks to tackle the problem of diversifying the topics of books recommended to users; 
            here, we apply it to Reddit comments. We first generate recommended items for the target query (at least 
            5N for the top-N final recommendations). For each N+1 position item, we calculate the ILS (intralist 
            similarity) if this item was part of the top-N list. Then we sort the remaining items in reverse (according
             to the ILS rank) to get their dissimilarity rank. We calculate the new rank for each item as defined in 
             the equation below, with P being the original rank, Pd being the dissimilarity rank and a, b being 
             constants in range [0, 1]. Lastly, we select the top-N items according to the newly calculated rank. 
             Reference: [Improving Recommendation Lists Through Topic Diversification](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.62.9683&rep=rep1&type=pdf) (2005, Cai-Nicolas Ziegler et al.)
             """
        )
        st.latex(r''' 
        r = (a * P) + (b * P_d), \quad a,b \in [0,1]
        ''')
        col1, col2, col3 = st.columns([0.75,1,0.75])
        col2.image("images/topic_pseudo.png")
        
    embedder, dataset, corpus, sarcasm_embeddings, subreddit_embeddings = load_data()
    algorithm = st.selectbox("Choose a diversity algorithm", diversity_algo_options)
    query_comment = st.text_input(label="Provide a comment to get diverse recommendations")
    
    if query_comment not in ["Provide a comment to get diverse recommendations", ""]:
        with st.spinner("Computing..."):
            normal_recommendations = get_similar_comments(embedder, dataset, corpus, sarcasm_embeddings, query_comment, 6)
            avg_dissim_control = compute_diversity(normal_recommendations[1], 6)

            if algorithm == diversity_algo_options[0]:
                pass
            elif algorithm == diversity_algo_options[1]:
                st.write("Recommendations computed with Bounded Greedy Selection:")
                recommendations = greedy_selection(embedder, dataset, corpus, sarcasm_embeddings, query_comment, 6)
                st.table(recommendations[0])
                avg_dissim_algo = compute_diversity(recommendations[1], 6)
                percent_change = compare_diversity(avg_dissim_algo, avg_dissim_control)
                st.text(
                    f"Compared to a normal recommender, this algorithm increased diversity by {percent_change}%"
                )
            else:
                st.write("Recommendations computed with Topic Diversification:")
                recommendations = topic_diversification(embedder, dataset, corpus, sarcasm_embeddings, query_comment, 6)
                st.table(recommendations[0])
                avg_dissim_algo = compute_diversity(recommendations[1], 6)
                percent_change = compare_diversity(avg_dissim_algo, avg_dissim_control)
                st.text(
                    f"Compared to a normal recommender, this algorithm increased diversity by {percent_change}%"
                )
                
    # Subreddit Recommender
    st.markdown("#### Subreddit Recommender")
    subreddit_options = ["None", "AskReddit", "Coronavirus", "antivax", "DebateCommunism", "worldnews", "DebateReligion"]
    chosen_subreddit = st.selectbox("Choose a Subreddit", subreddit_options)

    if chosen_subreddit is not subreddit_options[0]:
        st.write("Recommendations computed with Bounded Greedy Selection:")
        with st.spinner("Computing..."):
            normal_subreddit = get_similar_subreddits(subreddit_embeddings, chosen_subreddit, 10)[1]
            diversity_control = compute_diversity(normal_subreddit, 10)
            sub_recs = subreddit_greedy_selection(subreddit_embeddings, chosen_subreddit, 10)
            st.table(sub_recs[0])
            diversity_algo = compute_diversity(sub_recs[1], 10)
            percent_change = compare_diversity(diversity_algo, diversity_control)
            st.text(
                f"Compared to a normal recommender, this algorithm increased diversity by {percent_change}%"
            )

    # Applying filters to feed
    st.subheader("Putting It All Together")
    st.write(
        "To simulate the experience of Reddit user, we ask you to sign in as a user from the dataset and select a "
        "subreddit. In addition, we ask you to apply your filters and provide the number of posts you wish to see."
    )
    
    show_feed = False
    
    # Feed settings
    popular_users = list(data.author.value_counts().keys())[:100]
    user_name = st.selectbox("Username", popular_users)

    popular_reddits = list(data.subreddit.value_counts().keys())[:100]
    subreddit = st.selectbox("Subreddit", popular_reddits)
    
    num_posts = st.slider("How many posts do you want to see?", 5, 50, value=10)
    
    query = {
        "author": user_name,
        "subreddit": subreddit,
        "num_posts": num_posts
    }
    
    civility_filter = st.checkbox("Apply civility filter")
    diversity_filter = st.checkbox("Apply diversity filter")

    if st.button('Generate Feed'):
        show_feed = True
    
    unaltered_feed = load_recommender_feed(query, data)
    unaltered_feed = unaltered_feed.head(n=query["num_posts"])
    
    if civility_filter:
        civility_threshold = st.slider(
            "We envision online platforms where users have more control over what they see. Use the slider to change "
            "your toxicity tolerance level.",
            0.0,
            1.0,
            step=0.01,
            value=0.5
        )
    if diversity_filter:
        selected_algo = st.radio("Select a Diversity Algorithm", diversity_algo_options, index=0)
        options = unaltered_feed['comment'].to_list()[:num_posts]
        selected_comment = st.selectbox("Choose a query comment", options)
    
    removed_from_feed = None

    # Generate feed
    if show_feed:
        if civility_filter and diversity_filter:
            with st.spinner("Generating civil and diverse feed..."):
                feed, removed_from_feed, percent_change = generate_feed(
                    unaltered_feed,
                    data,
                    query,
                    civility_filter,
                    diversity_filter,
                    civility_threshold,
                    selected_algo,
                    selected_comment,
                    embedder,
                    dataset,
                    corpus,
                    sarcasm_embeddings
                )
            st.text(f"Compared to a normal recommender, this algorithm increased diversity by {percent_change}%")
            if selected_algo == diversity_algo_options[0]:
                feed = feed[["comment", "parent_comment", "author", "subreddit", "toxicity_score"]]
                removed_from_feed = removed_from_feed[
                    ["comment", "parent_comment", "author", "subreddit", "toxicity_score"]
                ]
            elif selected_algo == diversity_algo_options[1]:
                feed = feed[["comment", "parent_comment", "author", "subreddit", "similarity", "toxicity_score"]]
                removed_from_feed = removed_from_feed[
                    ["comment", "parent_comment", "author", "subreddit", "similarity", "toxicity_score"]
                ]
            else:
                feed = feed[["comment", "parent_comment", "author", "subreddit", "rank", "ils_score", "toxicity_score"]]
                removed_from_feed = removed_from_feed[
                    ["comment", "parent_comment", "author", "subreddit", "rank", "ils_score", "toxicity_score"]
                ]
        elif civility_filter:
            with st.spinner("Generating civil feed..."):
                feed, removed_from_feed = generate_feed(
                    unaltered_feed,
                    data,
                    query,
                    civility_filter,
                    diversity_filter,
                    civility_threshold
                )
                feed = feed[["comment", "parent_comment", "author", "subreddit", "toxicity_score"]]
                removed_from_feed = removed_from_feed[
                    ["comment", "parent_comment", "author", "subreddit", "toxicity_score"]
                ]

        elif diversity_filter:
            with st.spinner("Generating diverse feed..."):
                civility_threshold = None
                feed, percent_change = generate_feed(
                    unaltered_feed,
                    data,
                    query,
                    civility_filter,
                    diversity_filter,
                    civility_threshold,
                    selected_algo,
                    selected_comment,
                    embedder,
                    dataset,
                    corpus,
                    sarcasm_embeddings
                )
            st.text(f"Compared to a normal recommender, this algorithm increased diversity by {percent_change}%")
            if selected_algo == diversity_algo_options[0]:
                feed = feed[["comment", "parent_comment", "author", "subreddit"]]
            elif selected_algo == diversity_algo_options[1]:
                feed = feed[["comment", "parent_comment", "author", "subreddit", "similarity"]]
            else:
                feed = feed[["comment", "parent_comment", "author", "subreddit", "rank", "ils_score"]]
        else:
            with st.spinner("Generating feed..."):
                feed = generate_feed(
                    unaltered_feed,
                    data,
                    query,
                    civility_filter,
                    diversity_filter
                )

        st.write("")  # Blank space
        st.write("Here is your recommended feed:")
        st.table(feed)
            
    if removed_from_feed is not None:
        st.write("What was filtered:")
        st.table(removed_from_feed)
