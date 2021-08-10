import streamlit as st

from helpers import load_data, generate_feed, run_classifier


def demo():
    st.header("Demo")

    st.write(
        "Lets introduce you to the dataset! We are working with the **Sarcastic Comments - REDDIT** dataset: "
        "https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit."
    )
    st.write(
        "It consists of a balanced selection of sarcastic/genuine comment replies, made by a variety of users in and "
        "posted in several subreddits. Lets take a look at the data..."
    )

    with st.spinner("Loading data..."):
        data = load_data()
        st.table(data.head(n=3))

    # Civility Filter
    st.subheader("Civility Filter")
    st.write(
        "We leverage the Hugging Face transformer library to train transformer based NLP models on the civil_comments "
        "dataset. A score is assigned to convey the level of civility present in a post."
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

    # Diversity Filter
    st.subheader("Diversity Filter")

    # Applying filters to feed
    st.subheader("Putting It All Together")
    st.write(
        "To simulate the experience of Reddit user, we ask you to sign in as a user from the dataset and select a"
        "subreddit you want to explore."
    )
    st.write(
        "In addition, we ask you to apply your filters and provide the number of posts you wish to see."
    )

    popular_users = list(data.author.value_counts().keys())[:100]
    user_name = st.selectbox("Username", popular_users)

    popular_reddits = list(data.subreddit.value_counts().keys())[:100]
    subreddit = st.selectbox("Subreddit", popular_reddits)

    # Feed settings
    num_posts = st.slider("How many posts do you want to see?", 5, 100)
    civility_filter = st.checkbox("Apply civility filter")
    diversity_filter = st.checkbox("Apply diversity filter")

    if civility_filter:
        st.write(
            "We envision online platforms where users have more control over what they see. Use the slider to change "
            "the tolerance level of toxicity"
        )
        civility_threshold = st.slider("Set your tolerance level", 0.0, 1.0, step=0.01, value=0.5)
    else:
        civility_threshold = None

    st.write("")  # Blank space
    st.write(
        "Here is your recommended feed:"
    )
    query = {
        "user": user_name,
        "subreddit": subreddit,
        "num_posts": num_posts
    }
    feed = None
    removed_from_feed = None

    # Get feed
    with st.spinner("Getting feed..."):
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
                diversity_filter,
                civility_threshold
            )

        st.table(feed)
        if removed_from_feed is not None:
            st.write("What was filtered:")
            st.table(removed_from_feed)
