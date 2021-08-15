import streamlit as st


def next_steps():
    st.header("Next Steps")
    st.write(
        "This section outlines further work for Project Pensive."
    )

    # Civility
    st.subheader("Civility")
    with st.expander("Read more"):
        st.write(
            "If given more time, there are two things we would attempt to improve the civility filter."
        )
        st.write(
            "A boost in model performance would improve the overall quality of the filter. This could be achieved via "
            "more elaborate hyper-parameter optimization or further experimentation of the model architecture. "
            "`Hugging Face` offers a large selection of pretrained models. It is worthwhile to explore and record "
            "performances of architectures other than `DistilBERT`. `GPT-2` and `GPT-Neo` are attractive alternatives."
        )
        st.write(
            "Another valuable improvement would be to increase the number of labels output by the model. The "
            "`civil_comments` dataset provides many labels, not just the toxicity score. Adding more labels increases "
            "the user's control over their feed. These other labels are `severe toxicity`, `threat`, "
            "`identity attack`, `sexually explicit`, `insult`, `obscene`."
        )
        st.write(
            "To provide an example of where more labels would be useful, imagine a user is content with the "
            "occasional swear word but has no desire to see any identity attacks or sexually explicit content. With "
            "this addition, they would have the power to control that."
        )

    # Diversity
    st.subheader("Diversity")
    with st.expander("Read more"):
        st.write("Written by Sheen")

    # Recommender
    st.subheader("Recommender Engine")
    with st.expander("Read more"):
        st.write(
            "There are several ways to improve and increase the robustness of the current Recommender."
        )
        st.write(
            "First improvement is adding more input query structures. Currently, a query consists of an author and "
            "subreddit. This can be extended to support combinations of queries with authors, subreddits, comments, "
            "and others. Functionality to recommend authors and subreddits is another route to take."
        )
        st.write(
            "Another improvement builds more meaningful representations of comments. The implementation shown today "
            "encodes the comments into a lookup table. Even though this scheme works and provides good "
            "recommendations, it would be ideal to compute more meaningful word embeddings. This change would most "
            "likely result in even better comment recommendations."
        )
