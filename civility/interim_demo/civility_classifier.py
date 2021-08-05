import numpy as np
import pandas as pd
import streamlit as st
from datasets import load_dataset


def civility_classifier():
    st.header("Civility Classifier")

    st.write(
        "We leverage the Hugging Face transformer library to train transformer based NLP models on the civil_comments "
        "dataset. A score is assigned to convey the level of civility present in a post."
    )
    st.write(
        "To try out the civility classifier, write your own comments, or toggle between civil and uncivil feeds."
    )

    # Input box, runs text through model if present
    st.header("Try the Civility Classifier")
    custom_text = st.text_input("Provide a comment to see if it's considered civil by our model")
    if custom_text:
        is_toxic = False

        if is_toxic:
            st.write(f"This comment is considered **uncivil**, with a toxicity score of {0}")
        else:
            st.write(f"This comment is considered **civil**, with a toxicity score of {0}")

    # Custom feed
    st.header("Customize Feed")
    st.write(
        "We envision online platforms where users have more control over what they see. Toggle the civility filter to "
        "see how the user's feed changes. Use the slider to change the level of toxicity that is allowed."
    )

    show_toxic = st.checkbox("Hide toxic content", False)
    with st.spinner("Loading..."):
        dataset = load_dataset("civil_comments")
        np.random.seed(10)
        random_sampling = np.random.randint(0, 1000, 20)

        unfiltered_df = pd.DataFrame(
            dataset["test"][random_sampling]["text"],
            columns=["Text"]

        )
        unfiltered_df.insert(1, "Toxic Level", dataset["test"][random_sampling]["toxicity"], True)

        if not show_toxic:
            st.table(unfiltered_df)
        else:
            toxic_threshold = st.slider(
                "Select Toxicity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01
            )

            filtered_df = unfiltered_df[unfiltered_df["Toxic Level"] < toxic_threshold]
            st.table(filtered_df)
