import numpy as np
import pandas as pd
import streamlit as st

from datasets import load_dataset


def shorten(text):
    if len(text) > 200:
        return text[0:200] + "..."
    else:
        return text


def problem_framing():
    st.header("Problem Framing")

    # Description
    st.write(
        "For the problem of classifying civility, we leverage the *civil_comments* dataset. The dataset is available "
        "at https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data, with plugins for "
        "TensorFlow, Hugging Face, etc, available. The dataset was built by the ConversationAI team, a research group "
        "founded by Jigsaw and Google."
    )
    st.write(
        "Lets take a look at the data."
    )

    # Showing the dataset
    st.header("The *civil_comments* dataset")
    sample_size = st.slider(
        "How many data points do you want to see?",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )
    dataset = load_dataset("civil_comments")
    random_sampling = np.random.randint(0, 1000, sample_size)
    df = pd.DataFrame(
        dataset["test"][random_sampling]
    )
    df["text"] = df["text"].apply(shorten)
    st.table(df)
