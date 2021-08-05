import pandas as pd
import streamlit as st

from googleapiclient import discovery
from transformers import pipeline


def jigsaw_request(text):
    api_key = 'AIzaSyAQfy2kSqkRo7O_j7Zh7jT783OTEREV2m0'
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=api_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        cache_discovery=False
    )
    analyze_request = {
        "comment": {"text": f"{text}"},
        "requestedAttributes": {"TOXICITY": {}}
    }
    response = client.comments().analyze(body=analyze_request).execute()
    return response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]


def tools_explored():
    st.header("Tools Explored")
    st.write(
        "On the path to building a civility classifier, many tools were replicated and tested."
    )

    # Showing jigsaw
    st.header("Jigsaw Perspective")
    st.write(
        "The Jigsaw Perspective API leverages an NLP model trained on the *civil_comments* dataset. It supports "
        "multiple languages and provides scores for multiple categories, including toxicity, threat, insult, etc."
    )
    custom_text = st.text_input(label="Provide a comment to see if it's considered civil by Jigsaw Perspective")
    if custom_text:
        response = jigsaw_request(custom_text)
        if response > 0.5:
            sub_phrase = "IS NOT"
        else:
            sub_phrase = "IS"
        st.write(f"The phrase **{sub_phrase}** civil, toxicity score: {response}")

    # Showing neutralizing bias
    st.header("Neutralizing Bias")
    st.write(
        "The goal of this tool is to identify phrases that have been worded with some bias. These phrases are edited, "
        "with some alternative wordings suggested."
    )
    st.image("images/neutralizing_bias.png")

    st.write(
        "Below are some example phrases and their corrected parts."
    )

    phrases = pd.DataFrame({
        "Raw":
            [
                "another infamous period of colonisation in ancient times was from the romans",
                "photo sequence of astonishing 2005 chicago land crash with ryan briscoe",
                "his 45 - year career exceeded that of any other studio head",
                "along came a band of missionaries, but they were all horribly massacred",
                "during the unnecessary horse play , hamlin fell and severely injured his hand"
            ],
        "Corrected":
        [
            "another period of colonisation in ancient times was from the romans",
            "photo sequence of 2005 chicago land crash with ryan briscoe",
            "his 45 - year career was longer than that of any other studio head",
            "along came a band of missionaries , but they were all massacred",
            "during the horse play , hamlin fell and severely injured his hand"
        ]
    })
    st.table(phrases)

    # Showing Hugging Face
    st.header("Hugging Face")
    st.write(
        "Hugging Face is a Transformers library that can solve several NLP problems, including ASR, text "
        "classification, NER, question answering, etc. Along with the library comes a ton of pretrained models and "
        "datasets."
    )
    st.write(
        "Below, is a out-of-the-box model for sentiment analysis."
    )
    to_analyze = st.text_input("Type a phrase to test this sentiment classifier")
    if to_analyze:
        with st.spinner("Calculating..."):
            classifier = pipeline('sentiment-analysis')
            response = classifier(to_analyze)
            st.write(f"Sentiment Detected: {response[0]['label']}, Score: {response[0]['score']}")

    # Showing Recommender Systems
    st.header("Recommender Systems and Tensorflow Recommenders")
    st.write(
        "Some work has been done with the Tensorflow Recommenders package. I have acquired some experience working "
        "with embeddings and building recommender systems for movies/twitter data."
    )
    st.write("In the coming days, I will be building a recommender engine leveraging reddit data.")
    st.image("images/matrix_factorization.webp")

    # Showing others
    st.header("And others...")
    st.write("Recsys Twitter data, Conversation-ai, and others...")
