import streamlit as st

from googleapiclient import discovery


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
    custom_text = st.text_input(label="Provide a comment to see if it's considered civil by Jigsaw Perspective")
    if custom_text:
        response = jigsaw_request(custom_text)
        if response > 0.5:
            sub_phrase = "is not"
        else:
            sub_phrase = "is"
        st.write(f"The phrase {sub_phrase} civil, toxicity score: {response}")

    # Showing neutralizing bias
    st.header("Neutralizing Bias")

    # Showing conversation ai
    st.header("Conversation AI")

    # Showing others
    st.header("Others")
