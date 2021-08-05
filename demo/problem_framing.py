import streamlit as st


def problem_framing():
    st.header("Problem Framing")

    st.write(
        "The social media and internet landscape has grown immensely since its creation. Platforms such as Facebook, "
        "Instagram, Reddit, Twitter, etc. have made it easier than ever to stay connected. When used correctly, social "
        "media can pose several benefits to companies and the individual user."
    )
    st.write(
        "With its pros comes many cons. The content shown to users, provided by Recommender Engines, is often myopic "
        "and contributes to the increase in misinformation, polarization, depression, and disengagement."
    )
    st.write(
        "The Vector Institute wishes to combat this problem. Our initial efforts surround building a tool that "
        "leverages NLP and Machine Learning to apply **deep** content analysis to boost or suppress content "
        "recommendations based on a user's preferences. Vector envisions a social media platform where the user has "
        "control. They can apply various filters to their feed to enhance their own experience."
    )
    st.write(
        "Two such filters in development and presented today are our **Civility** and **Diversity** filters."
    )

    # Civility
    st.subheader("Civility")
    st.write(
        "Media feeds can often populate with hateful and uncivil speech. Simply excluding profane content is not a "
        "satisfactory solution. Demoting content that contains mildly inappropriate speech (swear words) but provides "
        "valid points is not ideal to every user. A more comprehensive solution assigns a toxicity score to content "
        "and allows individuals to set their own tolerance level."
    )

    # Diversity
    st.subheader("Diversity")
    st.write("Written by Sheen")
