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
        "The Vector Institute is tackling this problem. Our initial efforts surround building a tool that "
        "leverages NLP and Machine Learning to apply **deep** content analysis to boost or suppress content "
        "recommendations based on a user's preferences. Vector envisions a social media platform where the user has "
        "control. They can apply various filters to their feed and enhance their own experience."
    )
    st.write(
        "Two such filters in development and presented today are our `Civility` and `Diversity` filters."
    )

    # Civility
    st.subheader("Civility")
    with st.expander("Read more"):
        st.write(
            "It is an unfortunate reality that social media feeds often populate with hateful and uncivil speech. This "
            "often leads to further aggression among users and takes focus away from the more meaningful discussion "
            "and prosocial content. With the advances in NLP and Machine Learning, it is easier than ever to build "
            "models that can analyze text and provide meaningful insights. Our `civility filter` empowers the "
            "individual user to remove distasteful posts from their feeds."
        )
        st.write(
            "Our filter does more than simply exclude profane content. Demoting content that provides valid points but "
            "contains mildly inappropriate speech (swear words) is not ideal to every user. A more comprehensive "
            "solution assigns a toxicity score to content and allows individuals to set their own tolerance level. "
            "This approach maximizes feed customization while ensuring that users avoid content they are "
            "uncomfortable viewing."
        )

    # Diversity
    st.subheader("Diversity")
    with st.expander("Read more"):
        st.markdown(
            """On social media platforms, users have the ability to access and engage with millions of pieces of 
            content; this amount of data can become overwhelming and cause difficulty in finding relevant information 
            quickly. Recommendation systems were developed to combat this problem by selecting items a user may 
            appreciate based on their past preferences or demographic information. Traditionally, the focus of these 
            systems has been to optimize the accuracy of the recommendations presented to the user leading to a 
            concentration of highly accurate, but **overly narrow** set of results. This often traps users in a so-called
            `filter bubble` where they only encounter information and opinions that conform and reinforce their 
            own beliefs. This situation may seem innocuous, however it has been found to fuel issues such as 
            misinformation, unscientific propaganda, conspiracy theories and radical beliefs that have led to 
            real-world violence."""
        )
        st.markdown(
            """To combat this growing issue, our focus is to generate recommendations that concentrate on real user 
            experience, specifically increasing the diversity of content that a user views and engages with. Content 
            diversity has many benefits:
                \n- Increases user satisfaction
                \n- Allows users to explore more of the platform
                \n- Prevents the aforementioned `filter bubble` problem"""
        )
        st.markdown("Rather than focusing on short-term satisfaction metrics such as click-through rates, our `diversity filter` will focus on the diversity metric which helps users consciously widen the scope of the information they digest.")

    st.markdown(
        "With the inclusion of `civility` and `diversity` into the conversation of recommendation systems, we hope to **promote "
        "prosocial behaviour** in online communities and increase **long-term user satisfaction**."
    )
