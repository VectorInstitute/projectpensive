import streamlit as st


def introduction():
    st.header("Introduction to Project Pensive")

    st.write(
        "There is a growing need in the online media landscape to put some control of what a user sees back into their "
        "hands."
    )
    st.write(
        "We envision a platform where the user is provided with several optional filters they can apply to their "
        "feeds. Two such filters revolve around civility and diversity. Although this is somewhat implemented in "
        "current mainstream platforms, these implementations can be extended, and are not open-source."
    )
    st.write(
        "My work focuses on the civility aspect of Project Pensive. Several approaches are taken to provide insight on "
        "the civility of social media posts. These approaches are discussed further in the next sections."
    )
    st.image(
        "images/recommender_companies.png",
        caption="Most companies are using Recommender Systems, in some capacity."
    )
