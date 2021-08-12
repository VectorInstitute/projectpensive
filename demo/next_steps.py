import streamlit as st


def next_steps():
    st.header("Next Steps")

    # Civility
    st.subheader("Civility")
    with st.expander("Read more"):
        st.write("")

    # Diversity
    st.subheader("Diversity")
    with st.expander("Read more"):
        st.write("Written by Sheen")

    # Recommender
    st.subheader("Recommender Engine")
    with st.expander("Read more"):
        st.write("")
