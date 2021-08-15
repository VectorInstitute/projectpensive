import streamlit as st


def evaluation():
    st.header("Evaluation")
    st.write(
        "The following section shows the evaluation of the systems built for this project."
    )

    # Civility
    st.subheader("Civility")
    with st.expander("Read more"):
        st.write(
            "To evaluate the performance of the model, we will compare it with the Jigsaw Perspective API..."
        )

    # Diversity
    st.subheader("Diversity")
    with st.expander("Read more"):
        st.write("Written by Sheen")

    # Recommender
    st.subheader("Recommender Engine")
    with st.expander("Read more"):
        st.write(
            ""
        )
