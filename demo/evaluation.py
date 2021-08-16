import streamlit as st


def evaluation():
    st.header("Evaluation")
    st.write(
        "The following section shows the evaluation of the tools built for this project."
    )

    # Civility
    st.subheader("Civility")
    with st.expander("Read more"):
        st.markdown(
            """To evaluate our civility filter, we will compare its performance with the [Jigsaw Perspective API]
            (https://www.perspectiveapi.com/). The Jigsaw Perspective tool is built by [Conversation AI]
            (https://conversationai.github.io/), the maintainers of our `civil_comments` dataset."""
        )
        st.write(
            "The following test was run on a subset of the test dataset."
        )
        st.code(
            "jigsaw_precision, jigsaw_recall, jigsaw_f1_score, _ = sklearn.metrics.precision_recall_fscore_support("
            "\n\ty_true=dataset_is_toxic,"
            "\n\ty_pred=jigsaw_is_toxic,"
            '\n\taverage="weighted"'
            "\n)"
            "\nmodel_precision, model_recall, model_f1_score, _ = sklearn.metrics.precision_recall_fscore_support("
            "\n\ty_true=dataset_is_toxic,"
            "\n\ty_pred=model_is_toxic,"
            '\n\taverage="weighted"'
            "\n)"
        )
        st.write(
            "Output:"
        )
        st.code(
            "Jigsaw:"
            "\n\tPrecision: 0.957, Recall: 0.923, F1: 0.934"
            "\nModel:"
            "\n\tPrecision: 0.931, Recall: 0.936, F1: 0.933"
        )
        st.markdown(
            "As can be seen, our transformer model is achieving similar performance. Unfortunately, Jigsaw's model is "
            "not open source, and we cannot compare model sizes. However, [they state]"
            "(https://developers.perspectiveapi.com/s/about-the-api-model-cards) that they are using a distilled "
            "versions of `BERT`."
        )

    # Diversity
    st.subheader("Diversity")
    with st.expander("Read more"):
        st.write("Written by Sheen")
