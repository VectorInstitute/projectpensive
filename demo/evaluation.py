import streamlit as st


def evaluation():
    st.header("Evaluation")
    st.write(
        "The following section shows the evaluation of the systems built for this project."
    )

    # Civility
    st.subheader("Civility")
    with st.expander("Read more"):
        # interesting examples
        # eval against jigsaw
        st.write("**Performance Comparison with Jigsaw Perspective**")
        st.markdown(
            """To evaluate the model, we will compare its performance with the [Jigsaw Perspective API]
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
            "\n\tPrecision: 0.951, Recall: 0.913, F1: 0.926"
            "\nModel:"
            "\n\tPrecision: 0.934, Recall: 0.938, F1: 0.936"
        )
        st.markdown(
            "As can be seen, our transformer model is achieving similar performance. Unfortunately, Jigsaw's model is "
            "not open source, and we cannot compare model sizes. However, [they state]"
            "(https://developers.perspectiveapi.com/s/about-the-api-model-cards) that they are using a distilled "
            "versions of `BERT`."
        )

        st.write("**Interesting Examples**")

    # Diversity
    st.subheader("Diversity")
    with st.expander("Read more"):
        st.write("Written by Sheen")

    # Recommender
    st.subheader("Recommender Engine")
    with st.expander("Read more"):
        # interesting examples
        # mention that not limited to subreddit in query
        st.write(
            ""
        )
