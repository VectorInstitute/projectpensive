import streamlit as st


def failed_attempts():
    st.header("Failed Attempts")

    # Civility
    st.subheader("Civility")
    with st.expander("Read more"):
        st.write(
            "It took many iterations to build a robust civility classifier that can train on multiple GPUs. "
            "Incomplete implementations are detailed in this section, and are available for viewing in notebook form "
            "at `civility/classifier/other_implementations/`"
        )

        st.write("**TensorFlow Keras**")
        st.write(
            "`Hugging Face` provides model plugins for many frameworks. We successfully built a `TensorFlow Keras` "
            "model that learned to assign toxicity scores to comments. On a single GPU, this model took an extremely "
            "long time to train. This is due to the dataset's very large size. It was immediately clear that we needed "
            "a multi-GPU training scheme."
        )
        st.markdown(
            "After spending a lot of time debugging to get [TensorFlow's Distributed Training]"
            "(https://www.tensorflow.org/tutorials/distribute/keras) to play nicely with the `Hugging Face` model, we "
            "moved to the next implementation, which has built-in multi-GPU training support."
        )

        st.write("**Hugging Face Trainer Class**")
        st.write(
            "Although the `Hugging Face` `Trainer` class provides built-in multi-GPU training, it was not an ideal "
            "option. Most of the training loop is hidden to the developer, with little customization available. "
            "When the model did not converge as expected, we decided it would be easier to write our own training "
            "loop, the next implementation. "
        )

        st.write("**PyTorch Custom Training Loop**")
        st.write(
            "Once we built our own custom training loop, it was much easier to train a civility classifier that "
            "converged. Now we just need to provide multi-GPU training support for the training loop. To handle this, "
            "we leverage `PyTorch Lightning`. The combination of the custom training loop with a multi-GPU training "
            "scheme is our final implementation."
        )

    # Diversity
    st.subheader("Diversity")
    with st.expander("Read more"):
        st.write("Written by Sheen")

    # Recommender
    st.subheader("Recommender Engine")
    with st.expander("Read more"):
        st.write("**TensorFlow Recommenders**")
