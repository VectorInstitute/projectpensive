import streamlit as st


def failed_attempts():
    st.header("Failed Attempts")
    st.write(
        "It took several iterations to build the tools shown in this demo. The following shows the journey taken and "
        "some of the work not seen in the previous sections."
    )

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
        st.write("**Doc2Vec Implementation**")
        st.write("""
        The first sentence embedding model implemented was the `Doc2Vec` model which is an unsupervised algorithm that learns fixed-length feature representations from variable-length pieces of texts, such as sentences, paragraphs, and documents. The model was trained on the Sarcastic Comments dataset, however its performance, evaluated by eye, fell short compared to the pretrained SentenceTransformers model. Perhaps in the future, the model should be fine tuned and trained on a larger dataset to improve accuracy.
        """)
        st.write("**Total Diversity Effect (TDE) Ranking Algorithm**")
        st.write("""
        A third diversity algorithm implemented was Total Diversity Effect Ranking which improves the overall recommendation diversity by considering the diversity effect of each item on the final recommendation list. This involves first generating a list of the Top N+S recommendations (N between 3 and 10; S between 1 and 10). Then, calculating the TDE of each item as the sum of distances to all other (N+S-1) items on the list. Lastly, removing S items with the lowest TDE score and generating the Top N recommendations for the current user. When implemented, this algorithm increased the diversity of the result-set from the original target query, but did not also increase the diversity of recommendations within the result-set which is an important metric. 
        """)
        st.latex('''
        TDE(c_i) = \sum_{i=1..L} dist(c_i, c_j) ; \quad i ≠ j, c_i, c_j, \in L
        ''')
        
    # Recommender
    st.subheader("Recommender Engine")
    with st.expander("Read more"):
        st.write("**TensorFlow Recommenders**")
        st.markdown(
            "[`TensorFlow Recommenders`](https://www.tensorflow.org/recommenders/) is a deep learning library for "
            "building recommender systems. It is built on `Keras`, with all of its functionality exposed. We built our "
            "first model using this library. When the model had trouble converging, especially when the dataset was "
            "large, we migrated to a `PyTorch` version. This version was much easier to debug, and had more success "
            "converging."
        )
