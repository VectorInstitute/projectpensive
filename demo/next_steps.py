import streamlit as st


def next_steps():
    st.header("Next Steps")
    st.write(
        "This section outlines further work for Project Pensive."
    )

    # Civility
    st.subheader("Civility")
    with st.expander("Read more"):
        st.write(
            "Given more time, there are two things we would attempt to improve the civility filter."
        )
        st.write(
            "A boost in model performance would improve the overall quality of the filter. This could be achieved via "
            "more elaborate hyper-parameter optimization or further experimentation of the model architecture. "
            "`Hugging Face` offers a large selection of pretrained models. It is worthwhile to explore and record "
            "performances of architectures other than `DistilBERT`. `GPT-2` and `GPT-Neo` are attractive alternatives."
        )
        st.write(
            "Another valuable improvement would be to increase the number of labels output by the model. The "
            "`civil_comments` dataset provides many labels, not just the toxicity score. Adding more labels increases "
            "the user's control over their feed. These other labels are `severe toxicity`, `threat`, "
            "`identity attack`, `sexually explicit`, `insult`, `obscene`."
        )
        st.write(
            "To provide an example of where more labels would be useful, imagine a user is content with the "
            "occasional swear word but has no desire to see any identity attacks or sexually explicit content. With "
            "this addition, they would have the power to control that."
        )

    # Diversity
    st.subheader("Diversity")
    with st.expander("Read more"):
        st.write("Given more time, there are two things we would attempt to improve the diversity filter.")
        st.write('**Embeddings**')
        st.write("""The embeddings for both subreddits and comments were generated using limited techniques - Word2Vec skip-gram and SentenceTransformer models, respectively. For subreddits, other embeddings models should be explored and evaluated such as the `Continuous Bag-of-words` model in Word2Vec which takes the context words as input and predicts the center word within the window; `Glove` embedding, which is an unsupervised algorithm trained on global word-word co-occurence statistics leading to meaningful linear substructures in the word-vector space; `FastText`, which is pre-trained on English webcrawl and Wikipedia; and `ELMO`, which uses a deep bidirectional language model.
       """)
        st.write("""For comments, other pre-trained SentenceTransformer models should be compared such as `paraphrase-mpnet-base-v2`, `paraphrase-MiniLM-L12-v2`, and `paraphrase-TinyBERT-L6-v2`. 
        As well, other sentence embedding models should be trained including `InferSent`, created by Facebook and used to generate semantic sentence representations; and `Universal Sentence Encoder`, which encodes text into high dimensional vectors that can be used for text classification, semantic similarity, and clustering.
        The failed `Doc2Vec` implementation should also be further investigated and improved with fine tuning and hyperparameter optimization.
        """)
        st.write('**Diversity Algorithms**')
        st.write("""Given the limited time, we were only able to implement three diversity algorithms. However, there are many more that should be explored in the future as listed and described in [*Diversity in recommender systems – A survey*](https://papers-gamma.link/static/memory/pdfs/153-Kunaver_Diversity_in_Recommender_Systems_2017.pdf). 
        Further, the diversity algorithms implemented were ones that had already developed by other researchers. Given more time, we could have explored ways to improve the existing algorithms and even perhaps create a novel technique to diversify recommendations.
        """)

    # Recommender
    st.subheader("Recommender Engine")
    with st.expander("Read more"):
        st.write(
            "There are several ways to improve and increase the robustness of the current Recommender."
        )
        st.write(
            "First improvement is adding more input query structures. Currently, a query consists of an author and "
            "subreddit. This can be extended to support combinations of queries with authors, subreddits, comments, "
            "and others. Functionality to recommend authors and subreddits is another route to take."
        )
        st.write(
            "Another improvement builds more meaningful representations of comments. The implementation shown today "
            "encodes the comments into a lookup table. Even though this scheme works and provides good "
            "recommendations, it would be ideal to compute more meaningful word embeddings. This change would most "
            "likely result in even better comment recommendations."
        )
