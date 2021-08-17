import streamlit as st


def evaluation():
    st.header("Evaluation")
    st.write(
        "The following section shows the evaluation of the tools built for this project."
    )

    # Civility
    st.subheader("Civility")
    with st.expander("Read more"):
        st.write("**Comparison with Jigsaw**")
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

        st.write("**Interesting Examples**")
        st.write(
            "Lets take a look at some interesting examples. The civility filter is not perfect, but performs well over "
            "the entirety of the dataset."
        )
        st.code(
            "Low Toxicity Example"
            "\n\tText: 'Well, at least the democratic nomination race will be over Tuesday.'"
            "\n\tScore: 0.0000"
            "\nMedium Toxicity Example"
            "\n\tText: 'They forgot to add the major fuck ups, blow its, and legal troubles that have plagued the "
            "conman for decades.-- Whats up with that?'"
            "\n\tScore: 0.4630"
            "\nHigh Toxicity Example"
            "\n\tText: 'Oh bully, its really a shame that not more parties in Germany a racist shitfests'"
            "\n\tScore: 0.6410"
        )

        st.write("Below is an interesting (and concerning) edge case")
        st.code(
            "Text: 'Thankfully not into the gas pump behind the dangerous negr- I mean black man.'"
            "\nScore: 0.2260"
        )
        st.write(
            "The above text is clearly uncivil and should receive a much higher toxicity score. It is suspected that "
            "this is not the case because the 'negr' term is not recognized by the model tokenizer. Even without the "
            "slur, this sentence should have a higher toxicity score. More work is needed to improve the filter, and "
            "they are outlined in the `Next Steps` section of this demo."
        )

    # Diversity
    st.subheader("Diversity")
    with st.expander("Read more"):
        st.markdown("""To evaluate our `diversity filter`, we compare the diversity metric of individual algorithms compared to a normal reccomendation system. As proposed in [Improving Recommendation Diversity](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.5232&rep=rep1&type=pdf), the diversity of a set of items, c<sub>1</sub>, ... c<sub>n</sub>, is the average *dissimilarity* between all pairs of items in the result-set.""", unsafe_allow_html=True)
        st.latex(r'''Diversity(c_1, ... c_n) = \frac { \sum_{i=1..n} \sum_{j=1..n}(1 - Similarity(c_i, c_j))}
        {\frac {n}{2} * (n - 1)}
        ''')
        st.markdown("This definition of diversity was used to compare result-sets for normal recommendation versus diversified recommendations as seen in the *Demo* section.")
