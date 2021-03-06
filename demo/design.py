import streamlit as st


def design():
    st.header("Design")
    st.write(
        "The following section highlights the design of our filters, recommender, and how they interact as a system."
    )

    # Civility Filter
    st.subheader("Civility Filter")
    with st.expander("Read more"):
        st.write("**The Dataset**")
        st.markdown(
            "The `civil_comments` dataset is an archive of the Civil Comments platform, a plugin for independent "
            "news sites. Public comments, gathered from ~50 news sites, were recorded from 2015 through 2017. When the "
            "platform shut down, the archive was open-sourced [[1]](https://huggingface.co/datasets/civil_comments)."
        )
        st.write(
            "For each comment, provided is an assigned score (between 0 and 1) for the following categories: "
            "`toxicity`, `severe toxicity`, `threat`, `identity attack`, `sexually explicit`, `insult`, `obscene`. "
            "These labels were taken as averages across many crowd workers."
        )
        st.write(
            "Below is a preselected example:"
        )
        st.code(
            "{"
            "\n\t'text': 'The profoundly stupid have spoken.'"
            "\n\t'toxicity': 0.879"
            "\n\t'severe_toxicity': 0.0"
            "\n\t'threat': 0.0"
            "\n\t'identity_attack': 0.0"
            "\n\t'sexually_explicit': 0.0"
            "\n\t'insult': 0.845"
            "\n\t'obscene': 0.224"
            "\n}"
        )

        st.write("**The Model**")
        st.markdown(
            "To analyze text and assign a toxicity score, we leveraged a distilled version of `BERT` known as "
            "[`DistilBERT`](https://arxiv.org/abs/1910.01108). `DistilBERT` is designed to be smaller, faster, and "
            "cheaper. It is pretrained on the same data as `BERT`, but using `BERT` as its teacher in a "
            "self-supervised manner [[2]](https://huggingface.co/distilbert-base-uncased)."
        )
        st.write(
            "During training, the transformer section of the model is frozen. Parameter updates are only applied to the "
            "subsequent linear layers. These linear layers interpret the text representation provided by the language "
            "model and assign the input a toxicity score. For more information on the model, and how it was pretrained, "
            "visit `DistilBERT`'s model card at https://huggingface.co/distilbert-base-uncased."
        )

        st.write("**Tools Used**")
        st.markdown(
            "Tools fundamental to the training of the civility classifier are "
            "[`Hugging Face`](https://huggingface.co/) and [`PyTorch Lightning`](https://www.pytorchlightning.ai/)."
        )
        st.markdown(
            "`Hugging Face` is a popular deep learning platform for building, training, and deploying state of the art "
            "NLP models. Its `transformers` library, formerly known as `pytorch-transformers`, is used to build the "
            "civility classifier model."
        )
        st.markdown(
            "`PyTorch Lightning` is a research framework that can accelerate your `PyTorch` code. Among its other "
            "benefits, it provides builtin functionality for multi-GPU training. This greatly reduced training of models "
            "on the large `civil_comments` dataset from days to hours. The framework provides several hooks that "
            "ease to migration from traditional `PyTorch` code to `Pytorch Lightning`. Examples of such hooks are:"
        )
        st.code(
            "forward(), configure_optimizers(), training_step(), validation_step(), on_train_batch_end()"
        )

    # Diversity Filter
    st.subheader("Diversity Filter")
    with st.expander("Read more"):
        st.write("**Generating Subreddit Embeddings**")
        st.markdown("""
        In order to embed the subreddits into a latent space representation, a `Subreddit2Vec` model was developed 
        which is based on the Word2Vec model. The underlying intuition behind Word2Vec is that two words are similar 
        if they are used in similar ways. While there are various implementations of Word2Vec, we will focus on the 
        `Skip-gram model` which goes through each word in the text corpus and tries to predict n words on either side 
        of it, referred to as the context. This context of a word can be represented through a set of 
        `skip-gram pairs` of (target_word, context_word). The training objective of the skip-gram model is to 
        maximize the probability of predicting context words given the target word 
        [[1]](https://www.tensorflow.org/tutorials/text/word2vec). 
        
        In this case, we use a Subreddit2Vec model to generate subreddit embeddings rather than word embeddings. We 
        apply the Word2Vec algorithm on interaction data by `treating subreddits as "words" and the users that comment 
        on them as "contexts"` - every instance of a user commenting in a subreddit then becomes a word-context, or 
        subreddit-user, pair. Then, two subreddits are similar if and only if many similar users have the time and 
        interest to comment in them both. We would like to point out that this approach of `community embeddings` was inspired by the work Isaac Waller and Ashton Anderson from UofT [[2]](https://www.cs.toronto.edu/~ashton/pubs/cultural-dims2020.pdf).
        """)
        st.write("**Generating Comment Embeddings**")
        st.markdown("""
        The comments were embedded using the [SentenceTransformers](https://www.sbert.net/) framework which can be 
        used to compute sentence/text embeddings for over 100 languages. These embeddings can then be compared e.g. 
        with cosine-similarity to find sentences with a similar meaning. This can be useful for semantic textual 
        similar, semantic search, or paraphrase mining. There are various pretrained models available for use. In our 
        application of embedding Reddit comments, we made use of `paraphrase-MiniLM-L6-v2` which is a quick model with 
        high quality.
        """)
        
        st.write('**Diversity Algorithms**')
        st.markdown(
            'These subreddit and comment embeddings were used to diversify the recommendations generated. '
            'Specifically, two diversity algorithms, `Bounded Greedy Selection` and `Topic Diversification` are '
            'implemented. These algorithms use the cosine similarities of the vectors to compare items. The algorithms '
            'are further detailed in the *Demo* section.'
        )

    # Recommender
    st.subheader("Recommender Engine")
    with st.expander("Read more"):
        st.write("**The Dataset**")
        st.write(
            "The [**Sarcastic Comments - REDDIT**](https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit) "
            "dataset consists of a balanced selection of sarcastic/genuine comment replies, made by a variety of users "
            "and posted in several subreddits. This dataset was selected for its ample selection of sarcastic, and "
            "sometimes uncivil, text."
        )
        st.write(
            "Below is a preselected example:"
        )
        st.code(
            "{"
            "\n\t'comment': 'I think a significant amount would be against spending their tax dollars on other people.'"
            "\n\t'parent_comment': 'I bet if that money was poured into college debt or health debt relief, 81% of "
            "Americans would have been for it instead.'"
            "\n\t'label': 0"
            "\n\t'subreddit': 'politics'"
            "\n\t'score': 92"
            "\n\t'ups': 92"
            "\n\t'downs': 0"
            "\n\t'date': '2016-09'"
            "\n\t'created_utc': '2016-09-20 17:53:52'"
            "\n}"
        )

        st.write("**The Model**")
        st.write(
            "The Recommender Engine is responsible for analyzing a large number of comments and ranking them based on "
            "a query. This query comprises of a `username` and `subreddit`. When the model suggests comment, it "
            "simulates the experience of `user x` navigating to `subreddit y` on Reddit and receiving a feed of "
            "relevant posts."
        )
        st.write(
            "It is important to note that the recommender was not designed to only suggest comments "
            "from `subreddit y`. Due to the sparsity of the dataset, the decision was made to allow comments from "
            "other subreddits. The idea is that these other comments should be relevant to the user or subreddit."
        )
        st.write(
            "The recommender is composed of two sub-models, responsible for computing representations of the queries and "
            "comments. The outputs of the models are combined to assign a query-candidate affinity score. A greater "
            "affinity score symbolizes a stronger match between query and comment."
        )

        st.write("**Tools Used**")
        st.write(
            "Building and training the Recommender Engine was done in `PyTorch`. Builtin support for building "
            "`Embedding` layers was done with `torch.nn.Embedding()`. Further implementations of this model should "
            "leverage `Hugging Face` to build more meaningful text representations of the comments in the dataset."
        )

    # Component Interaction
    st.subheader("Component Interaction")
    with st.expander("Read more"):
        st.write(
            "Before any filters are applied, it is the recommender's job to analyze thousands of comments and rank "
            "them based on their match to the input query. Once this is done, our filters can be applied. The "
            "workflow following the initial recommendations depends on what combination of filters are used. If no "
            "filters are used, the feed presented to the user is simply the first `n` posts of highest rank from the "
            "recommender."
        )

        st.write(
            "If the civility filter is selected, all recommended comments are processed through the civility "
            "classifier and assigned a toxicity score. For each comment, if its score is higher than the tolerance "
            "level selected by the user, it is dropped from the feed. Posts dropped can be seen below, in the `What "
            "was filtered` section."
        )
        st.write(
            "If the diversity filter is used, the user can select one of two algorithms, Bounded Greedy Selection and Topic Diversification, to apply to the recommender system. Both of these algorithmically re-rank each comment to enhance diversity in the resulting recommendation set. This set is then compared to the original using a diversity metric and a percent change is displayed for performance evaluation."
        )
        st.write(
            "If the user wishes to apply both filters, comments provided by the recommender are suggested for use with "
            "the diversity filter. The diversity filter provides a sequence of diverse posts, which are then processed "
            "by the civility filter as they normally would."
        )
        col1, col2, col3 = st.columns([0.05, 1, 0.05])
        col2.image("images/workflow.png", width=1000)
