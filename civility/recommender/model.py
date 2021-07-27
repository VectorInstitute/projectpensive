from abc import ABC

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs


def create_cont_var_embedding(min_max_dict, dim, num_bins=1000):
    buckets = np.linspace(
        min_max_dict["min"], min_max_dict["max"], num=num_bins
    )

    # TODO: normalize?
    embedding = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Discretization(buckets.tolist()),
        tf.keras.layers.Embedding(len(buckets) + 1, dim),
    ])

    return embedding


class QueryModel(tf.keras.Model, ABC):
    """Model for encoding user queries"""

    def __init__(self, queries_ds):
        super().__init__()

        # Embeddings
        author_lookup = tf.keras.layers.experimental.preprocessing.StringLookup()
        author_lookup.adapt(queries_ds.map(lambda x: x["author"]))
        self.author_embedding = tf.keras.Sequential([
            author_lookup,
            tf.keras.layers.Embedding(
                input_dim=author_lookup.vocabulary_size(),
                output_dim=20
            )
        ])

        subreddit_lookup = tf.keras.layers.experimental.preprocessing.StringLookup()
        subreddit_lookup.adapt(queries_ds.map(lambda x: x["subreddit"]))
        self.subreddit_embedding = tf.keras.Sequential([
            subreddit_lookup,
            tf.keras.layers.Embedding(
                input_dim=subreddit_lookup.vocabulary_size(),
                output_dim=20
            )
        ])

    def call(self, query, predicting=False):

        if not predicting:
            embeddings = tf.concat(
                [self.author_embedding(query["author"]), self.subreddit_embedding(query["subreddit"])],
                axis=1
            )
        else:
            embeddings = tf.concat(
                [self.author_embedding(query["author"]), self.subreddit_embedding(query["subreddit"])],
                axis=0
            )
            embeddings = tf.expand_dims(embeddings, axis=0)

        return embeddings


class CommentModel(tf.keras.Model, ABC):
    """Model for encoding comments"""

    def __init__(self, comments_ds, min_max_dict):
        super().__init__()

        # Embeddings
        comment_vocab = tf.keras.layers.experimental.preprocessing.TextVectorization()
        comment_vocab.adapt(comments_ds.map(lambda x: x["comment"]))
        self.comment_embedding = tf.keras.Sequential([
            comment_vocab,
            tf.keras.layers.Embedding(
                comment_vocab.vocabulary_size(),
                20,
                mask_zero=True
            ),
            tf.keras.layers.GlobalAveragePooling1D()
        ])

        #   Score
        self.score_embedding = create_cont_var_embedding(min_max_dict["score"], dim=10)
        self.ups_embedding = create_cont_var_embedding(min_max_dict["ups"], dim=5)
        self.downs_embedding = create_cont_var_embedding(min_max_dict["downs"], dim=5)

    def call(self, comment):

        _comment = self.comment_embedding(comment["comment"])
        _score = self.score_embedding(comment["score"])
        _ups = self.ups_embedding(comment["ups"])
        _downs = self.downs_embedding(comment["downs"])
        embeddings = tf.concat([_comment, _score, _ups, _downs], axis=1)
        return embeddings


class CombinedModel(tfrs.models.Model, ABC):
    """
    Combines the query and comment models
    Dataset: https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit
    """

    def __init__(self, queries_ds, comments_ds, min_max_dict):
        super().__init__()

        # Define models
        self.query_model = QueryModel(queries_ds)
        self.comment_model = CommentModel(comments_ds, min_max_dict)

        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="results/checkpoints/"
        )

        candidates = comments_ds.batch(32).map(self.comment_model)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates
            )
        )

    def compute_loss(self, features, training=False):

        query_out = self.query_model({
            "author": features["author"],
            "subreddit": features["subreddit"]
        })
        comment_out = self.comment_model({
            "comment": features["comment"],
            "score": features["score"],
            "ups": features["ups"],
            "downs": features["downs"]
        })

        return self.task(
            query_out, comment_out
        )

    @staticmethod
    def take_first(x):
        return x[0]

    def get_recommendations(self, query, candidates, num_recommendations):
        query_rep = np.squeeze(self.query_model(query, predicting=True))
        comment_reps = [np.squeeze(self.comment_model(candidates[x])) for x in range(len(candidates))]

        out = [(np.dot(query_rep, comment_reps[x]), candidates[x]["comment"][0])for x in range(len(comment_reps))]
        out.sort(key=self.take_first, reverse=True)
        return out[:num_recommendations]
