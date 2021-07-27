import numpy as np
import pandas as pd
import tensorflow as tf

from model import CombinedModel


if __name__ == "__main__":

    # Data
    data = pd.read_csv("train-balanced-sarcasm.csv")
    data = data.drop(["label", "date", "created_utc", "parent_comment"], 1)
    data = data.astype({"comment": str, "author": str, "subreddit": str, "score": int, "ups": int, "downs": int})
    print(data.head())

    queries = data.drop(["score", "ups", "downs"], 1)
    comments = data.drop(["author", "subreddit"], 1)

    # Manipulate data
    ds = tf.data.Dataset.from_tensor_slices(dict(data))
    queries_ds = tf.data.Dataset.from_tensor_slices(dict(queries))
    comments_ds = tf.data.Dataset.from_tensor_slices(dict(comments))

    ds = ds.map(lambda x: {
        "comment": x["comment"],
        "author": x["author"],
        "subreddit": x["subreddit"],
        "score": x["score"],
        "ups": x["ups"],
        "downs": x["downs"]
    })
    queries_ds = queries_ds.map(lambda x: {
        "author": x["author"],
        "subreddit": x["subreddit"]
    })
    comments_ds = comments_ds.map(lambda x: {
        "comment": x["comment"],
        "score": x["score"],
        "ups": x["ups"],
        "downs": x["downs"]
    })

    min_max_dict = {
        "score": {
            "min": data["score"].min(),
            "max": data["score"].max()
        },
        "ups": {
            "min": data["ups"].min(),
            "max": data["ups"].max()
        },
        "downs": {
            "min": data["downs"].min(),
            "max": data["downs"].max()
        }
    }

    tf.random.set_seed(42)
    shuffled = ds.shuffle(len(ds), seed=42, reshuffle_each_iteration=False)

    train_split = int(len(ds) * 0.9)
    test_split = len(ds) - train_split
    batch_size = 32

    train = shuffled.take(train_split)
    test = shuffled.skip(train_split).take(test_split)

    cached_train = train.shuffle(train_split).batch(batch_size).cache()
    cached_test = test.batch(batch_size).cache()

    num_epochs = 20

    model = CombinedModel(queries_ds, comments_ds, min_max_dict)
    model.compile(optimizer=tf.keras.optimizers.Adam())

    history = model.fit(
        cached_train,
        validation_data=cached_test,
        epochs=num_epochs,
        callbacks=[model.checkpoint_callback]
    )

    tf.keras.models.save_model(model, "results/best_model")

    test_query = {"author": np.array("Trumpbar"), "subreddit": np.array("politics"), "score": 2, "ups": -1, "downs": -1}
    test_candidates = [
            {
                "comment": np.array(["NC and NH."]),
                "score": np.array([2]),
                "ups": np.array([-1]),
                "downs": np.array([-1])
            },
            {
                "comment": np.array(["You do know west teams play against west teams more than east teams right?"]),
                "score": np.array([-4]),
                "ups": np.array([-1]),
                "downs": np.array([-1])
            },
    ]
    result = model.get_recommendations(test_query, test_candidates, num_recommendations=10)
    print(result)

    print("Training program complete")
