import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification

from dataset import CivilCommentsDataset


class CivilityModel:
    """
    Trains a civility classifier model, leveraging Hugging Face and TensorFlow.
    """

    def __init__(self, num_labels=7):

        """
        num_labels: Number of units in final dense layer of network. Defaults to 7 for the 7 categories of
            incivility in the dataset. Set to 1 for a simple civil/uncivil classifier.
        """

        # Define model and dataset
        self.dataset = CivilCommentsDataset()
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_labels
        )
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="civility_model",
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
        self.model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=loss, metrics="accuracy")

    def train(self, epochs, batch_size=32):

        print("Beginning train...")
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            with tf.device(gpus[0]):
                history = self.model.fit(
                    self.dataset.train_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=self.dataset.val_data,
                    callbacks=[self.model_checkpoint_callback]
                )
        else:
            history = self.model.fit(
                self.dataset.train_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=self.dataset.val_data,
                callbacks=[self.model_checkpoint_callback]
            )

        return history

    def test(self, batch_size=32):

        print("Beginning evaluation")
        self.model.evaluate(
           self.dataset.test_data,
           batch_size=batch_size
        )

    def predict(self, x):
        return self.model.predict(x)
