import tensorflow as tf

from model import Model
import numpy as np


IMAGE_SIZE = 28

# tf.compat.v1.disable_eager_execution()

class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        # features = tf.compat.v1.placeholder(
        #     tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        # labels = tf.compat.v1.placeholder(tf.int64, shape=[None], name='labels')

        inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE*IMAGE_SIZE,))
        input_layer = tf.keras.layers.Reshape([IMAGE_SIZE, IMAGE_SIZE, 1])(inputs)

        conv1 = tf.keras.layers.Conv2D(32, [5,5], padding = "same", activation='relu')(input_layer)
        #   inputs=input_layer,
        #   filters=32,
        #   kernel_size=[5, 5],
        #   padding="same",
        #   activation=tf.nn.relu)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)
        conv2 = tf.keras.layers.Conv2D(64, [5,5], padding="same", activation='relu')(pool1)
            # inputs=pool1,
            # filters=64,
            # kernel_size=[5, 5],
            # padding="same",
            # activation=tf.nn.relu)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)
        # pool2_flat = tf.keras.layers.Reshape([-1, 7 * 7 * 64])(pool2)
        pool2_flat = tf.keras.layers.Flatten()(pool2)
        dense = tf.keras.layers.Dense(units=2048, activation='relu')(pool2_flat)
        logits = tf.keras.layers.Dense(units=self.num_classes)(dense)
        # predictions = {
        #   "classes": tf.argmax(input=logits, axis=1),
        #   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        # }
        model = tf.keras.Model(inputs=inputs, outputs=logits)

        model.compile(
            optimizer = self._optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        # loss = tf.keras.losses.SparseCategoricalCrossentropy(labels, logits)
        # TODO: Confirm that opt initialized once is ok?
        # train_op = self.optimizer.minimize(
        #     loss=loss,
        #     global_step=tf.train.get_global_step())
        # eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        # return features, labels, train_op, eval_metric_ops, loss
        return model

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
