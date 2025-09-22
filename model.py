# """Interfaces for ClientModel and ServerModel."""

# from abc import ABC, abstractmethod
# import numpy as np
# import os
# import sys
# import tensorflow as tf

# from baseline_constants import ACCURACY_KEY

# from utils.model_utils import batch_data
# from utils.tf_utils import graph_size


# class Model(ABC):

#     def __init__(self, seed, lr, optimizer=None):
#         self.lr = lr
#         self.seed = seed
#         self._optimizer = optimizer

#         # self.graph = tf.Graph()
#         # with self.graph.as_default():
#         #     tf.random.set_seed(123 + self.seed)
#         #     self.features, self.labels, self.train_op, self.eval_metric_ops, self.loss = self.create_model()
#         #     self.saver = tf.train.Saver()
#         # self.sess = tf.Session(graph=self.graph)

#         # self.size = graph_size(self.graph)

#         self.model = self.create_model()

#         if self._optimizer is None:
#             self._optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)

#         self.model.compile(
#             optimizer=self._optimizer,
#             loss =tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics = ['accuracy'])
        
#         # with self.graph.as_default():
#         #     self.sess.run(tf.global_variables_initializer())

#         #     metadata = tf.RunMetadata()
#         #     opts = tf.profiler.ProfileOptionBuilder.float_operation()
#         #     self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

#         np.random.seed(self.seed)

#     def set_params(self, model_params):
#         # with self.graph.as_default():
#         #     all_vars = tf.trainable_variables()
#         #     for variable, value in zip(all_vars, model_params):
#         #         variable.load(value, self.sess)
#         self.model.set_weights(model_params)

#     def get_params(self):
#         # with self.graph.as_default():
#         #     model_params = self.sess.run(tf.trainable_variables())
#         # return model_params
#         return self.model.get_weights()

#     # @property
#     # def optimizer(self):
#     #     """Optimizer to be used by the model."""
#     #     if self._optimizer is None:
#     #         self._optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)

#     #     return self._optimizer

#     @abstractmethod
#     def create_model(self):
#         """Creates the model for the task.

#         Returns:
#             A 4-tuple consisting of:
#                 features: A placeholder for the samples' features.
#                 labels: A placeholder for the samples' labels.
#                 train_op: A Tensorflow operation that, when run with the features and
#                     the labels, trains the model.
#                 eval_metric_ops: A Tensorflow operation that, when run with features and labels,
#                     returns the accuracy of the model.
#         """
#         return None, None, None, None, None

#     def train(self, data, num_epochs=1, batch_size=10):
#         """
#         Trains the client model.

#         Args:
#             data: Dict of the form {'x': [list], 'y': [list]}.
#             num_epochs: Number of epochs to train.
#             batch_size: Size of training batches.
#         Return:
#             comp: Number of FLOPs computed while training given data
#             update: List of np.ndarray weights, with each weight array
#                 corresponding to a variable in the resulting graph
#         """

#         x = self.process_x(data['x'])
#         y = self.process_y(data['y'])
#         self.model.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=0)

#         # for _ in range(num_epochs):
#         #     self.run_epoch(data, batch_size)

#         update = self.get_params()
#         comp = num_epochs * (len(data['y'])//batch_size) * batch_size # * self.flops
#         return comp, update

#     # def run_epoch(self, data, batch_size):

#     #     for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            
#     #         input_data = self.process_x(batched_x)
#     #         target_data = self.process_y(batched_y)
            
#     #         with self.graph.as_default():
#     #             self.sess.run(self.train_op,
#     #                 feed_dict={
#     #                     self.features: input_data,
#     #                     self.labels: target_data
#     #                 })

#     def test(self, data):
#         """
#         Tests the current model on the given data.

#         Args:
#             data: dict of the form {'x': [list], 'y': [list]}
#         Return:
#             dict of metrics that will be recorded by the simulation.
#         """
#         x_vecs = self.process_x(data['x'])
#         labels = self.process_y(data['y'])
#         # with self.graph.as_default():
#         #     tot_acc, loss = self.sess.run(
#         #         [self.eval_metric_ops, self.loss],
#         #         feed_dict={self.features: x_vecs, self.labels: labels}
#         #     )
#         results = self.model.evaluate(x_vecs, labels, verbose=0, return_dict=True)
#         # acc = float(tot_acc) / x_vecs.shape[0]
#         return {ACCURACY_KEY: results.get('sparse_categorical_accuracy',0.0), 'loss': results['loss']}

#     # def close(self):
#     #     self.sess.close()

#     @abstractmethod
#     def process_x(self, raw_x_batch):
#         """Pre-processes each batch of features before being fed to the model."""
#         pass

#     @abstractmethod
#     def process_y(self, raw_y_batch):
#         """Pre-processes each batch of labels before being fed to the model."""
#         pass


# class ServerModel:
#     def __init__(self, model):
#         self.model = model

#     @property
#     def size(self):
#         # return self.model.size
#         return self.model.model.count_params()

#     @property
#     def cur_model(self):
#         return self.model

#     def send_to(self, clients):
#         """Copies server model variables to each of the given clients

#         Args:
#             clients: list of Client objects
#         """
#         # var_vals = {}
#         # with self.model.graph.as_default():
#         #     all_vars = tf.trainable_variables()
#         #     for v in all_vars:
#         #         val = self.model.sess.run(v)
#         #         var_vals[v.name] = val
#         weights = self.model.get_params()
#         for c in clients:
#             # with c.model.graph.as_default():
#             c.model.set_params(weights)
#                 # all_vars = tf.trainable_variables()
#                 # for v in all_vars:
#                 #     v.load(var_vals[v.name], c.model.sess)

#     def save(self, path='checkpoints/model.ckpt'):
#         return self.model.save(path)

#     def close(self):
#         self.model.close()

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras import Model, layers, optimizers
from baseline_constants import ACCURACY_KEY
from utils.model_utils import batch_data

class Model(ABC):
    def __init__(self, seed, lr, optimizer=None):
        self.lr = lr
        self.seed = seed
        self._optimizer = optimizer
        tf.random.set_seed(123 + self.seed)
        np.random.seed(self.seed)

        if self._optimizer is None:
            self._optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        
        # Create model
        self.model = self.create_model()
        
        # Calculate model size (number of parameters)
        self.size = sum(np.prod(var.shape) for var in self.model.trainable_variables)
        
        # Calculate FLOPs
        self.flops = self._calculate_flops()

    def _calculate_flops(self):
        """Calculate FLOPs for the model."""
        # Note: TF2 doesn't have a direct equivalent to tf.profiler.profile
        # This is a simplified approximation
        return sum(tf.keras.backend.count_params(var) for var in self.model.trainable_variables)

    def set_params(self, model_params):
        """Set model weights."""
        self.model.set_weights(model_params)

    def get_params(self):
        """Get model weights."""
        # return self.model.trainable_variables
        return self.model.get_weights()
    
    def save_model(self, path):
        "save model weights"
        self.model.save_weights(path)

    # @property
    # def optimizer(self):
    #     """Optimizer to be used by the model."""
    #     if self._optimizer is None:
    #         self._optimizer = optimizers.SGD(learning_rate=self.lr)
    #     return self._optimizer

    @abstractmethod
    def create_model(self):
        """Creates the Keras model for the task."""
        pass

    def train(self, data, num_epochs=1, batch_size=10):
        """
        Trains the model.
        
        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training
            update: List of np.ndarray weights
        """
        for _ in range(num_epochs):
            self.run_epoch(data, batch_size)
        
        update = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return comp, update

    def run_epoch(self, data, batch_size):
        """Run one training epoch."""
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)
            self.model.train_on_batch(input_data, target_data)

    def test(self, data):
        """
        Tests the model on the given data.
        
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics
        """
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        
        loss, acc = self.model.evaluate(x_vecs, labels, verbose=0)
        return {ACCURACY_KEY: acc, 'loss': loss}

    def close(self):
        """Clean up resources."""
        # In TF2, no explicit session closing is needed
        pass

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels."""
        pass

class ServerModel:
    def __init__(self, model):
        self.model = model

    @property
    def size(self):
        return self.model.size

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients."""
        var_vals = self.model.get_params()
        for client in clients:
            client.model.set_weights(var_vals)

    def save(self, path='checkpoints/model.h5'):
        """Save model weights."""
        self.model.model.save_weights(path)
        return path

    def close(self):
        self.model.close()

# class ClientModel(Model):
#     def __init__(self, seed, lr, num_classes, image_size=28):
#         self.num_classes = num_classes
#         self.image_size = image_size
#         super().__init__(seed, lr)

#     def create_model(self):
#         """Model function for CNN using Keras Model subclassing."""
#         class CNNModel(Model):
#             def __init__(self, num_classes, image_size):
#                 super().__init__()
#                 self.reshape = layers.Reshape((image_size, image_size, 1))
#                 self.conv1 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')
#                 self.pool1 = layers.MaxPooling2D((2, 2), strides=2)
#                 self.conv2 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')
#                 self.pool2 = layers.MaxPooling2D((2, 2), strides=2)
#                 self.flatten = layers.Flatten()
#                 self.dense1 = layers.Dense(2048, activation='relu')
#                 self.dense2 = layers.Dense(num_classes)

#             def call(self, inputs):
#                 x = self.reshape(inputs)
#                 x = self.conv1(x)
#                 x = self.pool1(x)
#                 x = self.conv2(x)
#                 x = self.pool2(x)
#                 x = self.flatten(x)
#                 x = self.dense1(x)
#                 x = self.dense2(x)
#                 return x

#         model = CNNModel(self.num_classes, self.image_size)
#         model.compile(
#             optimizer=self.optimizer,
#             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics=['accuracy']
#         )
#         return model

#     def process_x(self, raw_x_batch):
#         """Pre-processes each batch of features."""
#         return np.array(raw_x_batch)

#     def process_y(self, raw_y_batch):
#         """Pre-processes each batch of labels."""
#         return np.array(raw_y_batch)