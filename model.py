

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
