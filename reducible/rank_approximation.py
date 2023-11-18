import tensorflow as tf 
from tensorflow import keras
from keras import backend as K
from keras import activations
import numpy as np

@keras.saving.register_keras_serializable()
class RankKApprox(tf.keras.layers.Layer):
    def __init__(self, layer, k, activation=None, **kwargs):
        """
        Dense layer which stores a rank k approximation of the original layer

        Args:
            layer: layer of the model to compress
            k: rank of approximate matrix. 
                if weight matrix is size m*n, stores an approximation with size k(m + n)
            activation: activation function

        Result:
            Approximated dense layer

        Raises:
            ValueError: layer is a layer with no weight matrix
        """
        super().__init__(**kwargs)

        self.trainable=False
        self.activation = activations.get(activation)
        self.k = k

        if not layer.get_weights(): 
            raise ValueError("No weight matrix found for layer")
        
        w, b = layer.get_weights()
        self.b = K.constant(b)

        u, s, vT = np.linalg.svd(w, full_matrices=False)
        self.A = K.constant(u[:, :k] @ np.diag(s[:k]))
        self.B = K.constant(vT[:k])
        
    def call(self, inputs):  
        return self.activation(tf.matmul(inputs, tf.matmul(self.A, self.B)) + self.b)

    def get_config(self):
        """
        Serializes the object's config

        Returns:
            A Python dictionary to be seralized
        """
        base_config = super().get_config()
        config = {
            "k": self.k,
            "activation": self.activation
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        """
        Creates an object from a serialized one
        Object will have empty weights and biases

        Returns:
            An object from a serialized config

        Raises:
            KeyError: Raises an exception.
        """
        layer = cls([None, None], **config)
        return layer
    
    
