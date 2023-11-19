import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import activations
import numpy as np
import copy

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
        
        if layer is None: 
            self.A = []
            self.B = []
            return
        
        if not layer.get_weights(): 
            raise ValueError("No weight matrix found for layer")
        
        w, b = layer.get_weights()
        self.b = K.constant(b)

        u, s, vT = np.linalg.svd(w, full_matrices=False)
        self.A = K.constant(u[:, :k] @ np.diag(s[:k]))
        self.B = K.constant(vT[:k])
        
    def call(self, inputs):  
        if len(self.A) != 0 and len(self.B) != 0:
            return self.activation(tf.matmul(inputs, tf.matmul(self.A, self.B)) + self.b)
        return K.constant([])

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
        layer = cls(None, **config)
        return layer
    
def model_from_ks_(base_model, ks): 
    """
    Builds a rank-k approximate model

    Args:
        base_model: model to reduce
        ks: the rank of each RankKApprox layer

    Returns:
        the optimized model with k's for values
    """
    ks = copy.deepcopy(ks)
    
    new_model = tf.keras.models.Sequential()
    
    for layer in base_model.layers:
        layer = copy.deepcopy(layer)
        # replacing Dense layers with RankKApprox
        if isinstance(layer, tf.keras.layers.Dense): 
            new_model.add(RankKApprox(layer, ks.pop(0), activation='relu'))
        else: 
            new_model.add(layer)
            
    cfg = base_model.get_compile_config()
    new_model.compile(optimizer=base_model.optimizer, loss = base_model.loss, metrics=cfg['metrics'])
    
    return new_model

def naive_k_finding_(base_model, alpha, bias=0): 
    """
    Finds the k-values based on a target percentage model size

    Args:
        base_model: model to reduce
        alpha: percent of original model to use

    Returns:
        approximate k-values, maximal k-values
    """
    # selects k to approximate target size
    target = lambda m, n, t: t*m*n/(m+n)
    
    ks = []
    max_ks = []
    
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Dense): 
            s = (layer.get_weights()[0]).shape
            k = target(s[0], s[1], alpha)
            k = np.floor(k).astype(int) + bias
            ks.append(k)
            max_ks.append(min(s))
    return ks, max_ks
    

def naive_model_reduction(base_model, alpha):
    """
    Reduces the base model to use approximately alpha*original_size

    Args:
        base_model: model to reduce
        alpha: percent of original model to use

    Returns:
        reduced model, k-values of reduction
    """
    ks, _ = naive_k_finding_(base_model, alpha)
    
    reduced = model_from_ks_(base_model, ks)
    
    return reduced, ks

def optimize_model(base_model, x_opt, y_opt, alpha=0.5, opt_factor=0.25, MAX_ITER=25):
    """
    Search algorithm to find optimal ranks for each layer
    Will often overestimate alpha, increase opt_factor or lower MAX_ITER to account for this

    Args:
        base_model: model to reduce
        x_opt: testing data to evaluate accuracy
        y_opt: testing data to evaluate accuracy
        
        alpha: percent of original model to initially optimize for
        opt_factor: how much a step needs to affect the accuracy 
            for the k-values to update
        MAX_ITER: maximum number of iterations for search 

    Returns:
        an optimized model
    """
    pcnt_diff = lambda xf, xi: np.abs((xf - xi) / (xi + 1e-32))
    accuracy = lambda bm, ks: model_from_ks_(bm, ks).evaluate(x_opt,  y_opt, verbose=2)[1]
    
    ks, max_ks= naive_k_finding_(base_model, alpha)
    
    acc_i = accuracy(base_model, ks)
    
    iters = 0
    # caps out at MAX_ITER
    while iters <= MAX_ITER:
        
        kupdate = [0 for _ in ks]
        for i, k in enumerate(ks):
            iters += 1
            
            # exit conditions
            if (iters > MAX_ITER) or k >= max_ks[i]:
                break
                
            # finds necessary 'steps'
            kstep = copy.deepcopy(ks) 
            kstep[i] += 1
            kupdate[i] = int(opt_factor < pcnt_diff(accuracy(base_model, kstep), acc_i))
        
        # performs the step
        if all(ku == 0 for ku in kupdate):
            break
        ks = [ku + k for ku, k in zip(kupdate, ks)]
        
    return model_from_ks_(base_model, ks)