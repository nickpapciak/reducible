import struct
import os
import uuid
import tensorflow as tf
import numpy as np
from reducible.optimize import RankKApprox

def save_model_to_file(model, filepath): 
    """
    Saves a model to a file
    Args:
        model: the optimized model to save
        filepath: the file to save it to 
            must have .gyatt file extension

    Returns:
        original filepath
    """

    if not filepath.endswith('.gyatt'): 
        raise ValueError(f'filepath must end with .gyatt.\nFilepath was \"{filepath}\"')
    
    opt_weights = []

    for layer in model.layers:
        if isinstance(layer, RankKApprox):
            opt_weights.append(layer.A.numpy())
            opt_weights.append(layer.B.numpy())
            opt_weights.append(layer.b.numpy())
        elif layer.get_weights():
            opt_weights.append(layer.get_weights()[0])
            opt_weights.append(layer.get_weights()[1])
    
    model_file = uuid.uuid4().hex + ".keras"

    opt_weights = np.array(opt_weights, dtype=object)
    opt_weights_file = uuid.uuid4().hex + ".npy"

    tf.keras.models.save_model(model, model_file)
    np.save(opt_weights_file, opt_weights)

    return merge_binary_files_(model_file, opt_weights_file, filepath)

def load_model_from_file(filepath):
    """
    Loads a model from a file
    Args:
        model: the filepath of the saved model. Must end in .gyatt file extension

    Returns:
        loaded model
    """
    model, opt_weights = unmerge_binary_file_(filepath)

    i = 0
    for layer in model.layers:
        if isinstance(layer, RankKApprox):
            layer.A = opt_weights[i]
            layer.B = opt_weights[i+1]
            layer.b = opt_weights[i+2]
            i+=3
        elif layer.get_weights():
            layer.set_weights([opt_weights[i], opt_weights[i+1]])
            i+= 1

    return model




def merge_binary_files_(file_a, file_b, file_c):
    """
    Merges two binary files into custom filetype
    Args:
        file_a: model file
        file_b: weights file
        file_c: output filepath

    Returns:
        output filepath
    """
    with open(file_a, 'rb') as fa, open(file_b, 'rb') as fb:
        content_a = fa.read()
        content_b = fb.read()
        
    # two integer representation 
    metadata = struct.pack('QQ', len(content_a), len(content_b))

    # Write metadata 
    with open(file_c, 'wb') as fc:
        fc.write(metadata)
        fc.write(content_a)
        fc.write(content_b)

    os.remove(file_a)
    os.remove(file_b)
    return file_c


def unmerge_binary_file_(merged_file):
    """
    Unmerges two binary files for custom filetype
    Args:
        merged_file: file as input

    Returns:
        model, weights
    """
    # other issues are caught by open method.
    if merged_file.split(".")[-1] != "gyatt":
        raise FileNotFoundError("No gyatt file was found.")
    
    output_file_a, output_file_b = uuid.uuid4().hex + ".keras", uuid.uuid4().hex + ".npy"
    with open(merged_file, 'rb') as f:
        metadata = f.read(struct.calcsize('QQ'))
        size_a, size_b = struct.unpack('QQ', metadata)
        content = f.read()
        content_a = content[:size_a]
        content_b = content[size_a:]
    with open(output_file_a, 'wb') as fa, open(output_file_b, 'wb') as fb:
        fa.write(content_a)
        fb.write(content_b)

    model_ = tf.keras.models.load_model(output_file_a)
    array_ = np.load(output_file_b, allow_pickle=True)
    os.remove(output_file_a)
    os.remove(output_file_b)
    return model_, array_