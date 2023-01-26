import numpy as np
import tensorflow as tf

def set_random_seeds(random_seed=42):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    #print(np.random.get_state()[1][:2])

def merge_dataclasses(*args):
    """
    an object's __dict__ contains all its 
    attributes, methods, docstrings, etc.
    """
    base_obj = args[0]
    for obj in args[1:]:
        base_obj.__dict__.update(obj.__dict__)
    return base_obj