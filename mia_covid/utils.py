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


def check_create_folder(dir: str):
    """Check if a folder exists on the current file, if not, this function creates that folder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    check_dir = os.path.join(base_dir, dir)
    if not os.path.exists(check_dir):
        print(f"Directory {check_dir} does not exist, creating it")
        os.mkdir(check_dir)


def get_img(x, y):
    """Load image from path and return together with label."""
    path = x
    label = y
    # load the raw data from the file as a string
    img = tf.io.read_file(path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    return img, label