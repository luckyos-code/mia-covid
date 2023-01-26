import os, glob
import gdown
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Layer, Resizing, Rescaling, RandomFlip, RandomRotation, RandomTranslation, RandomZoom
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.experimental.AUTOTUNE

"""
Load image from path and return together with label
"""
def get_img(x, y):
    path = x
    label = y
    # load the raw data from the file as a string
    img = tf.io.read_file(path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    return img, label

"""
Layer for converting 1-channel grayscale input to 3-channel rgb
"""
class GrayscaleToRgb(Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, x):
    return tf.image.grayscale_to_rgb(x)

"""
Layer for random brightness augmentation in images
"""
class RandomBrightness(Layer):
  def __init__(self, factor=0.1, **kwargs):
    super().__init__(**kwargs)
    self.factor = factor
  def call(self, x):
    return tf.image.random_brightness(x, max_delta=self.factor)

"""
Prepare and batch a dataset
"""
def prepare_dataset(ds, cache=True, pre=True, convert_rgb=False, img_shape=(224,224,3), shuffle=False, batch_size=1, augment=False):
    # Resize and rescale images
    # - before cache for reusing
    if pre:
        preprocessing = tf.keras.models.Sequential()
        if convert_rgb:
            preprocessing.add(GrayscaleToRgb())
        preprocessing.add(Resizing(img_shape[0], img_shape[1]))
        preprocessing.add(Rescaling(scale=1./255))

        ds = ds.map(lambda x, y: (preprocessing(x), y),
                    num_parallel_calls=AUTOTUNE)

    # give string to cache for saving outside of memory on disk
    # - before random operations to avoid caching randomness
    if cache:
        if isinstance(cache, str):
            reset_cache = False
            if reset_cache:
                for filename in glob.glob(cache+'*'):
                    os.remove(filename)
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    
    # shuffle dataset
    # - after cache since randomness
    # - full dataset size as buffer for true randomness
    if shuffle:
        ds = ds.shuffle(buffer_size=ds.cardinality().numpy())
    
    # batch dataset 
    # - after shuffling to get unique batches at each epoch.
    # - before augmentation for vectorization
    if batch_size:
        ds = ds.batch(batch_size)
    
    # use data augmentation - before augmentation for better speed from batch-wise vecotrization
    if augment:
        data_augmentation = tf.keras.models.Sequential([
          RandomFlip("horizontal"),
          RandomRotation(0.1, fill_mode='constant'),
          RandomTranslation(0.1, 0.1, fill_mode='constant'),
          RandomZoom(0.15, fill_mode='constant'),
          RandomBrightness(0.1),
        ])

        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                    num_parallel_calls=AUTOTUNE)
        
    return ds.prefetch(buffer_size=AUTOTUNE)

"""
Prepare and return all needed datset parts
"""
def get_prepared_datasets(dataset_name, data_path, train_ds, val_ds, test_ds, convert_rgb, img_shape, batch_size, augment_train):
    train_batched = prepare_dataset(
        ds=train_ds,
        cache=os.path.join(data_path, 'data.tfcache.'+dataset_name),
        pre=True,
        convert_rgb=convert_rgb,
        img_shape=img_shape,
        shuffle=True,
        batch_size=batch_size,
        augment=augment_train)

    if val_ds:
        val_batched = prepare_dataset(
            ds=val_ds,
            cache=True,
            pre=True,
            convert_rgb=convert_rgb,
            img_shape=img_shape,
            shuffle=False,
            batch_size=batch_size,
            augment=False)
    else:
        val_batched = None

    test_batched = prepare_dataset(
        ds=test_ds,
        cache=True,
        pre=True,
        convert_rgb=convert_rgb,
        img_shape=img_shape,        
        shuffle=False,
        batch_size=batch_size,
        augment=False)

    # set for attack on train set
    train_attack_data = prepare_dataset(
        ds=train_ds,
        cache=True,
        pre=True,
        convert_rgb=convert_rgb,
        img_shape=img_shape,        
        shuffle=False,
        batch_size=1,
        augment=False)

    # set for attack on test set
    test_attack_data = prepare_dataset(
        ds=test_ds,
        cache=True,
        pre=True,
        convert_rgb=convert_rgb,
        img_shape=img_shape,        
        shuffle=False,
        batch_size=1,
        augment=False)

    return train_batched, val_batched, test_batched, train_attack_data, test_attack_data

"""
COVID-19 Radiography Database
[Kaggle](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
[Paper 1](https://ieeexplore.ieee.org/document/9144185)
[Paper 2](https://doi.org/10.1016/j.compbiomed.2021.104319)
"""
def get_covid(dataset_name, data_path, img_shape, batch_size, val_test_split=(0.05, 0.15), imbalance_ratio=1.5, random_seed=42):
    # download dataset
    dataset_path = os.path.join(data_path, 'COVID-19_Radiography_Dataset')
    if not os.path.exists(dataset_path):
        url = 'https://drive.google.com/uc?id=1ZMgUQkwNqvMrZ8QaQmSbiDqXOWAewwou&confirm=t'
        output = os.path.join(data_path, 'COVID-19_Radiography_Database.zip')
        gdown.cached_download(url, output, quiet=False, use_cookies=False, postprocess=gdown.extractall)
        os.remove(output)

    # collect from image paths
    normal_imgpath = os.path.join(dataset_path, 'Normal')
    covid_imgpath = os.path.join(dataset_path, 'COVID')
    
    excl_imgs = ['Normal-'+str(i)+'.png' for i in range(8852, 10192+1)] # excluding duplicates normal images from pneumonia pre-training
    
    normal_images = [os.path.join(normal_imgpath, name) for name in os.listdir(normal_imgpath) if os.path.isfile(os.path.join(normal_imgpath, name)) and name not in excl_imgs]
    covid_images = [os.path.join(covid_imgpath, name) for name in os.listdir(covid_imgpath) if os.path.isfile(os.path.join(covid_imgpath, name))]

    # create train-test split
    label_encoding = ['normal', 'COVID-19'] # normal = 0, COVID-19 = 1
    files, labels= [], []
    
    np.random.shuffle(covid_images)
    files.extend(covid_images)
    labels.extend(np.full(len(covid_images), label_encoding.index('COVID-19')))

    np.random.shuffle(normal_images)
    if imbalance_ratio:
        normal_images = normal_images[:int(imbalance_ratio*len(covid_images))]
    files.extend(normal_images)
    labels.extend(np.full(len(normal_images), label_encoding.index('normal')))

    files, labels = np.array(files), np.array(labels)

    val_split = val_test_split[0]
    test_split = val_test_split[1]
    x_train, x_rest, y_train, y_rest = train_test_split(files, labels, test_size=val_split+test_split, random_state=random_seed)
    x_test, x_val, y_test, y_val = train_test_split(x_rest, y_rest, test_size=val_split/(val_split+test_split), random_state=random_seed)

    # build tensorflow dataset
    train_files = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_files = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_files = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # image retrieval
    train_ds = train_files.map(get_img, num_parallel_calls=AUTOTUNE)
    val_ds = val_files.map(get_img, num_parallel_calls=AUTOTUNE)
    test_ds = test_files.map(get_img, num_parallel_calls=AUTOTUNE)

    # class statistics
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y_train),
                                                      y=y_train)
    class_weights = dict(enumerate(class_weights))
    
    train_batched, val_batched, test_batched, train_attack_data, test_attack_data = get_prepared_datasets(
        dataset_name=dataset_name,
        data_path=data_path,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        convert_rgb=False,
        img_shape=img_shape,
        batch_size=batch_size,
        augment_train=True)

    ds_info = {
        'name': dataset_name,
        'img_shape': img_shape,
        'total_count': len(files),
        'class_counts': {'normal (0)': len(normal_images), 'COVID-19 (1)': len(covid_images)},
        'train_count': len(y_train),
        'val_count': len(y_val),
        'test_count': len(y_test),
        'class_weights': class_weights
    }
    
    return ds_info, train_batched, val_batched, test_batched, train_attack_data, test_attack_data

"""
MNIST Database
[Website](http://yann.lecun.com/exdb/mnist/)
"""
def get_from_tfds(dataset_name, data_path, img_shape, batch_size): #TODO options for split, shuffle_files (imagenet), validation set, convert_rgb, augment_train, class_weights
    # load tfds dataset
    (train_ds, test_ds), tfds_info = tfds.load(
        name=dataset_name,
        split=['train', 'test'],
        data_dir=os.path.join(data_path, dataset_name),
        as_supervised=True,
        with_info=True,
    )

    # class statistics
    y_train = np.fromiter(train_ds.map(lambda x, y: y), int)
    distribution = np.unique(y_train, return_counts=True)

    class_counts = {}
    for y, count in zip(*distribution):
        class_counts[y] = count
    
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y_train),
                                                      y=y_train)
    class_weights = dict(enumerate(class_weights))
    
    train_batched, val_batched, test_batched, train_attack_data, test_attack_data = get_prepared_datasets(
        dataset_name=dataset_name,
        data_path=data_path,
        train_ds=train_ds,
        val_ds=False,
        test_ds=test_ds, 
        convert_rgb=True,
        img_shape=img_shape,
        batch_size=batch_size,
        augment_train=False)

    ds_info = {
        'name': dataset_name,
        'img_shape': img_shape,
        'total_count': tfds_info.splits.total_num_examples,
        'class_counts': class_counts,
        'train_count': tfds_info.splits['train'].num_examples,
        'val_count': None,
        'test_count': tfds_info.splits['test'].num_examples,
        'class_weights': None #TODO use class_weights?
    }
    
    return ds_info, train_batched, val_batched, test_batched, train_attack_data, test_attack_data

"""
Loader for correct dataset builders
"""
def load_dataset(setting):
    if not os.path.exists(setting.data_path):
        os.mkdir(setting.data_path)
    
    if setting.dataset_name == 'covid':
        return get_covid(dataset_name=setting.dataset_name,
                         data_path=setting.data_path,
                         img_shape=setting.img_shape,
                         batch_size=setting.batch_size,
                         val_test_split=setting.val_test_split,
                         imbalance_ratio=setting.imbalance_ratio,
                         random_seed=setting.random_seed)
    elif setting.dataset_name == 'mnist':
        return get_from_tfds(dataset_name=setting.dataset_name,
                             data_path=setting.data_path,
                             img_shape=setting.img_shape,
                             batch_size=setting.batch_size)

"""
General Dataset class for variable structure
"""
class Dataset:
    def __init__(self, setting):
        self.info, self.train_batched, self.val_batched, self.test_batched, self.train_attack_data, self.test_attack_data = load_dataset(setting)
