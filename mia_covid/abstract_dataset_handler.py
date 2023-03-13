import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.utils import class_weight

from tensorflow.keras.layers import Layer, Resizing, Rescaling, RandomFlip, RandomRotation, RandomTranslation, RandomZoom
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List, Union
import scipy.stats
import os
import glob


class GrayscaleToRgb(Layer):
    """Layer for converting 1-channel grayscale input to 3-channel rgb."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return tf.image.grayscale_to_rgb(x)


class RandomBrightness(Layer):
    """Layer for random brightness augmentation in images."""

    def __init__(self, factor=0.1, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return tf.image.random_brightness(x, max_delta=self.factor)


class ModelPreprocessing(Layer):
    """Layer for specific model preprocessing steps."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return tf.keras.applications.resnet50.preprocess_input(x)


@dataclass(eq=True, frozen=False)
class AbstractDataset():
    # image shape needed for the model (used for rescaling)
    dataset_name: str
    dataset_path: Optional[str]
    # shape that the dataset should be transformed to
    model_img_shape: Tuple[int, int, int]
    train_val_test_split: Tuple[float, float, float]
    batch_size: int
    convert_rgb: bool
    augment_train: bool
    # TODO: use a functional approach here and save a reference to a preprocessing function passed when instantiating the class
    resnet50_preprocessing: bool
    shuffle: bool
    is_tfds_ds: bool
    # if True, automatically builds ds_info after loading dataset data
    builds_ds_info: bool = field(default=False, repr=False)

    imbalance_ratio: Optional[float] = None
    variants: Optional[List[Dict]] = None

    # image shape of the original dataset data (currently has no real function but interesting to know)
    dataset_img_shape: Optional[Tuple[int, int, int]] = None
    # optionally providable class_names, only for cosmtig purposes when printing out ds_info
    class_names: Optional[List[str]] = None
    random_rotation: float = 0.1
    random_zoom: float = 0.15
    random_flip: str = "horizontal"
    random_brightness: float = 0.1
    random_translation_width: float = 0.1
    random_translation_height: float = 0.1
    random_seed: int = 42
    repeat: bool = False

    class_labels: Optional[Tuple[Any]] = None
    class_counts: Optional[Tuple[int]] = None
    class_distribution: Optional[Tuple[int]] = None

    # TODO: make this a dataclass instead of dict -> since we need the attributes well defined through the code
    ds_info: Dict[str, Any] = field(init=False, default_factory=dict)
    ds_train: tf.data.Dataset = field(init=False, repr=False)
    ds_val: tf.data.Dataset = field(init=False, repr=False, default=None)
    ds_test: Optional[tf.data.Dataset] = field(init=False, repr=False, default=None)

    ds_attack_train: tf.data.Dataset = field(init=False, repr=False, default=None)
    ds_attack_test: tf.data.Dataset = field(init=False, repr=False, default=None)

    def _load_dataset(self):
        """Load dataset from tfds library.

        This function should be overwritten by all classes which do not utilize the tfds library to load the dataset.
        Overwrite this function with the needed functionality to load the dataset from files. Then call the 'load_dataset()' function to bundle
        data loading and dataset info creation.
        """
        if self.is_tfds_ds:
            self.__load_from_tfds()

    def load_dataset(self):
        print(f"Loading {self.dataset_name}")
        self._load_dataset()
        if self.builds_ds_info:
            self.build_ds_info()

    def __load_from_tfds(self):
        """Load dataset from tensorflow_datasets via 'dataset_name'."""
        if not self.is_tfds_ds:
            print("Cannot load dataset from tfds since it is not a tfds dataset!")
            return

        ds_train: Optional[tf.data.Dataset] = None
        ds_val: Optional[tf.data.Dataset] = None
        ds_test: Optional[tf.data.Dataset] = None

        train_split = self.train_val_test_split[0] * 100
        val_split = self.train_val_test_split[1] * 100
        test_split = self.train_val_test_split[2] * 100

        if self.dataset_path is not None:
            data_dir = os.path.join(self.dataset_path, self.dataset_name)
        else:
            data_dir = None

        if train_split > 0:
            ds_train = tfds.load(
                name=self.dataset_name,
                split=f"train[0%:{train_split:.0f}%]",
                data_dir=data_dir,
                as_supervised=True,
                with_info=False
            )

        if test_split > 0:
            ds_test = tfds.load(
                name=self.dataset_name,
                split=f"test[0%:{train_split:.0f}%]",
                data_dir=data_dir,
                as_supervised=True,
                with_info=False
            )

        if val_split > 0:
            ds_val = tfds.load(
                name=self.dataset_name,
                split=f"validation[0%:{train_split:.0f}%]",
                data_dir=data_dir,
                as_supervised=True,
                with_info=False
            )

        if ds_train is not None:
            self.ds_train = ds_train
        if ds_val is not None:
            self.ds_val = ds_val
        if ds_test is not None:
            self.ds_test = ds_test

    def set_class_names(self, class_names: List[str]):
        self.class_names = class_names

    def prepare_datasets(self):
        """Prepare all currently stored datasets (train, val, test) and the corresponding attack datsets (train, test).

        Preparation can include data shuffling, augmentation and resnet50-preprocessing.
        Augmentation is applied to train dataset if specified, augmentation is never applied to validation or test dataset

        """
        # prepare attack datasets
        # we need to first prepare the attack DS since they depend on the unmodified original datasets
        self.ds_attack_train = self.prepare_ds(self.ds_train, cache=True, resize_rescale=True, img_shape=self.model_img_shape, batch_size=1, convert_rgb=self.convert_rgb, preprocessing=self.resnet50_preprocessing, shuffle=False, augment=False)
        if self.ds_test is not None:
            self.ds_attack_test = self.prepare_ds(self.ds_test, cache=True, resize_rescale=True, img_shape=self.model_img_shape, batch_size=1, convert_rgb=self.convert_rgb, preprocessing=self.resnet50_preprocessing, shuffle=False, augment=False)
        elif self.ds_val is not None:
            self.ds_attack_test = self.prepare_ds(self.ds_val, cache=True, resize_rescale=True, img_shape=self.model_img_shape, batch_size=1, convert_rgb=self.convert_rgb, preprocessing=self.resnet50_preprocessing, shuffle=False, augment=False)

        dataset_path: str = ""
        # prepare non-attack datasets
        if self.dataset_path:
            dataset_path = self.dataset_path

        self.ds_train = self.prepare_ds(self.ds_train, cache=os.path.join(dataset_path, 'data.tfcache.' + self.dataset_name), resize_rescale=True, img_shape=self.model_img_shape, batch_size=self.batch_size, convert_rgb=self.convert_rgb, preprocessing=self.resnet50_preprocessing, shuffle=self.shuffle, augment=self.augment_train)

        if self.ds_val is not None:
            self.ds_val = self.prepare_ds(self.ds_val, cache=True, resize_rescale=True, img_shape=self.model_img_shape, batch_size=self.batch_size, convert_rgb=self.convert_rgb, preprocessing=self.resnet50_preprocessing, shuffle=False, augment=False)

        if self.ds_test is not None:
            self.ds_test = self.prepare_ds(self.ds_test, cache=True, resize_rescale=True, img_shape=self.model_img_shape, batch_size=self.batch_size, convert_rgb=self.convert_rgb, preprocessing=self.resnet50_preprocessing, shuffle=False, augment=False)

    def prepare_ds(self, ds: tf.data.Dataset, resize_rescale: bool, img_shape: Tuple[int, int, int], batch_size: Optional[int], convert_rgb: bool, preprocessing: bool, shuffle: bool, augment: bool, cache: Union[str, bool] = True) -> tf.data.Dataset:
        """Prepare datasets for training and validation for the ResNet50 model.

        This function applies image resizing, resnet50-preprocessing to the dataset. Optionally the data can be shuffled or further get augmented (random flipping, etc.)

        Parameter
        --------
        ds: tf.data.Dataset - dataset used for preparation steps
        resize_rescale: bool - if True, resizes the dataset to 'img_shape' and rescales all pixel values to a value between 0 and 255
        img_shape: Tuple[int, int, int] - if resize_rescale is True, than this value is used to rescale the image data to this size, consist of [height, width, color channel] -> only width and height are used for rescaling
        batch_size: int | None - batch size specified by integer value, if None is passed, no batching is applied to the data
        convert_rgb: bool - if True, the data is converted vom grayscale to rgb values
        preprocessing: bool - if True, model specific preprocessing is applied to the data (currently resnet50_preprocessing)
        shuffle: bool - if True, the data is shuffled, the used shuffle buffer for this has the size of the data
        augment: bool - if True, data augmentation (random flip, random rotation, random translation, random zoom, random brightness) is applied to the data

        """
        AUTOTUNE = tf.data.AUTOTUNE

        preprocessing_layers = tf.keras.models.Sequential()
        if convert_rgb:
            preprocessing_layers.add(GrayscaleToRgb())

        if resize_rescale:
            preprocessing_layers.add(Resizing(img_shape[0], img_shape[1]))
            preprocessing_layers.add(Rescaling(scale=1. / 255))

        # TODO: replace static set resnet50 preprocessing function with dynamic prepcrocessing layer function
        if preprocessing:
            preprocessing_layers.add(ModelPreprocessing())

        if convert_rgb or resize_rescale or preprocessing:
            ds = ds.map(lambda x, y: (preprocessing_layers(x), y),
                        num_parallel_calls=AUTOTUNE)

        if cache:
            if isinstance(cache, str):
                reset_cache = False
                if reset_cache:
                    for filename in glob.glob(cache + '*'):
                        os.remove(filename)
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        if shuffle:
            ds = ds.shuffle(buffer_size=ds.cardinality().numpy(), seed=self.random_seed)

        if batch_size is not None:
            ds = ds.batch(batch_size)

        if augment:
            augmenter = tf.keras.Sequential([
                RandomFlip(self.random_flip),
                RandomRotation(self.random_rotation, fill_mode="constant"),
                RandomTranslation(self.random_translation_height, self.random_translation_width, fill_mode="constant"),
                RandomZoom(self.random_zoom, fill_mode="constant"),
                RandomBrightness(self.random_brightness),
            ])
            ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)

        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=AUTOTUNE)

    def calculate_class_weights(self) -> Tuple[Optional[Dict[int, int]], Optional[Dict[int, float]]]:
        """Calculate class weights and class counts for train dataset."""
        class_labels, class_counts, class_distribution = self.get_class_distribution()

        class_counts_dict: Dict[str, int] = {}
        for y, count in zip(class_labels, class_counts):
            if self.class_names is not None and len(self.class_names) == len(class_labels):
                class_counts_dict[f"{self.class_names[y]}({y})"] = count
            else:
                class_counts_dict[y] = count

        weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(class_distribution),
                                                    y=class_distribution)

        class_weights: Dict[str, float] = {}
        if self.class_names is not None and len(self.class_names) == len(class_labels):
            for i, weight in enumerate(weights):
                class_weights[f"{self.class_names[y]}({y})"] = weight
        else:
            class_weights = dict(enumerate(weights))
        return (class_counts_dict, class_weights)

    def get_dataset_count(self) -> Dict[str, int]:
        """Calculate number of datapoints for each part of the dataset (train,test,val)."""
        ds_count: Dict[str, int] = defaultdict(int)
        if self.ds_train is not None:
            ds_count["train"] = self.ds_train.cardinality().numpy()

        if self.ds_val is not None:
            ds_count["val"] = self.ds_val.cardinality().numpy()

        if self.ds_test is not None:
            ds_count["test"] = self.ds_test.cardinality().numpy()

        return ds_count

    def get_class_distribution(self, ds: Optional[tf.data.Dataset] = None, force_recalcuation: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate and return absolute class distribution from train dataset.

        This function returns the desired class_labels, class_counts and class_distribution values but also sets these variables as class variables.
        This is useful to not execute the function again, but only return the class variables, unless the 'force_recalcuation' flag is set to True.

        Parameter:
        --------
        ds: tf.data.Dataset - an optional dataset can be given to this function to calculate the class distribution of the given datasets
                              if ds is not set, it is assumed to calculate the class distribution from the current train dataset
        force_recalcuation: bool - (default = False), if set to True, this function calculates the class_distribution again

        Return:
        ------
        (np.ndarray, np.ndarray, np.ndarray): three numpy arrays
            -> first one containing the class number
            -> second one containing the number of datapoints in the class (ordered)
            -> third one as a class representation for all datapoints
        f.e.: ([1,2,3,4,5],[404,133,313,122,10], [4,1,0,2,5,4,1,4,3,2,4,3,3,1,...])

        """
        if self.class_counts is not None and self.class_labels is not None and self.class_distribution is not None and force_recalcuation is not True:
            return (self.class_labels, self.class_counts, self.class_distribution)

        if ds is not None:
            y_train = np.fromiter(ds.map(lambda _, y: y), int)
        else:
            y_train = np.fromiter(self.ds_train.map(lambda _, y: y), int)

        distribution = np.unique(y_train, return_counts=True)

        self.class_labels = distribution[0]
        self.class_counts = distribution[1]
        self.class_distribution = y_train

        return distribution + (y_train,)

    def calculate_class_imbalance(self) -> float:
        """Calculate class imbalance value for the train dataset.

        Idea of using shannon entropy to calculate balance from here: https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
        For data of n instances, if we have k classes of size c_i, we can calculate the entropy

        Return:
        ------
        float:  class imbalance value [0,1]
                0 - unbalanced dataset
                1 - balanced dataset

        """
        _, class_counts, _ = self.get_class_distribution()

        n: int = sum(class_counts)
        k: int = len(class_counts)
        H: float = 0.0
        for c in class_counts:
            H += (c / n) * np.log((c / n))

        H *= -1
        B: float = H / np.log(k)
        return B

    def calculate_data_entropy(self, ds: Optional[tf.data.Dataset] = None) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Calculate and return data entropy values and normed entropy values.

        Parameter:
        --------
        ds: tf.data.Dataset - Optional, if passed, the entropy of the given dataset is calculated instead of the current training dataset

        Return:
        ------
        1 (float, float, float) : average entropy value, min entropy value, max entropy value
        2 (float, float, float) : normed average entropy value, normed min entropy value, normed max entropy value

        """
        entropy_val_list: List[float] = []
        normed_entropy_val_list: List[float] = []

        if ds is None:
            ds = self.ds_train

        for _, (data, _) in enumerate(ds):
            values, counts = np.unique(data, return_counts=True)
            entropy_val = scipy.stats.entropy(counts)
            entropy_val_list.append(entropy_val)

            normed_entropy_val = entropy_val / np.log(len(values))
            normed_entropy_val_list.append(normed_entropy_val)

        max_entropy = max(entropy_val_list)
        min_entropy = min(entropy_val_list)
        avg_entropy = sum(entropy_val_list) / len(entropy_val_list)

        normed_max_entropy = max(normed_entropy_val_list)
        normed_min_entropy = min(normed_entropy_val_list)
        normed_avg_entropy = sum(normed_entropy_val_list) / len(normed_entropy_val_list)

        return ((avg_entropy, min_entropy, max_entropy), (normed_avg_entropy, normed_min_entropy, normed_max_entropy))

    def build_ds_info(self):
        """Build dataset info dictionary.

        This function needs to be called after initializing and loading the dataset
        """
        class_counts, class_weights = self.calculate_class_weights()
        ds_count = self.get_dataset_count()
        total_count: int = sum(ds_count.values())
        class_imbalance: float = self.calculate_class_imbalance()
        entropy_values, normed_entropy_values = self.calculate_data_entropy()

        self.ds_info = {
            'name': self.dataset_name,
            'dataset_img_shape': self.dataset_img_shape,
            'model_img_shape': self.model_img_shape,
            'total_count': total_count,
            'train_count': ds_count["train"],
            'val_count': ds_count["val"],
            'test_count': ds_count["test"],
            'classes': len(class_counts),
            'class_imbalance': class_imbalance,
            'class_counts': class_counts,
            'class_weights': class_weights,
            'avg_entropy': entropy_values[0],
            'min_entropy': entropy_values[1],
            'max_entropy': entropy_values[2],
            'normed_avg_entropy': normed_entropy_values[0],
            'normed_min_entropy': normed_entropy_values[1],
            'normed_max_entropy': normed_entropy_values[2],
        }
