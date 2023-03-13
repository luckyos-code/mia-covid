from mia_covid.abstract_dataset_handler import AbstractDataset
from mia_covid.evaluation import getMetrics

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasAdamOptimizer

from mia_covid.settings import run_settings


def compileModel(model, dataset: AbstractDataset, setting: run_settings):
    """Compile models with non-private or private settings"""
    metrics = getMetrics(dataset.ds_info)

    if not (setting.privacy.target_eps):
        optimizer = Adam(setting.commons.learning_rate)
        if len(dataset.ds_info['class_counts']) == 2:
            loss = 'binary_crossentropy'
        elif len(dataset.ds_info['class_counts']) > 2:
            loss = 'sparse_categorical_crossentropy'
    else:
        optimizer = VectorizedDPKerasAdamOptimizer(
            l2_norm_clip=setting.commons.l2_clip,
            noise_multiplier=setting.privacy.noise,
            num_microbatches=setting.model_setting.batch_size,
            learning_rate=setting.commons.learning_rate
        )

        if len(dataset.ds_info['class_counts']) == 2:
            loss = BinaryCrossentropy(
                from_logits=True,
                reduction=tf.compat.v1.losses.Reduction.NONE
                # reduction is set to NONE to get loss in a vector form
            )
        elif len(dataset.ds_info['class_counts']) > 2:
            loss = SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.compat.v1.losses.Reduction.NONE
                # reduction is set to NONE to get loss in a vector form
            )

    model.compile(optimizer, loss, metrics)
    # model.summary()

    return model


def trainModel(model, dataset: AbstractDataset, setting: run_settings, callbacks=[]):
    """Train"""
    if dataset.ds_info['val_count']:
        learning_rate_decay = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.1, min_lr=1e-6)
    elif not dataset.ds_info['val_count']:
        learning_rate_decay = ReduceLROnPlateau(monitor='loss', patience=2, factor=0.1, min_lr=1e-6)

    steps_per_epoch = None if setting.dataset.imbalance_ratio else dataset.ds_info['train_count'] // setting.model_setting.batch_size

    print("Training %s ..." % (model.name))
    history = model.fit(
        dataset.ds_train,
        epochs=setting.commons.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=dataset.ds_val,  # might be None
        class_weight=dataset.ds_info['class_weights'],  # might be None
        callbacks=[learning_rate_decay, *callbacks],
        verbose=2,
    )
    print('\n')

    return model, history
