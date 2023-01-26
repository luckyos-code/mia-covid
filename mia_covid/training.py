from mia_covid.evaluation import getMetrics

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasAdamOptimizer

"""
Compile models with non-private or private settings
"""
def compileModel(model, dataset, setting):
    metrics = getMetrics(dataset.info)
    
    if not(setting.target_eps):
        optimizer = Adam(setting.learning_rate)
        if len(dataset.info['class_counts']) == 2:
            loss = 'binary_crossentropy'
        elif len(dataset.info['class_counts']) > 2:
            loss = 'sparse_categorical_crossentropy'
    else:
        optimizer = VectorizedDPKerasAdamOptimizer(
            l2_norm_clip=setting.l2_clip,
            noise_multiplier=setting.noise,
            num_microbatches=setting.batch_size,
            learning_rate=setting.learning_rate
        )

        if len(dataset.info['class_counts']) == 2:
            loss = BinaryCrossentropy(
                from_logits=True,
                reduction=tf.compat.v1.losses.Reduction.NONE
                # reduction is set to NONE to get loss in a vector form
            )
        elif len(dataset.info['class_counts']) > 2:
            loss = SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.compat.v1.losses.Reduction.NONE
                # reduction is set to NONE to get loss in a vector form
            )
            
    model.compile(optimizer, loss, metrics)
    #model.summary()

    return model

"""
Train
"""
def trainModel(model, dataset, setting, callbacks=[]):
    if dataset.info['val_count']:
        learning_rate_decay = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.1, min_lr=1e-6)
    elif not dataset.info['val_count']:
        learning_rate_decay = ReduceLROnPlateau(monitor='loss', patience=2, factor=0.1, min_lr=1e-6)

    print("Training %s ..." % (model.name))
    history = model.fit(
        dataset.train_batched,
        epochs=setting.epochs,
        validation_data=dataset.val_batched, # might be None
        class_weight=dataset.info['class_weights'], # might be None
        callbacks=[learning_rate_decay, *callbacks],
        verbose=2,
    )
    print('\n')

    return model, history
