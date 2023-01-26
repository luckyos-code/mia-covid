import os
import tensorflow as tf
from tensorflow.keras.layers import Input, InputLayer, Conv2D, Activation, Add, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, ZeroPadding2D
from classification_models.tfkeras import Classifiers

""" ResNet [Paper](https://scholar.google.com/scholar?cluster=9281510746729853742&hl=en&as_sdt=0,5) """
def resnet_block(x, filters, kernel_size=3, stride=1,
           conv_shortcut=False, name=None):
    preact = BatchNormalization(epsilon=1.001e-5, name=name + '_preact_bn')(x)
    preact = Activation('tanh', name=name + '_preact_tanh')(preact)
    if conv_shortcut is True:
        shortcut = Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
    else:
        shortcut = MaxPooling2D(1, strides=stride, name=name + 'pool_pool')(x) if stride > 1 else x
    x = Conv2D(filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = Activation('tanh', name=name + '_1_tanh')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = Conv2D(filters, kernel_size, strides=stride, use_bias=False, name=name + '_2_conv')(x)
    x = BatchNormalization(epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = Activation('tanh', name=name + '_2_tanh')(x)
    x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = Add(name=name + '_out')([shortcut, x])
    return x

def resnet_stack(x, filters, blocks, stride1=2, name=None):
    x = resnet_block(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = resnet_block(x, filters, name=name + '_block' + str(i))
    x = resnet_block(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x

def ResNet(stack_fn, input, model_name='resnet'):
    # bottom
    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input)
    x = Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = MaxPooling2D(3, strides=2, name='pool1_pool')(x)
    # body
    x = stack_fn(x)
    # no top added
    # Create model
    model = tf.keras.models.Model(input, x, name=model_name)
    return model
    
def ResNet18_tanh(input):
    def stack_fn(x):
        x = resnet_stack(x, 64, 2, name='conv2')
        x = resnet_stack(x, 128, 2, name='conv3')
        x = resnet_stack(x, 256, 2, name='conv4')
        x = resnet_stack(x, 512, 2, stride1=1, name='conv5')
        x = BatchNormalization(epsilon=1.001e-5, name='post_bn')(x)
        x = Activation('tanh', name='post_tanh')(x)
        return x
    return ResNet(stack_fn, input, 'resnet18v2')

def resnet18_builder(name='resnet18', activation='relu', weights=None, weights_path=None, classes=2, dropout=None, img_shape=None):
    ResNet18, _ = Classifiers.get('resnet18')
    if activation == 'tanh':
        if weights == 'imagenet':
            resnet18 = ResNet18(include_top = False,
                                weights = weights,
                                input_shape = img_shape)
            for layer in resnet18.layers:
                if (hasattr(layer,'activation'))==True and layer.activation==tf.keras.activations.relu:
                    layer.activation = tf.keras.activations.tanh
        else:
            resnet18 = ResNet18_tanh(Input(shape=img_shape))
    elif activation == 'relu':
        _weights = None if weights == 'pneumonia' else weights
        resnet18 = ResNet18(include_top = False,
                            weights = _weights,
                            input_shape = img_shape)
    
    resnet18.trainable = True # make layers trainable
    
    # classification layer for binary or multiclass
    output = Dense(units=1, activation='sigmoid', name='Output') if classes==2 else Dense(units=classes, activation='softmax', name='Output')

    seq = [
        InputLayer(input_shape=img_shape, name='Input'),
        resnet18, # add resnet18 without head
        GlobalAveragePooling2D(name='AvgPool'), # add last pooling layer
        output # add output layer
    ]

    if dropout:
        seq.insert(3, Dropout(rate=dropout, name='Dropout')) # add dropout (inspired by inception-resnet-v2)

    model = tf.keras.models.Sequential(seq, name=name)

    # load weights from pneumonia pretraining
    if weights == 'pneumonia' and activation == 'relu':
        model.load_weights(os.path.join(weights_path, 'pneumonia', 'resnet18_relu_public_weights.h5'), by_name=True)
    if weights == 'pneumonia' and activation == 'tanh':
        model.load_weights(os.path.join(weights_path, 'pneumonia', 'resnet18_tanh_public_weights.h5'), by_name=True)
    
    return model

def ResNet50_tanh(input):
    def stack_fn(x):
        x = resnet_stack(x, 64, 3, name='conv2')
        x = resnet_stack(x, 128, 4, name='conv3')
        x = resnet_stack(x, 256, 6, name='conv4')
        x = resnet_stack(x, 512, 3, stride1=1, name='conv5')
        x = BatchNormalization(epsilon=1.001e-5, name='post_bn')(x)
        x = Activation('tanh', name='post_tanh')(x)
        return x
    return ResNet(stack_fn, input, 'resnet50v2')

def resnet50_builder(name='resnet50', activation='relu', weights=None, weights_path=None, classes=2, dropout=None, img_shape=None):
    ResNet50, _ = Classifiers.get('resnet50')
    if activation == 'tanh':
        if weights == 'imagenet':
            resnet50 = ResNet50(include_top = False,
                                weights = weights,
                                input_shape = img_shape)
            for layer in resnet50.layers:
                if (hasattr(layer,'activation'))==True and layer.activation==tf.keras.activations.relu:
                    layer.activation = tf.keras.activations.tanh
        else:
            resnet50 = ResNet50_tanh(Input(shape=img_shape))
    elif activation == 'relu':
        _weights = None if weights == 'pneumonia' else weights
        resnet50 = ResNet50(include_top = False,
                            weights = _weights,
                            input_shape = img_shape)
    
    resnet50.trainable = True # make layers trainable

    # classification layer for binary or multiclass
    output = Dense(units=1, activation='sigmoid', name='Output') if classes==2 else Dense(units=classes, activation='softmax', name='Output')

    seq = [
        InputLayer(input_shape=img_shape, name='Input'),
        resnet50, # add resnet50 without head
        GlobalAveragePooling2D(name='AvgPool'), # add last pooling layer
        output # add output layer
    ]

    if dropout:
        seq.insert(3, Dropout(rate=dropout, name='Dropout')) # add dropout (inspired by inception-resnet-v2)

    model = tf.keras.models.Sequential(seq, name=name)

    # load weights from pneumonia pretraining
    if weights == 'pneumonia' and activation == 'relu':
        model.load_weights(os.path.join(weights_path, 'pneumonia', 'resnet50_relu_public_weights.h5'), by_name=True)
    if weights == 'pneumonia' and activation == 'tanh':
        model.load_weights(os.path.join(weights_path, 'pneumonia', 'resnet50_tanh_public_weights.h5'), by_name=True)

    return model

def resnet_builder(architecture='resnet18', activation='relu', pretraining=None, weights_path=None, classes=2, dropout=None, img_shape=None):
    name = architecture + '-' + activation + '-' + pretraining if pretraining else architecture + '-' + activation

    if architecture=='resnet18':
        model = resnet18_builder(name=name, activation=activation, weights=pretraining, weights_path=weights_path, classes=classes, dropout=dropout, img_shape=img_shape)
    elif architecture=='resnet50':
        model = resnet50_builder(name=name, activation=activation, weights=pretraining, weights_path=weights_path, classes=classes, dropout=dropout, img_shape=img_shape)

    model_info = {
        'model_name': name,
        'architecture': architecture,
        'pretraining': pretraining,
        'dropout': dropout,
        'activation': activation,
        'classes': classes,
        'img_shape': img_shape,
    }

    return model, model_info
