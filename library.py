import keras as kr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import os
import pickle
import gzip
import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from i3d_inception import Inception_Inflated3d, conv3d_bn
from keras.ops import mean
from custom_stgcn import Model as STGCNModel
from keras.layers import Dropout, Reshape, Lambda, Activation
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.utils import Sequence, to_categorical

def baseline_model_i3d_flow(
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-7
) -> Model:
    base_model = Inception_Inflated3d(
        include_top=False,
        weights="flow_kinetics_only",
        input_shape=(64, 224, 224, 2),
    )

    x = base_model.output

    x = Dropout(0.5, name='Dropout_5a')(x)

    x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same', 
                    use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    num_frames = int(x.shape[1])  # Number of frames in the input video
    x = Reshape((num_frames, num_classes))(x)

    x = Lambda(lambda x: mean(x, axis=1, keepdims=False),
                output_shape=lambda s: (s[0], s[2]))(x)
    
    x = Activation('softmax', name='prediction')(x)

    model_flow = Model(inputs=base_model.input, outputs=x, name='Inception_Inflated3d')

    model_flow.compile(
        optimizer=kr.optimizers.Adam(learning_rate, weight_decay),
        loss=kr.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    return model_flow

def baseline_model_stgcn(
    num_classes: int,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4
):
    model_pose = STGCNModel(num_classes=num_classes)

    dummy_input = np.random.normal(size=(1, 3, 64, 46, 1))  # Note: 3 channels (x,y,z coordinates)
    _ = model_pose(dummy_input, training=True)  # This builds the model

    # # add activation layer softmax
    # model_pose = Activation('softmax', name='prediction')(model_pose)

    model_pose.compile(
        optimizer=kr.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=kr.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model_pose

def baseline_model_i3d(
        num_classes: int, 
        learning_rate: float = 1e-3, 
        weight_decay: float = 1e-7
        ) -> Model:
    base_model = Inception_Inflated3d(
        include_top=False,
        weights="rgb_imagenet_and_kinetics",
        input_shape=(64, 224, 224, 3),
    )

    # turn on training for the base model
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output

    x = Dropout(0.5, name='Dropout_5a')(x)

    x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same', 
                    use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    num_frames = int(x.shape[1])  # Number of frames in the input video
    x = Reshape((num_frames, num_classes))(x)

    x = Lambda(lambda x: mean(x, axis=1, keepdims=False),
                output_shape=lambda s: (s[0], s[2]))(x)

    x = Activation('softmax', name='prediction')(x)

    model_face = Model(inputs=base_model.input, outputs=x, name='Inception_Inflated3d')

    model_face.compile(
        optimizer=kr.optimizers.Adam(learning_rate, weight_decay),
        loss=kr.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    return model_face