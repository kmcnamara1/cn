import tensorflow as tf
import argparse
import os
import numpy as np
import json
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import categorical_crossentropy  

import numpy as np

INPUT_TENSOR_NAME = "inputs_input" # Watch out, it needs to match the name of the first layer + "_input"
BATCH_SIZE = 64
HEIGHT = 224
WIDTH = 224
DEPTH = 3

def model_compile(learning_rate):
    model = Sequential()
    model.add(Conv2D(input_shape=(HEIGHT,WIDTH,DEPTH),filters=64,kernel_size=(3,3),padding="same",activation="relu",name="inputs"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=8, activation="softmax"))

    opt = Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def model_fit(model, data_dir, epoch, batch_size):
    
    train_datagen = ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
    
    train_generator = train_datagen.flow_from_directory(data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size, shuffle=True,)    
    
    model.fit(train_generator, epochs=epoch)

    return model


def _load_training_data(data_dir, batch_size):
    assert os.path.exists(data_dir), ("Unable to find images resources for input, are you sure you downloaded them ?")

    datagen = ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
    generator = datagen.flow_from_directory(data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size, shuffle=True,)
    images, labels = generator.next()

    return {INPUT_TENSOR_NAME: images}, labels

def _load_testing_data(data_dir, batch_size):
    
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size, shuffle=True,)
    images, labels = generator.next()

    return {INPUT_TENSOR_NAME: images}, labels

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)

    return parser.parse_known_args()

if __name__ == "__main__":
    args, unknown = _parse_args()
#     train_data, train_labels = _load_training_data(args.train, args.batch_size)
#     eval_data, eval_labels = _load_testing_data(args.train, args.batch_size)  
    raw_classifier = model_compile(args.learning_rate)
    view_classifier = model_fit(raw_classifier, args.train, args.epochs, args.batch_size)
    
#     if args.current_host == args.hosts[0]:
#         view_classifier.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')
        
        
# def keras_model_fn(hyperparameters):

#     model = Sequential()
#     model.add(Conv2D(input_shape=(HEIGHT,WIDTH,DEPTH),filters=64,kernel_size=(3,3),padding="same",activation="relu",name="inputs"))
#     model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
#     model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
#     model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
#     model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
#     model.add(Dropout(0.5))
#     model.add(Flatten())
#     model.add(Dense(units=4096,activation="relu"))
#     model.add(Dense(units=4096,activation="relu"))
#     model.add(Dense(units=8, activation="softmax"))

#     opt = Adam(lr=0.001)
#     model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])
#     return model


# def serving_input_fn(hyperparameters):
#     tensor = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, DEPTH])
#     inputs = {INPUT_TENSOR_NAME: tensor}
#     return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    
# def train_input_fn(training_dir, hyperparameters):
#     return _input(tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE, data_dir=training_dir)


# def eval_input_fn(training_dir, hyperparameters):
#     return _input(tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE, data_dir=training_dir)


# def _input(mode, batch_size, data_dir):
#     assert os.path.exists(data_dir), ("Unable to find images resources for input, are you sure you downloaded them ?")

#     if mode == tf.estimator.ModeKeys.TRAIN:
#         datagen = ImageDataGenerator(
#             rescale= 1./255,
#             shear_range=0.2,
#             zoom_range=0.2,
#             horizontal_flip=True
#         )
#     else:
#         datagen = ImageDataGenerator(rescale=1. / 255)

#     generator = datagen.flow_from_directory(data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size)
#     images, labels = generator.next()

#     return {INPUT_TENSOR_NAME: images}, labels