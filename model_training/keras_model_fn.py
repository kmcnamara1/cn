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

def model_fit(model, train_data_dir, valid_data_dir, epoch, batch_size):
    
    train_datagen = ImageDataGenerator(rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
    
    train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size, shuffle=True,) 

    validation_generator = train_datagen.flow_from_directory(valid_data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size, shuffle=True,)    
    
    history = model.fit(train_generator, epochs=epoch, validation_data=validation_generator)

    print("History:")
    print(history)

    return model, history


def _load_training_data(train_data_dir, batch_size):
    assert os.path.exists(train_data_dir), ("Unable to find images resources for input, are you sure you downloaded them ?")

    datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
    generator = datagen.flow_from_directory(train_data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size, shuffle=True,)
    images, labels = generator.next()

    return {INPUT_TENSOR_NAME: images}, labels

def _load_testing_data(test_data_dir, batch_size):
    
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(test_data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size, shuffle=True,)
    images, labels = generator.next()

    return {INPUT_TENSOR_NAME: images}, labels

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--validation', type=str)

    return parser.parse_known_args()

if __name__ == "__main__":
    args, unknown = _parse_args()
#     train_data, train_labels = _load_training_data(args.train, args.batch_size)
#     eval_data, eval_labels = _load_testing_data(args.train, args.batch_size)  
    raw_model = model_compile(args.learning_rate)
    view_classifier, history = model_fit(raw_model, args.train, args.validation, args.epochs, args.batch_size)
    
    
#     if args.current_host == args.hosts[0]:
#         view_classifier.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')
        
