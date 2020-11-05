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

def model_fit(model, train_data_dir, epoch, batch_size):
    
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size, shuffle=True,)    
    
#     history = model.fit(train_generator, epochs=epoch)
    model.fit(train_generator, epochs=epoch)
#     print("History:")
#     print(history)

    return model


def _load_training_data(train_data_dir, batch_size):
    assert os.path.exists(train_data_dir), ("Unable to find images resources for input, are you sure you downloaded them ?")

    datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
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
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    return parser.parse_known_args()

def main():
    args, unknown = _parse_args()
    raw_model = model_compile(args.learning_rate)
    view_classifier = model_fit(raw_model, args.train, args.epochs, args.batch_size)
    
    model_dir  = args.model_dir
    sess = K.get_session()
    tf.saved_model.simple_save(
            tf.Session(),
            os.path.join(model_dir, 'model/1'),
            inputs={'inputs': model.input},
            outputs={t.name: t for t in model.outputs})

#     view_classifier.save(os.path.join(args.model_dir, '000000001'), 'my_model.h5')
    
if __name__ == "__main__":
    main()
    
#     if args.current_host == args.hosts[0]:
#     view_classifier.save(args.model_dir)
#     view_classifier.save(os.path.join(args.model_dir, '000000001'), 'my_model.h5')
        
