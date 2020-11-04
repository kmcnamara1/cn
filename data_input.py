import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def _input(mode, batch_size, data_dir):
    assert os.path.exists(data_dir), ("Unable to find images resources for input, are you sure you downloaded them ?")

    if mode == tf.estimator.ModeKeys.TRAIN:
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size)
    images, labels = generator.next()

    return {INPUT_TENSOR_NAME: images}, labels