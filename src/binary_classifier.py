import os
import zipfile
from typing import List

import keras
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import wget
from keras.preprocessing.image import ImageDataGenerator

#  Training Parameters
NUM_EPOCHS = 15
NUM_BATCHES = 8
LEARNING_RATE = 0.001
CALLBACK_MIN_ACCURACY = 0.99

#  Image Metadata
INPUT_SHAPE = (300, 300, 3)
TARGET_SIZE = (300, 300)


# Download data as zip and extract to directory.
def dwnld_to_dir(url, data_root_dir):
    file_name = url.split('/')[-1]
    dir_name = file_name.split('.')[0]
    img_root_dir = data_root_dir + dir_name
    try:
        dwn_load = wget.download(url, out=data_root_dir)
        print(f"Downloaded: {dwn_load} file\n")

        local_zip = data_root_dir + file_name
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall(img_root_dir)
        print(f"Extracted data to {img_root_dir}\n")

        zip_ref.close()
    except Exception as e:
        print(f"Error downloading from {url}: {e}")


# Show 10 images from each class.
def show_img_grid(class_one_dir, class_two_dir):
    img_names_class_one: List = os.listdir(class_one_dir)
    img_names_class_two: List = os.listdir(class_two_dir)

    # Parameters for our graph; we'll output images in a 4x5 configuration
    nrows = 5
    ncols = 5

    # Index for iterating over images
    pic_index = 10

    # Set up matplotlib fig, and size.
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    # A list of 10 file paths to images from class one.
    class_one_pix_paths: List = [os.path.join(class_one_dir, fname)
                                 for fname in img_names_class_one[pic_index - 10:pic_index]]

    # A list of 10 file paths to images from class two.
    class_two_pix_paths: List = [os.path.join(class_two_dir, fname)
                                 for fname in img_names_class_two[pic_index - 10:pic_index]]

    for i, img_path in enumerate(class_one_pix_paths + class_two_pix_paths):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        ith_img = mpimg.imread(img_path)
        plt.imshow(ith_img)

    plt.show()


# Create CNN with 5 convolutions, 5 poolings and 2 dense layers.
def create_model():
    """
        Notice the pattern, as we make the image smaller with pooling we increase the
        number of filters i.e. increasing the number of output channels for the next layer.
    """
    return keras.models.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        # This is binary classification 0=horse and 1=human.
        keras.layers.Dense(1, activation='sigmoid')
    ])


# Compile model using RMSprop optimizer.
def compile_model(the_model, learning_rate=LEARNING_RATE):
    the_model.compile(
        loss='binary_crossentropy',
        # RMSprop, automatically tunes the learning-rate.
        # (Note, non-legacy runs slow on M2 mac's).
        optimizer=keras.optimizers.legacy.RMSprop(learning_rate),
        metrics=['accuracy']
    )
    return the_model


class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > CALLBACK_MIN_ACCURACY:
            print(f"\nTraining halted: Model exceeded minimum accuracy threshold of"
                  f" {CALLBACK_MIN_ACCURACY*100}%")
            self.model.stop_training = True


# ImageDataGenerator, streams images from source img_dir in batches of
# batch_size generating labels on the fly; e.g. 0=horse and 1=human.
def data_gen_helper(img_dir, batch_size):
    return ImageDataGenerator(rescale=1 / 255).flow_from_directory(
        img_dir,
        target_size=TARGET_SIZE,  # All images will be resized to 300x300.
        batch_size=batch_size,
        class_mode='binary'  # Used binary_crossentropy loss in 'compile' => 'binary' labels.
    )


def train_model(the_model, training_dir, validation_dir):
    training_data_gen = data_gen_helper(training_dir, 128)
    validation_data_gen = data_gen_helper(validation_dir, 32)

    the_model.fit(
        training_data_gen,
        steps_per_epoch=NUM_BATCHES,  # Specifies how many batches will determine one epoch.
        epochs=NUM_EPOCHS,
        verbose=1,
        callbacks=MyCallback(),
        validation_data=validation_data_gen,
        validation_steps=NUM_BATCHES
    )
    return the_model


# Load, resize and normalise an image for classification.
def format_img(img_path):
    # Load image into PIL format.
    img = keras.utils.load_img(img_path, target_size=TARGET_SIZE)
    # Converts from PIL to numpy array.
    img = keras.utils.img_to_array(img)
    # Normalise in range [0, 1].
    img /= 255
    # Add dim for batch size; (300, 300, 3) -> (1, 300, 300, 3)
    return np.expand_dims(img, axis=0)


# Classify the image.
def classify(the_model, img, class_desc: List):
    # Classify the image - feeding 10 pix at a time through to the network.
    classes = the_model.predict(img, batch_size=10)

    # It's binary therefore, only one value in array, sigmoid -> [0, 1].
    if classes[0] > 0.5:
        return class_desc[1]
    else:
        return class_desc[0]
