import os
from typing import List

import keras

import src.binary_classifier as bc

"""
    ______________________________________________________________________________________________
                Binary Classifier
    ----------------------------------------------------------------------------------------------
    This application downloads two classes of images, humans and horses, approx 500 of each.
    The images are used to train a Convolutional Neural Network which will then attempt to
    classify previously unseen image as either human or horse.
    ______________________________________________________________________________________________
"""


TRAINING_URL = 'https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip'
VALIDATION_URL = 'https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip'
ROOT_DATA_DIR = './data/'
TRAINING_DIR = ROOT_DATA_DIR + TRAINING_URL.split('/')[-1].split('.')[0]
VALIDATION_DIR = ROOT_DATA_DIR + VALIDATION_URL.split('/')[-1].split('.')[0]
SAVE_MODEL_DIR = './saved_models/'
SAVED_MODEL_NAME = 'my_model_v1'
TEST_IMG_DIR = './data/imgs_to_classify'
CLASS_DESC = ['horse', 'human']

if __name__ == '__main__':

    # Download training data if the directory doesn't exist.
    if not (os.path.exists(TRAINING_DIR) and os.path.isdir(TRAINING_DIR)):
        print(f"Downloading zip and extracting to {TRAINING_DIR} directory\n")
        bc.dwnld_to_dir(TRAINING_URL, ROOT_DATA_DIR)

    # The horse and human image directories used for training the cnn.
    train_horse_dir = os.path.join(TRAINING_DIR, CLASS_DESC[0])
    train_human_dir = os.path.join(TRAINING_DIR, CLASS_DESC[1])

    # Display an image grid with 10 training pictures from each class.
    # bc.show_img_grid(train_horse_dir, train_human_dir)

    # Download validation data if the directory doesn't exist.
    if not (os.path.exists(VALIDATION_DIR) and os.path.isdir(VALIDATION_DIR)):
        print(f"Downloading zip and extracting to {VALIDATION_DIR} directory\n")
        bc.dwnld_to_dir(VALIDATION_URL, ROOT_DATA_DIR)

    validation_horse_dir = os.path.join(VALIDATION_DIR, CLASS_DESC[0])
    validation_human_dir = os.path.join(VALIDATION_DIR, CLASS_DESC[1])

    # Display an image grid with 10 validation pictures from each class.
    # bc.show_img_grid(validation_horse_dir, validation_human_dir)

    # Create, train and save model if it doesn't exist.
    if not (os.path.exists(SAVE_MODEL_DIR + SAVED_MODEL_NAME)):

        print("Building model... \n")
        my_model = bc.create_model()

        my_model.summary()

        print("Compiling model... \n")
        my_model = bc.compile_model(my_model)

        # Typically, takes about 2 min on mac-M2.
        print("Training model... \n")
        my_model = bc.train_model(my_model, TRAINING_DIR, VALIDATION_DIR)

        print(f"Saving model as {SAVED_MODEL_NAME}\n")
        # Save the entire model (architecture, weights, and optimizer state)
        my_model.save(SAVE_MODEL_DIR + SAVED_MODEL_NAME)

    # Otherwise, load model from disk.
    else:
        print("Loading model... \n")
        my_model = keras.models.load_model(SAVE_MODEL_DIR + SAVED_MODEL_NAME)

    # Load and classify all image from 'imgs_to_classify' directory.
    images_to_classify: List = os.listdir(TEST_IMG_DIR)
    for img_name in images_to_classify:
        img_path = TEST_IMG_DIR + '/' + img_name
        formatted_img = bc.format_img(img_path)
        classification_result = bc.classify(my_model, formatted_img, CLASS_DESC)

        print(f"\nThe image {img_name} has been classified as a {classification_result}!\n")
