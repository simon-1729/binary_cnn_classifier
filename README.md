# Binary CNN Classifier

### Details
A simple binary convolutional neural network built using TensorFlow.
The CNN comprises 5 convolution, 5 pooling and 2 dense, layers.
The main application will attempt to download two classes of images,
horses and humans, roughly 500 of each - this may take a few minutes.
The application will then create, compile, train and save the model.
Note, that the entire model i.e. architecture, weights and optimizer
etc., are saved. 

Should you wish to experiment with the model i.e. make changes to the
src/binary_classifier create_model() function. Remember to update the
value for SAVED_MODEL_NAME property in main.py file. Otherwise, the 
application will load and use the previously saved model.

Once trained the application will attempt to classify all the previously 
unseen images in the data/imgs_to_classify folder.

Note, you can test this yourself by adding your own pictures to this folder.
Obviously, they should be either  pictures of a horse or a human.

### Setup (virtual environment and install dependencies) 
###### mac m2, Python 3.9.6
```bash
> cd binary_cnn_classifier
> pyenv install 3.9.6
> pyenv local  3.9.6
> python -m venv env
> source ./env/bin/activate
> pip install -r requirements.txt
```

### Start application

```bash
> python main.py
```
___

### Exit virtual environment 

```bash
> deactivate
```

