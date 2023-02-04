# Visual Studio AI Number Detection Project

A machine learning project that aims to detect numbers present in an image using artificial intelligence. This project is built using TensorFlow and Keras libraries in Python. It employs a convolutional neural network (CNN) to identify and classify the numbers in an image. The model is trained on a large dataset of handwritten digits and can recognize numbers with high accuracy. 

## Requirements
- Python 3.x
- Tensorflow
- Keras
- OpenCV
- Numpy
- Matplotlib (optional for visualizing results)

## Installation
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a virtual environment and activate it. This is optional but highly recommended to avoid any package conflicts with other projects.
4. Run `pip install -r requirements.txt` to install all the required packages.

## Usage
1. Run the `main.py` file to start the model training process. The model training process can take a while depending on the size of your dataset and the power of your machine. 
2. You can modify the hyperparameters such as the number of epochs, batch size, etc. in the `main.py` file. A higher number of epochs will result in better accuracy, but will also increase the training time.
3. The model will be saved in the `models` directory after training.
4. Run the `predict.py` file to make predictions on new images. The script accepts a path to an image as a command line argument.

## Note
The model is just a demo version and may not perform well on all images. Further improvements can be made by using more data, modifying the model architecture, and fine-tuning the hyperparameters. Additionally, the training dataset used in this project is limited in size, so the model may not be able to generalize well to new images that it has not seen before. To achieve the best results, it is recommended to train the model on a larger and more diverse dataset.
