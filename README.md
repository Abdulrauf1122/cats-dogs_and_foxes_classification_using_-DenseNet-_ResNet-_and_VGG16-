# cats-dogs_and_foxes_classification_using_-DenseNet-_ResNet-_and_VGG16-
This project uses deep learning models (DenseNet, ResNet, VGG16) to classify images into three animal categories: Cats, Dogs, and Foxes. The models are pre-trained on ImageNet and fine-tuned using transfer learning for this classification task. It uses Keras and TensorFlow for model development and evaluation.

# Cats, Dogs, and Foxes Image Classification using Deep Learning

This project uses deep learning models (DenseNet, ResNet, and VGG16) to classify images into three animal categories: **Cats**, **Dogs**, and **Foxes**. The models are pre-trained on ImageNet and fine-tuned using transfer learning for this classification task. The project uses **Keras** and **TensorFlow** for model development and evaluation.

## Dataset

The dataset used for this project is the [Animal Image Dataset: Cats, Dogs, and Foxes](https://www.kaggle.com/datasets/snmahsa/animal-image-dataset-cats-dogs-and-foxes) available on Kaggle. You can download the dataset from Kaggle and use it to train and evaluate the models.

The dataset has the following folder structure:
dataset/ ├── train/ │ ├── cats/ │ ├── dogs/ │ └── foxes/ ├── test/ │ ├── cats/ │ ├── dogs/ │ └── foxes/ └── val/ ├── cats/ ├── dogs/ └── foxes/

csharp
Copy code

## Installation Instructions

To get started with this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cats-dogs-foxes-classification.git
Navigate to the project folder:

bash
Copy code
cd cats-dogs-foxes-classification
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Download the dataset from Kaggle:

Animal Image Dataset: Cats, Dogs, and Foxes
Make sure your dataset is structured as follows:

bash
Copy code
dataset/
├── train/
│   ├── cats/
│   ├── dogs/
│   └── foxes/
├── test/
│   ├── cats/
│   ├── dogs/
│   └── foxes/
└── val/
    ├── cats/
    ├── dogs/
    └── foxes/
Open and run the train_and_evaluate.ipynb notebook. This will handle the training, evaluation, and prediction tasks.

Usage
Once the models are trained, you can use them to evaluate their performance on the test dataset and make predictions on new images.

Evaluate the models:
The notebook includes code to evaluate the models on the test dataset and print the classification report and confusion matrix.

Make Predictions:
The notebook also includes functionality to make predictions on new images, simply by providing the image path.

Model Details
This project uses three pre-trained models:

DenseNet121: A deep convolutional network known for its efficient architecture. Pre-trained on ImageNet.
ResNet101V2: A deep residual network that helps in overcoming the vanishing gradient problem. Pre-trained on ImageNet.
VGG16: A simpler model with 16 layers, pre-trained on ImageNet.
All models were fine-tuned on the custom dataset (Cats, Dogs, and Foxes) using transfer learning.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contributing
Feel free to fork the repository, submit pull requests, and open issues. Contributions are welcome to improve the model or add new features!
