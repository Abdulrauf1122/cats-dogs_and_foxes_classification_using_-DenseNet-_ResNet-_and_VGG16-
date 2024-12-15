# cats-dogs_and_foxes_classification_using_-DenseNet-_ResNet-_and_VGG16-
This project uses deep learning models (DenseNet, ResNet, VGG16) to classify images into three animal categories: Cats, Dogs, and Foxes. The models are pre-trained on ImageNet and fine-tuned using transfer learning for this classification task. It uses Keras and TensorFlow for model development and evaluation.

# Cats, Dogs, and Foxes Image Classification using Deep Learning

This project classifies images of **Cats**, **Dogs**, and **Foxes** using deep learning models. The models are pre-trained on ImageNet and fine-tuned using transfer learning for this classification task. The project utilizes **TensorFlow** and **Keras** for model development and evaluation.

The models used in this project include:
- **DenseNet121**
- **ResNet101V2**
- **VGG16**

The dataset for this classification task is the [Animal Image Dataset: Cats, Dogs, and Foxes](https://www.kaggle.com/datasets/snmahsa/animal-image-dataset-cats-dogs-and-foxes), which is openly available on Kaggle.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Installation Instructions](#installation-instructions)
3. [Usage](#usage)
4. [Model Details](#model-details)
5. [Evaluation and Prediction](#evaluation-and-prediction)
6. [License](#license)
7. [Contributing](#contributing)
8. [Contact](#contact)

---

## Dataset

The dataset used for training and evaluation is the **Animal Image Dataset: Cats, Dogs, and Foxes**, available on Kaggle. You can download the dataset from the following link:

- [Download Dataset from Kaggle](https://www.kaggle.com/datasets/snmahsa/animal-image-dataset-cats-dogs-and-foxes)

### Dataset Structure

The dataset should be structured as follows:

dataset/ 
├── train/  
    ├── cats/
    ├── dogs/ 
    └── foxes/ 
├── test/ 
   ├── cats/ 
   ├── dogs/ 
   └── foxes/ 
└── val/ 
   ├── cats/ 
   ├── dogs/ 
   └── foxes/

## Installation Instructions

Follow these steps to get started with the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/cats-dogs-foxes-classification.git
Navigate to the project folder:

bash
Copy code
cd cats-dogs-foxes-classification
Install the required dependencies: The project requires several Python packages. You can install them using pip:

bash
Copy code
pip install -r requirements.txt
Download the dataset:

Visit the dataset page: Animal Image Dataset: Cats, Dogs, and Foxes and download it.
Place the dataset in a folder named dataset with the structure mentioned above.
Usage
Once the dataset is ready, you can run the Jupyter Notebook to train and evaluate the models.

Steps to Use:
Open the train_and_evaluate.ipynb notebook.
Run the notebook step by step. The notebook will handle:
Data Preprocessing
Model Training using DenseNet121, ResNet101V2, and VGG16
Model Evaluation (Classification Report and Confusion Matrix)
Prediction on new images
Running the Notebook
To start the Jupyter notebook, execute the following:

bash
jupyter notebook
Then open the main.ipynb notebook and run the cells in order.

##Model Details
This project uses three pre-trained models for image classification:

1. DenseNet121:
Architecture: Dense convolutional network.
Key Features: Efficient feature reuse and improved training efficiency.
Transfer Learning: Fine-tuned on the custom dataset.
2. ResNet101V2:
Architecture: Deep residual network.
Key Features: Solves vanishing gradient problem with skip connections.
Transfer Learning: Fine-tuned on the custom dataset.
3. VGG16:
Architecture: A simple, yet effective deep convolutional network.
Key Features: 16 layers for feature extraction.
Transfer Learning: Fine-tuned on the custom dataset.

##Evaluation and Prediction
Model Evaluation:
The notebook will evaluate each model on the test dataset and output:

##Classification Report (precision, recall, f1-score)
Confusion Matrix (to visualize model performance)
Making Predictions:
The notebook includes functionality to make predictions on new images. You can upload an image and the model will classify it as either Cat, Dog, or Fox.

##License
This project is licensed under the MIT License. See the LICENSE file for more details.

##Contributing
Feel free to fork the repository, submit pull requests, and open issues. Contributions are welcome to improve the model, add new features, or suggest improvements!
