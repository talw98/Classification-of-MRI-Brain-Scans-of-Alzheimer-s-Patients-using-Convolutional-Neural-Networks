# Classification of MRI Brain Scans of Alzheimer's Patients using Convolutional Neural Networks
Using CNN to classify MRI Images of brains of Alzheimer's patients

# Alzheimer's Disease Classification

## Background
This repository focuses on a Convolutional Neural Network (CNN) model for classifying MRI brain scans into four classes related to Alzheimer's disease. The model aims to distinguish between the following categories:
1. Mild Demented
2. Moderate Demented
3. Non Demented
4. Very Mild Demented

The dataset used for training and evaluation is the [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images) available on Kaggle.

Furthermore. the project was done on Google Colab.

## Dataset
The Alzheimer's Disease Dataset comprises MRI images of patients categorized into the four classes mentioned above. The dataset's structure includes separate folders for training and testing, each containing subdirectories for different classes. An exploration of the dataset reveals imbalances in the number of samples across classes, a factor considered during model training.

## Model Training
The CNN model is constructed using TensorFlow and Keras. The architecture involves convolutional and separable convolutional layers, batch normalization, max-pooling, and dense layers. The model is trained on the training dataset, with a subset reserved for validation. Due to class imbalances, the Area Under the ROC Curve (AUC) is chosen as a metric instead of accuracy.

The training process is optimized using the Adam optimizer, and a learning rate schedule is implemented. Model checkpoints and early stopping callbacks are employed to save the best weights and prevent overfitting.

## Model Evaluation
The trained model is evaluated on a separate test dataset, providing the following scores:
- Loss: 0.9144
- AUC: 0.8405

The loss of 0.9144 signifies the average loss over all samples, with lower values indicating better performance. The AUC of 0.8405 demonstrates the model's ability to discriminate between different Alzheimer's disease classes. These scores collectively indicate that the trained model performs reasonably well in classifying Alzheimer's disease based on MRI brain scans.

## Conclusion
The model shows promising performance in classifying Alzheimer's disease based on MRI scans. Further refinement, hyperparameter tuning, and potentially addressing class imbalances could enhance its accuracy. This project serves as a foundation for leveraging deep learning in medical image analysis, contributing to the ongoing efforts in Alzheimer's disease research.


Dataset Link: [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)
