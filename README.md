# MNIST Handwritten Digit Recognition

## Project Overview

This project is a machine learning system developed to recognize handwritten digits from **0 to 9** using the **MNIST dataset**.

The project follows a complete machine learning workflow, including dataset loading, image visualization, data preprocessing, model training, evaluation, model comparison, and testing on new handwritten digit images.

Two machine learning models were implemented and evaluated:

- Logistic Regression
- Random Forest Classifier

## Project Objective

The main objectives of this project are:

- Load and understand the MNIST dataset
- Visualize handwritten digit images
- Analyze pixel values
- Normalize image pixel values
- Split the dataset into training and testing sets
- Train machine learning classification models
- Evaluate model performance
- Analyze confusion matrices and classification reports
- Compare different machine learning models
- Test the trained model on new handwritten digit images

## Dataset

The project uses the **MNIST handwritten digit dataset**.

The dataset contains:

- **70,000** handwritten digit images
- **10 classes**, representing digits from 0 to 9
- Each image has a size of **28 × 28 pixels**
- Each image contains **784 pixel features**
- Original pixel values range from **0 to 255**

The dataset was loaded using Scikit-learn's `fetch_openml()` function.

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml(
    'mnist_784',
    version=1,
    as_frame=False,
    cache=True
)
