# Breast-Cancer-Machine-learning-
Learning how to use Scikit-learn and develop three different machine learning models.

# Breast Cancer Prediction Using Machine Learning Models

## Overview

This project aims to predict breast cancer outcomes using machine learning classification models. Using  Scikit-learn's built-in breast cancer dataset, which contains serveral features of cell nuclei, and the goal is to classify the tumors as either benign or malignant. Through three different machine learning models  **Logistic Regression**, **Random Forest**, and **Decision Tree**, and evaluate their performance using multiple metrics. The best-performing model is identified and analyzed in detail.

## Purpose

The purpose of this project is to use machine learning techniques to predict the presence of breast cancer based on the provided dataset. Early detection can significantly improve survial rating, and by classifying the tumors as either benign or malignant, these models can help in making timely and accurate medical decisions. The models are evaluated using several performance metrics such as accuracy, precision, recall, F1 score, and ROC-AUC to determine their effectiveness in real-world applications.

## Dataset

The dataset used is the **Breast Cancer Wisconsin dataset** from Scikit-learn. It includes the following:

- **Features**: Various measurements of cell nuclei from breast cancer biopsies.
- **Target**: A binary target indicating whether the tumor is benign (0) or malignant (1).

## Libraries Used 
- Numpy as np
- Pandas as pd
- Sklearn

## Sklearn Modules Used 
- **datasets**:  Provides built-in datasets
- **metrics**: measurements 
- **model_selection**: split data into testing and training 
- **preprocessing**: cleans and prepares data 
- **linear_model**:making predictions 
- **ensemble**: combines models 
- **tree**: decision tree models 


## Models Used

- **Logistic Regression**: predict whether something belongs to one of two categories (by likeloodhood) 
- **Random Forest**: creates a group of decision trees and combines their results to make a final prediction
- **Decision Tree**: makes predictions by splitting data into smaller and smaller groups based on certain features

## Evaluation Metrics

The following metrics are used to evaluate the models:

- **Accuracy**: How often the model is correct in its predictions.
- **Precision**: How many of the positive predictions the model made are actually correct.
- **Recall**: How many of the actual positive cases the model correctly identified.
- **F1 Score**: How well the model does overall.
- **ROC-AUC**: How well model can distinguish between classes.

## Project Structure

- **Data Preprocessing**: 
  - The data is split into training (80%) and testing (20%) sets.
  - Features are standardized to have a mean of 0 and variance of 1.
  
- **Model Evaluation**:
  - Each model is trained on the scaled training data and evaluated on the testing data.
  - Model performance is compared using the metrics mentioned above.
  
- **Feature Importance**:
  - Indicates how much each feature contributes to the model's ability to make accurate predictions.
  - Features with higher importance have a greater impact on the model's decision-making process.

## Limitations 
 - Limited data may not represent all types or conditions of breast cancer
 - Data set size is on the smaller end and may limit the models
 - There could be overfiting and the models might struggle on new data 
   
