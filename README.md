# 306projectknn
# Mushroom Classification - K-Nearest Neighbors (KNN) Model & random forest model

## Project Overview
This project classifies mushrooms as either edible or poisonous using machine learning models. The goal is to apply classification techniques to a mushroom dataset and compare model performances. This part of the project focuses on the **K-Nearest Neighbors (KNN)** model and the random forest model.

## Dataset
We use the **Mushroom Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Mushroom). The dataset includes 22 features describing mushroom properties, and the target variable is whether the mushroom is **edible** or **poisonous**. The dataset file is provided in this repo as agaricus-lepiota.data.

## K-Nearest Neighbors (KNN) Model (Mariana Hernandez)

### Implementation Steps:
1. **Data Preprocessing**: Clean the data, handle missing values, and encode categorical features using `LabelEncoder`.
2. **Model Training**: Train the KNN model using `KNeighborsClassifier` from `sklearn`.
3. **Model Evaluation**: Evaluate the model's performance with metrics like **accuracy**, **precision**, **recall**, and **F1-score**.
4. **Hyperparameter Tuning**: Experiment with different values of `k` and distance metrics (Euclidean, Manhattan, Minkowski).
5. **Cross-validation**: Ensure robustness by performing cross-validation.

### Expected Outcome
The KNN model will be evaluated based on its classification accuracy and other metrics, and its performance will be compared to Decision Tree and Random Forest models to identify the best classifier.

## Random Forest Model (Mahnoor Aftab)

### Implementation Steps:
1. **Data Preprocessing**: The dataset is preprocessed by handling missing values and encoding categorical features using OneHotEncoder.
2. **Model Training**: A Random Forest Classifier from sklearn.ensemble is trained using 100 estimators (trees).
3. **Model Evaluation**: The model's accuracy is evaluated. A classification report and confusion matrix are generated to showcase performance.
4. **Feature Importance**: Feature importances are calculated.

### Expected Outcome:
The Random Forest model aims to acheive an accuracy of 100% on data classification,  provides feature importances to show the most important features for classifying mushrooms.

## Team Members:
- **Sophia**: Decision Tree Model
- **Mahnoor**: Random Forest Model
- **Mariana**: K-Nearest Neighbors (KNN) Model 

## References:
- Mushroom Dataset: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Mushroom)
