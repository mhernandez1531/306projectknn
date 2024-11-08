# 306projectknn
# Mushroom Classification - K-Nearest Neighbors (KNN) Model

## Project Overview
This project classifies mushrooms as either edible or poisonous using machine learning models. The goal is to apply classification techniques to a mushroom dataset and compare model performances. This part of the project focuses on the **K-Nearest Neighbors (KNN)** model.

## Dataset
We use the **Mushroom Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Mushroom). The dataset includes 22 features describing mushroom properties, and the target variable is whether the mushroom is **edible** or **poisonous**.

## K-Nearest Neighbors (KNN) Model

### Implementation Steps:
1. **Data Preprocessing**: Clean the data, handle missing values, and encode categorical features using `LabelEncoder`.
2. **Model Training**: Train the KNN model using `KNeighborsClassifier` from `sklearn`.
3. **Model Evaluation**: Evaluate the model's performance with metrics like **accuracy**, **precision**, **recall**, and **F1-score**.
4. **Hyperparameter Tuning**: Experiment with different values of `k` and distance metrics (Euclidean, Manhattan, Minkowski).
5. **Cross-validation**: Ensure robustness by performing cross-validation.

### Expected Outcome
The KNN model will be evaluated based on its classification accuracy and other metrics, and its performance will be compared to Decision Tree and Random Forest models to identify the best classifier.

## Team Members:
- **Sophia**: Decision Tree Model
- **Mahnoor**: Random Forest Model
- **Mariana**: K-Nearest Neighbors (KNN) Model (this file)

## References:
- Mushroom Dataset: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Mushroom)
