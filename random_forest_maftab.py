# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = 'agaricus-lepiota.data'
df = pd.read_csv(file_path, header=None)

# Add column names
df.columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
              'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
              'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
              'stalk-surface-below-ring', 'stalk-color-above-ring',
              'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
              'ring-type', 'spore-print-color', 'population', 'habitat']

# Replace '?' with NaN and handle missing data
df.replace('?', np.nan, inplace=True)

# Handle missing values using SimpleImputer (impute with the most frequent value)
imputer = SimpleImputer(strategy='most_frequent')  # Impute with the most frequent value
df.iloc[:, :] = imputer.fit_transform(df)

# Display first 5 rows and check for missing values
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

# Histograms separated by class labels
for column in df.columns:
    if column != 'class':
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x=column, hue='class', multiple='dodge', shrink=0.8)
        plt.title(f'Histogram of {column} (Separated by Class)')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f"histogram_{column}_by_class.png")  # Save
        plt.show()

# Histograms for the overall dataset
for column in df.columns:
    if column != 'class':
        plt.figure(figsize=(8, 6))
        df[column].value_counts().plot(kind='bar', color='blue', alpha=0.7)
        plt.title(f'Histogram of {column} (Overall)')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f"histogram_{column}_overall.png")  # Save
        plt.show()

# Separate features and target variable
X = df.drop('class', axis=1)
y = df['class']

# Stratified train-test split for balanced class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# One-hot encode categorical features independently for train and test
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# Train Random Forest with more restrictive parameters and cross-validation
rf = RandomForestClassifier(random_state=42)

# Adjusting the hyperparameters further to reduce overfitting
param_grid = {
    'n_estimators': [50, 100],  # Limit the number of estimators
    'max_depth': [3, 5, 7],  # Further reduce depth of trees
    'min_samples_split': [5, 10],  # Increase the number of samples to split nodes
    'min_samples_leaf': [2, 5],  # Increase the number of samples in each leaf node
    'max_features': ['sqrt', 'log2'],  # Limit the number of features each tree uses
}

# Perform GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
grid_search.fit(X_encoded, y)

# Best parameters and model
best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Cross-validation scores
cv_scores = cross_val_score(best_rf, X_encoded, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.2f}%")

# Fit the best model on the full training set
X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)
best_rf.fit(X_train_encoded, y_train)

# Evaluate the model on the test set
rf_y_pred = best_rf.predict(X_test_encoded)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred, labels=best_rf.classes_)

# Random Forest metrics
rf_accuracy = accuracy_score(y_test, rf_y_pred)

print("Random Forest Model Evaluation:")
print(f"Accuracy on Test Set: {rf_accuracy * 100:.2f}%")

# Classification Report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))

# Confusion Matrix for Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['edible', 'poisonous'], yticklabels=['edible', 'poisonous'])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")  # Saves the plot as an image
plt.show()  # Displays the plot
