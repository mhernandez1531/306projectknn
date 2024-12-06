# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import graphviz

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
imputer = SimpleImputer(strategy='most_frequent')  # Impute with the most frequent value
df.iloc[:, :] = imputer.fit_transform(df)

# Separate features and target variable
X = df.drop('class', axis=1)
y = df['class']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)  # Updated for latest version
X_encoded = encoder.fit_transform(X)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_encoded, y)

# Generate predictions and confusion matrix
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
y_pred = rf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred, labels=rf.classes_)

# Model Evaluation
print("Random Forest Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['edible', 'poisonous'], yticklabels=['edible', 'poisonous'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # Saves the plot as an image
plt.show()  # Displays the plot
