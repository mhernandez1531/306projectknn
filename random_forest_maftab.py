# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)  # Updated for latest version
X_encoded = encoder.fit_transform(X)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_encoded, y)

# Train K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_encoded, y)

# Train Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_encoded, y)

# Generate predictions and confusion matrix for Random Forest
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Random Forest predictions
rf_y_pred = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred, labels=rf.classes_)

# K-NN predictions
knn_y_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
knn_precision = classification_report(y_test, knn_y_pred, output_dict=True)['accuracy']
knn_recall = classification_report(y_test, knn_y_pred, output_dict=True)['macro avg']['recall']
knn_f1 = classification_report(y_test, knn_y_pred, output_dict=True)['macro avg']['f1-score']

# Decision Tree predictions
dt_y_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_precision = classification_report(y_test, dt_y_pred, output_dict=True)['accuracy']
dt_recall = classification_report(y_test, dt_y_pred, output_dict=True)['macro avg']['recall']
dt_f1 = classification_report(y_test, dt_y_pred, output_dict=True)['macro avg']['f1-score']

# Random Forest metrics
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_precision = classification_report(y_test, rf_y_pred, output_dict=True)['accuracy']
rf_recall = classification_report(y_test, rf_y_pred, output_dict=True)['macro avg']['recall']
rf_f1 = classification_report(y_test, rf_y_pred, output_dict=True)['macro avg']['f1-score']

# Model Evaluation and Accuracy Comparison
print("Random Forest Model Evaluation:")
print(f"Accuracy: {rf_accuracy * 100:.2f}%")
print("\nK-NN Model Evaluation:")
print(f"Accuracy: {knn_accuracy * 100:.2f}%")
print("\nDecision Tree Model Evaluation:")
print(f"Accuracy: {dt_accuracy * 100:.2f}%")

# Classification Report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))

print("\nK-NN Classification Report:")
print(classification_report(y_test, knn_y_pred))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_y_pred))

# Confusion Matrix Visualization for Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['edible', 'poisonous'], yticklabels=['edible', 'poisonous'])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")  # Saves the plot as an image
plt.show()  # Displays the plot

# Accuracy, Precision, Recall, F1 Score Comparison Chart
models = ['Random Forest', 'K-NN', 'Decision Tree']
accuracies = [rf_accuracy, knn_accuracy, dt_accuracy]
precision_scores = [rf_precision, knn_precision, dt_precision]
recall_scores = [rf_recall, knn_recall, dt_recall]
f1_scores = [rf_f1, knn_f1, dt_f1]

# Create the comparison bar chart
bar_width = 0.2
index = range(len(models))

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(index, accuracies, bar_width, label='Accuracy')
bar2 = ax.bar([i + bar_width for i in index], precision_scores, bar_width, label='Precision')
bar3 = ax.bar([i + 2 * bar_width for i in index], recall_scores, bar_width, label='Recall')
bar4 = ax.bar([i + 3 * bar_width for i in index], f1_scores, bar_width, label='F1 Score')

ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of KNN, Decision Tree, and Random Forest Models')
ax.set_xticks([i + 1.5 * bar_width for i in index])
ax.set_xticklabels(models)
ax.legend()
plt.tight_layout()

# Save the comparison chart as a .png image
plt.savefig("model_comparison_chart.png")  # Saves the comparison chart as an image
plt.show()  # Displays the comparison chart
