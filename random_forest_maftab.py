# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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

# Separate features and target variable
X = df.drop('class', axis=1)
y = df['class']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)  # Updated for latest version
X_encoded = encoder.fit_transform(X)

# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_encoded, y)

# Generate predictions and confusion matrix
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
y_pred_rf = rf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred_rf, labels=rf.classes_)

# K-Nearest Neighbors Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Decision Tree Model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Model Evaluation for Random Forest
print("Random Forest Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Model Evaluation for KNN
print("K-NN Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

# Model Evaluation for Decision Tree
print("Decision Tree Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# Comparison chart values
models = ['Random Forest', 'K-NN', 'Decision Tree']
accuracies = [accuracy_score(y_test, y_pred_rf) * 100,
              accuracy_score(y_test, y_pred_knn) * 100,
              accuracy_score(y_test, y_pred_dt) * 100]

precision_scores = [
    classification_report(y_test, y_pred_rf, output_dict=True)['1']['precision'],
    classification_report(y_test, y_pred_knn, output_dict=True)['1']['precision'],
    classification_report(y_test, y_pred_dt, output_dict=True)['1']['precision']
]

recall_scores = [
    classification_report(y_test, y_pred_rf, output_dict=True)['1']['recall'],
    classification_report(y_test, y_pred_knn, output_dict=True)['1']['recall'],
    classification_report(y_test, y_pred_dt, output_dict=True)['1']['recall']
]

f1_scores = [
    classification_report(y_test, y_pred_rf, output_dict=True)['1']['f1-score'],
    classification_report(y_test, y_pred_knn, output_dict=True)['1']['f1-score'],
    classification_report(y_test, y_pred_dt, output_dict=True)['1']['f1-score']
]

# Create the comparison bar chart
bar_width = 0.2
index = range(len(models))

fig, ax = plt.subplots(figsize=(8, 5))
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

# Save the comparison image
plt.tight_layout()
plt.savefig("model_comparison_chart_small.png", dpi=300)
plt.show()

# Confusion matrix visualization
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['edible', 'poisonous'], yticklabels=['edible', 'poisonous'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("rf_confusion_matrix_small.png", dpi=300)
plt.show()
