import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


# Dataset
df = pd.read_csv('/Users/marianahernandez/Downloads/agaricus-lepiota.data', header=None)


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
imputer = SimpleImputer(strategy='most_frequent')
df.iloc[:, :] = imputer.fit_transform(df)




# Display first 5 rows and check for missing values
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())




# Histograms separated by class labels (only for K-NN)
for column in df.columns:
    if column != 'class':
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x=column, hue='class', multiple='dodge', shrink=0.8)
        plt.title(f'Histogram of {column} (Separated by Class)')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()




# Histograms of overall dataset (only for K-NN)
for column in df.columns:
    if column != 'class':
        plt.figure(figsize=(8, 6))
        df[column].value_counts().plot(kind='bar', color='blue', alpha=0.7)
        plt.title(f'Histogram of {column} (Overall)')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()


# Separate features and target variable
X = df.drop('class', axis=1)
y = df['class']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Normalize features (scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Train K-NN with Hamming distance
knn_hamming = KNeighborsClassifier(n_neighbors=5, metric='hamming')
knn_hamming.fit(X_train, y_train)


# Generate predictions for K-NN with Hamming distance
knn_hamming_y_pred = knn_hamming.predict(X_test)

# Model Evaluation Metrics for K-NN with Hamming distance
print("\nK-NN Model Evaluation with Hamming Distance:")
print(f"Accuracy: {accuracy_score(y_test, knn_hamming_y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, knn_hamming_y_pred, average='macro'):.4f}")
print(f"Recall: {recall_score(y_test, knn_hamming_y_pred, average='macro'):.4f}")
print(f"F1 Score: {f1_score(y_test, knn_hamming_y_pred, average='macro'):.4f}")

# Confusion Matrix Visualization for K-NN with Hamming distance
knn_hamming_conf_matrix = confusion_matrix(y_test, knn_hamming_y_pred, labels=knn_hamming.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(knn_hamming_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['edible', 'poisonous'], yticklabels=['edible', 'poisonous'])
plt.title("K-NN Confusion Matrix with Hamming Distance")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# Cross-validation scores for K-NN with Hamming distance
cv_scores_knn_hamming = cross_val_score(knn_hamming, X_scaled, y, cv=5, scoring='accuracy')


print(f"\nK-NN Model with Hamming Distance Cross-Validation Accuracy: {cv_scores_knn_hamming.mean():.4f} ± {cv_scores_knn_hamming.std():.4f}")


# Classification Report for K-NN with Hamming distance
print("\nK-NN Classification Report with Hamming Distance:")
print(classification_report(y_test, knn_hamming_y_pred))


# Add Hamming distance results to the Model Comparison Bar Plot
models = ['K-NN (Hamming)']
accuracies = [accuracy_score(y_test, knn_hamming_y_pred)]
precision_scores = [precision_score(y_test, knn_hamming_y_pred, average='macro')]
recall_scores = [recall_score(y_test, knn_hamming_y_pred, average='macro')]
f1_scores = [f1_score(y_test, knn_hamming_y_pred, average='macro')]


bar_width = 0.2
index = np.arange(len(models))


fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(index, accuracies, bar_width, label='Accuracy')
bar2 = ax.bar(index + bar_width, precision_scores, bar_width, label='Precision')
bar3 = ax.bar(index + 2 * bar_width, recall_scores, bar_width, label='Recall')
bar4 = ax.bar(index + 3 * bar_width, f1_scores, bar_width, label='F1 Score')


ax.set_xlabel('Models')
ax.set_title('Model Evaluation (K-NN with Hamming Distance)')
ax.set_xticks(index + bar_width * 1.5)
ax.set_xticklabels(models)
ax.legend()


plt.tight_layout()
plt.show()


# Correlation Matrix Visualization
plt.figure(figsize=(10, 8))
corr_matrix = np.corrcoef(X_scaled.T)
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Matrix')
plt.show()


# Overall target distribution
plt.figure(figsize=(8, 6))
y.value_counts().plot(kind='bar', color='purple', alpha=0.7)
plt.title("Target Variable Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

