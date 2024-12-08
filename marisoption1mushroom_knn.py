import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('/Users/marianahernandez/Downloads/agaricus-lepiota.data', header=None)  

# Define column names
columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
           'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
           'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
           'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
df.columns = columns

# Replace missing values denoted by '?' with NaN
df.replace('?', np.nan, inplace=True)

# Exploratory Data Analysis (EDA)
print("Column Data Types:")
print(df.dtypes)

# Overall dataset distribution for all columns
for col in df.columns:
    sns.countplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()

# Split by class label and plot histograms
for col in df.columns[1:]:  # Skip 'class'
    sns.histplot(data=df, x=col, hue='class', multiple='dodge', shrink=0.8)
    plt.title(f'{col} by Class')
    plt.xticks(rotation=45)
    plt.show()

# Preprocess the dataset
# Use OneHotEncoder for categorical data
categorical_features = df.columns[1:]  # Exclude 'class'
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Impute missing values and preprocess in a pipeline
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', preprocessor)
])

X = df.drop('class', axis=1)
y = df['class']

# Apply the pipeline to transform features
X_transformed = pipeline.fit_transform(X)

# Encode the target variable ('class')
y = y.map({'e': 0, 'p': 1})  # Encode 'e' (edible) as 0, 'p' (poisonous) as 1

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42, stratify=y)

# Initialize KNN with Hamming distance
knn = KNeighborsClassifier(n_neighbors=5, metric='hamming')

# Cross-validation to evaluate metrics
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(knn, X_transformed, y, cv=cv, scoring='accuracy')
precision_scores = cross_val_score(knn, X_transformed, y, cv=cv, scoring='precision')
recall_scores = cross_val_score(knn, X_transformed, y, cv=cv, scoring='recall')
f1_scores = cross_val_score(knn, X_transformed, y, cv=cv, scoring='f1')

# Train the KNN model
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Model evaluation on test set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print cross-validation results
print("Cross-Validation Metrics:")
print(f"Accuracy: {accuracy_scores.mean():.4f} ± {accuracy_scores.std():.4f}")
print(f"Precision: {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
print(f"Recall: {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
print(f"F1 Score: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

# Print test set results
print("\nTest Set Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
