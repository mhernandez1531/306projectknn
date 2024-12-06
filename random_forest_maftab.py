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

# Display first 5 rows and check for missing values
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

# Separate features and target variable
X = df.drop('class', axis=1)
y = df['class']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Create histograms
def plot_histograms(data, class_col, title):
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))
    axes = axes.flatten()
    for i, column in enumerate(data.columns):
        sns.histplot(data=data, x=column, hue=class_col, kde=False, ax=axes[i], palette="Set2")
        axes[i].set_title(column)
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# Overall histograms
plot_histograms(df, None, "Overall Feature Distribution")

# Histograms separated by class labels
plot_histograms(df, df['class'], "Feature Distribution by Class")

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_encoded, y)

# Visualize one tree from the Random Forest
tree_index = 0  # Index of the tree to visualize
tree = rf.estimators_[tree_index]

dot_data = export_graphviz(
    tree,
    out_file=None,
    feature_names=encoder.get_feature_names_out(X.columns),
    class_names=['edible', 'poisonous'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("random_forest_tree", format="png", cleanup=True)

# Compute Hamming Distance Matrix
hamming_distances = pairwise_distances(X_encoded, metric='hamming')

# K-NN Model with Hamming Distance
knn = NearestNeighbors(n_neighbors=5, metric='precomputed')
knn.fit(hamming_distances)

# Cross-Validation for Random Forest
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
}

cv_results = cross_validate(rf, X_encoded, y, cv=5, scoring=scoring)
print("Cross-Validation Results:")
for metric, scores in cv_results.items():
    if metric.startswith("test_"):
        print(f"{metric}: Mean={np.mean(scores):.2f}, Std Dev={np.std(scores):.2f}")

# Visualize Feature Importances
importances = rf.feature_importances_
features = encoder.get_feature_names_out(X.columns)

plt.figure(figsize=(10, 8))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()

# Generate Predictions and Confusion Matrix
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
y_pred = rf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred, labels=rf.classes_)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['edible', 'poisonous'], yticklabels=['edible', 'poisonous'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Print model evaluation
print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Print the first 5 rows and missing values summary
print("\nFirst 5 Rows:")
print(df.head())
print("\nMissing Values Summary:")
print(df.isnull().sum())
