import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'agaricus-lepiota.data'
df = pd.read_csv(file_path, header=None)

columns = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]
data = pd.read_csv("/Users/venus/Downloads/HomanDemo/mushroom/agaricus-lepiota.data", header=None, names=columns)

# Replace "?" with NaN and fill missing values
data.replace("?", np.nan, inplace=True)
data["stalk-root"] = data["stalk-root"].fillna(data["stalk-root"].mode()[0])

# Separate features and target
X = data.drop("class", axis=1)
y = data["class"]

# Encode categorical variables using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Train a Decision Tree Classifier with max_depth = 7
dt_model = DecisionTreeClassifier(
    max_depth=7,  # Increased max depth from 5 to 7
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)
dt_model.fit(X_train, y_train)

# Evaluate the model
y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(dt_model, X_encoded, y, cv=cv)
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred, labels=dt_model.classes_)
cm_df = pd.DataFrame(cm, index=dt_model.classes_, columns=dt_model.classes_)

# Display the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Display other metrics
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")
print(f"Stratified Cross-Validation Accuracy: {cv_mean:.2f} ± {cv_std:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred))

# Feature Importance
feature_names = encoder.get_feature_names_out(X.columns)
feature_importances = pd.Series(dt_model.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop Features Driving the Model:\n", feature_importances.head(10))

# Number of duplicate rows
duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Decision tree visualization
plt.figure(figsize=(15, 10))
plot_tree(dt_model, feature_names=feature_names, class_names=dt_model.classes_, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Add optional noisy data test
X_test_noisy = X_test.copy()
np.random.seed(42)
random_indices = np.random.choice(X_test.shape[0], size=int(0.1 * X_test.shape[0]), replace=False)
random_features = np.random.choice(X_test.shape[1], size=int(0.1 * X_test.shape[1]), replace=False)
for i in random_indices:
    for j in random_features:
        X_test_noisy[i, j] = np.random.choice([0, 1])  # Introduce noise
noisy_test_accuracy = accuracy_score(y_test, dt_model.predict(X_test_noisy))
print(f"Accuracy with Noisy Test Data: {noisy_test_accuracy:.2f}")
