# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "agaricus-lepiota.data"
df = pd.read_csv(file_path, header=None)

# Add column names
df.columns = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

# Encode categorical features
le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

# Print dataset info and missing values
print(df.head())
print(df.isnull().sum())

# Separate features and target variable
X = df.drop("class", axis=1)
y = df["class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and evaluate Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Model Evaluation:")
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy_rf * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Feature importances
feature_importances = pd.DataFrame(
    {"feature": X.columns, "importance": rf.feature_importances_}
).sort_values(by="importance", ascending=False)
print("\nFeature Importances:")
print(feature_importances)

# Initialize and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Evaluate KNN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

# Initialize and train Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Evaluate Decision Tree model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

# Comparison Bar Chart
models = ["KNN", "Decision Tree", "Random Forest"]
accuracies = [accuracy_knn, accuracy_dt, accuracy_rf]
precision_scores = [precision_knn, precision_dt, precision_rf]
recall_scores = [recall_knn, recall_dt, recall_rf]
f1_scores = [f1_knn, f1_dt, f1_rf]

bar_width = 0.2
index = range(len(models))

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(index, accuracies, bar_width, label="Accuracy")
bar2 = ax.bar(
    [i + bar_width for i in index], precision_scores, bar_width, label="Precision"
)
bar3 = ax.bar(
    [i + 2 * bar_width for i in index], recall_scores, bar_width, label="Recall"
)
bar4 = ax.bar(
    [i + 3 * bar_width for i in index], f1_scores, bar_width, label="F1 Score"
)

ax.set_xlabel("Models")
ax.set_ylabel("Scores")
ax.set_title("Comparison of KNN, Decision Tree, and Random Forest Models")
ax.set_xticks([i + 1.5 * bar_width for i in index])
ax.set_xticklabels(models)
ax.legend()

# Save the plot as an image
plt.savefig("plot.png")
print("\nPlot saved as 'plot.png'.")
plt.show()


