import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('/Users/marianahernandez/Downloads/agaricus-lepiota.data', header=None)  # Use the correct path to your dataset

# The dataset doesn't have column names, so we will assign them
columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
           'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
           'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
           'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
df.columns = columns

# Preprocess the dataset
# Encoding categorical variables to numerical values
label_encoder = LabelEncoder()
for column in df.columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split the dataset into features (X) and target (y)
X = df.drop('class', axis=1)  # 'class' is the target variable
y = df['class']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Model evaluation for KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

# Initialize and train Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Model evaluation for Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

# Initialize and train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Model evaluation for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Comparison Bar Chart (including all models)
models = ['KNN', 'Decision Tree', 'Random Forest']
accuracies = [accuracy_knn, accuracy_dt, accuracy_rf]
precision_scores = [precision_knn, precision_dt, precision_rf]
recall_scores = [recall_knn, recall_dt, recall_rf]
f1_scores = [f1_knn, f1_dt, f1_rf]

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
plt.show()
