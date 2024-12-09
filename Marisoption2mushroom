import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Dataset loading
df = pd.read_csv('/Users/marianahernandez/Downloads/agaricus-lepiota.data', header=None)

# initial data summary
print("Dataset Overview:")
print(df.info())

# 1. if theres missing values handle them
# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Check for missing data
missing_percentage = df.isnull().mean() * 100
print("Missing Data Percentage per Feature:")
print(missing_percentage)

# SimpleImputer to handle data thats missing
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df))

# 2. One-hot encode categorical variables
X = df_imputed.drop(columns=[22])  # Assuming the target column is column 22
y = df_imputed[22]  # Target variable

# make sure column names are strings
X.columns = X.columns.astype(str)

# Apply OneHotEncoder to categorical features
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Convert numpy array  to a DataFrame with proper column names
encoded_columns = encoder.get_feature_names_out(X.columns)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)

# 3. Normalize features (scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded_df)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. K-Nearest Neighbors (KNN) Model with Hamming distance
knn = KNeighborsClassifier(n_neighbors=5, metric='hamming')
knn.fit(X_train, y_train)

# 6. Predictions and evaluation
y_pred = knn.predict(X_test)
print("KNN Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy:.4f}")

# 7. Histograms for overall dataset and by class label
# Histogram for the overall dataset (all features)
for column in df_imputed.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df_imputed[column], kde=False, bins=20)
    plt.title(f'Histogram of {column}')
    plt.show()

# Histograms separated by the class label target
sns.histplot(df_imputed, x=0, hue=22, kde=False, bins=20)  # Adjust according to actual target column
plt.title('Histogram of Class Labels')
plt.show()

# 8. Decision Tree and Random Forest Models (for visual inspection)
# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=X_encoded_df.columns)
plt.title("Decision Tree Visualization")
plt.show()

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Visualize one of trees from the Random Forest
plt.figure(figsize=(20,10))
plot_tree(rf.estimators_[0], filled=True, feature_names=X_encoded_df.columns)
plt.title("Random Forest Visualization (First Tree)")
plt.show()

# 9. Cross-validation to evaluate model performance
cv_scores_knn = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
cv_scores_dt = cross_val_score(dt, X_scaled, y, cv=5, scoring='accuracy')
cv_scores_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')

print(f"KNN Model Cross-Validation Accuracy: {cv_scores_knn.mean():.4f} ± {cv_scores_knn.std():.4f}")
print(f"Decision Tree Model Cross-Validation Accuracy: {cv_scores_dt.mean():.4f} ± {cv_scores_dt.std():.4f}")
print(f"Random Forest Model Cross-Validation Accuracy: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

# Additionally, report F1-score, precision, recall for each model
print("\nKNN Cross-Validation Classification Report:")
print(cross_val_score(knn, X_scaled, y, cv=5, scoring='f1_macro'))

print("\nDecision Tree Cross-Validation Classification Report:")
print(cross_val_score(dt, X_scaled, y, cv=5, scoring='f1_macro'))

print("\nRandom Forest Cross-Validation Classification Report:")
print(cross_val_score(rf, X_scaled, y, cv=5, scoring='f1_macro'))

# Reporting Precision and Recall
print("\nKNN Precision and Recall:")
print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")

print("\nDecision Tree Precision and Recall:")
y_pred_dt = dt.predict(X_test)
print(f"Precision: {precision_score(y_test, y_pred_dt, average='macro'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt, average='macro'):.4f}")

print("\nRandom Forest Precision and Recall:")
y_pred_rf = rf.predict(X_test)
print(f"Precision: {precision_score(y_test, y_pred_rf, average='macro'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf, average='macro'):.4f}")
