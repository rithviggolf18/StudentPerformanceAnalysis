import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv("student_data.csv")

# Encode categorical variables
df_encoded = pd.get_dummies(df)

# Drop 'G1' and 'G2'
df_filtered = df_encoded.drop(['G1', 'G2'], axis=1)

# Split the data into features and target variable
X = df_filtered.drop(['G3'], axis=1)
y = np.where(df_filtered['G3'] >= 12, 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Imbalanced Class Handling (SMOTE)
smote = SMOTE(random_state=18)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Hyperparameter tuning for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm_model = SVC()
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train_resampled, y_train_resampled)
best_svm_model = grid_search_svm.best_estimator_

# Fit the best SVM model
best_svm_model.fit(X_train_resampled, y_train_resampled)

# Predict and evaluate the SVM model
y_pred_svm = best_svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

# Split the data into features and target variable including 'G1' and 'G2'
X_with_G1_G2 = df_encoded.drop(['G3'], axis=1)
y_with_G1_G2 = np.where(df_encoded['G3'] >= 12, 1, 0)

# Split the data into training and testing sets including 'G1' and 'G2'
X_train_with_G1_G2, X_test_with_G1_G2, y_train_with_G1_G2, y_test_with_G1_G2 = train_test_split(X_with_G1_G2, y_with_G1_G2, test_size=0.3, random_state=18)

# Standardize the features including 'G1' and 'G2'
scaler_with_G1_G2 = StandardScaler()
X_train_scaled_with_G1_G2 = scaler_with_G1_G2.fit_transform(X_train_with_G1_G2)
X_test_scaled_with_G1_G2 = scaler_with_G1_G2.transform(X_test_with_G1_G2)

# Imbalanced Class Handling (SMOTE) including 'G1' and 'G2'
smote_with_G1_G2 = SMOTE(random_state=18)
X_train_resampled_with_G1_G2, y_train_resampled_with_G1_G2 = smote_with_G1_G2.fit_resample(X_train_scaled_with_G1_G2, y_train_with_G1_G2)

# Hyperparameter tuning for SVM
param_grid_svm_with_G1_G2 = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm_model_with_G1_G2 = SVC()
grid_search_svm_with_G1_G2 = GridSearchCV(estimator=svm_model_with_G1_G2, param_grid=param_grid_svm_with_G1_G2, cv=5, scoring='accuracy')
grid_search_svm_with_G1_G2.fit(X_train_resampled_with_G1_G2, y_train_resampled_with_G1_G2)
best_svm_model_with_G1_G2 = grid_search_svm_with_G1_G2.best_estimator_

# Fit the best SVM model including 'G1' and 'G2'
best_svm_model_with_G1_G2.fit(X_train_resampled_with_G1_G2, y_train_resampled_with_G1_G2)

# Predict and evaluate the SVM model including 'G1' and 'G2'
y_pred_svm_with_G1_G2 = best_svm_model_with_G1_G2.predict(X_test_scaled_with_G1_G2)
accuracy_svm_with_G1_G2 = accuracy_score(y_test_with_G1_G2, y_pred_svm_with_G1_G2)
precision_svm_with_G1_G2 = precision_score(y_test_with_G1_G2, y_pred_svm_with_G1_G2)
recall_svm_with_G1_G2 = recall_score(y_test_with_G1_G2, y_pred_svm_with_G1_G2)
f1_svm_with_G1_G2 = f1_score(y_test_with_G1_G2, y_pred_svm_with_G1_G2)

# Confusion matrix for the SVM model including 'G1' and 'G2'
cm_svm_with_G1_G2 = confusion_matrix(y_test_with_G1_G2, y_pred_svm_with_G1_G2)
plt.figure()
sns.heatmap(cm_svm_with_G1_G2, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVM Model with G1 and G2')
plt.show()
print("SVM Model Accuracy with G1 and G2:", accuracy_svm_with_G1_G2)
print("SVM Model Precision with G1 and G2:", precision_svm_with_G1_G2)
print("SVM Model Recall with G1 and G2:", recall_svm_with_G1_G2)
print("SVM Model F1 Score with G1 and G2:", f1_svm_with_G1_G2)


# Confusion matrix for the SVM model
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure()
sns.heatmap(cm_svm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVM Model without G1 and G2')
plt.show()
print("SVM Model Accuracy without G1 and G2:", accuracy_svm)
print("SVM Model Precision without G1 and G2:", precision_svm)
print("SVM Model Recall without G1 and G2:", recall_svm)
print("SVM Model F1 Score without G1 and G2:", f1_svm)