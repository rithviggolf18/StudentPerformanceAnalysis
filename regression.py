import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

df = pd.read_csv("student_data.csv")

# Drop 'G1' and 'G2' columns
df_filtered = df.drop(['G1', 'G2'], axis=1)

# Perform one-hot encoding on categorical variables
df_encoded = pd.get_dummies(df_filtered)

# Split the data into features and target variable
X = df_encoded.drop(['G3'], axis=1)  
y = df_encoded['G3']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVR with linear kernel
svr = SVR(kernel='linear')
svr.fit(X_train_scaled, y_train)

# Get feature importances (coefficients)
coefficients = svr.coef_[0]
feature_importance = pd.Series(coefficients, index=X_train.columns)
sorted_importance = feature_importance.abs().sort_values(ascending=False)

# Predict on the test set
y_pred = svr.predict(X_test_scaled)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", mae)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importance.head(15), y=sorted_importance.head(15).index, hue=sorted_importance.head(15).index, palette='coolwarm', legend=False)
plt.xlabel('Absolute Coefficient')
plt.ylabel('Feature')
plt.title('Feature Importance (Absolute Coefficients)')
plt.show()
