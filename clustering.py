import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("student_data.csv")

# First clustering analysis with selected features
# Select relevant features for the first clustering analysis
selected_features_1 = ['age', 'Medu', 'Fedu', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
X_1 = df[selected_features_1]

# Standardize the data for the first clustering analysis
scaler_1 = StandardScaler()
X_scaled_1 = scaler_1.fit_transform(X_1)

# Choose the number of clusters for the first analysis (k)
k_1 = 3 

# Apply K-means clustering for the first analysis
kmeans_1 = KMeans(n_clusters=k_1, random_state=34, n_init=20, max_iter=300, tol=0.0001)
kmeans_1.fit(X_scaled_1)

# Add cluster labels to the dataset for the first analysis
df['cluster_1'] = kmeans_1.labels_

# Analyze the clusters for the first analysis
cluster_centers_1 = scaler_1.inverse_transform(kmeans_1.cluster_centers_)  # Convert centroids back to original scale
cluster_data_1 = pd.DataFrame(cluster_centers_1, columns=selected_features_1)
print("Cluster Data for First Group:")
print(cluster_data_1)

# Second clustering analysis with different selected features
# Select relevant features for the second clustering analysis
selected_features_2 = ['Dalc', 'Walc', 'famrel', 'goout', 'health', 'traveltime', 'G1', 'G2', 'G3']
X_2 = df[selected_features_2]

# Standardize the data for the second clustering analysis
scaler_2 = StandardScaler()
X_scaled_2 = scaler_2.fit_transform(X_2)

# Choose the number of clusters for the second analysis (k)
k_2 = 3 

# Apply K-means clustering for the second analysis
kmeans_2 = KMeans(n_clusters=k_2, random_state=34, n_init=20, max_iter=300, tol=0.0001)
kmeans_2.fit(X_scaled_2)

# Add cluster labels to the dataset for the second analysis
df['cluster_2'] = kmeans_2.labels_

# Analyze the clusters for the second analysis
cluster_centers_2 = scaler_2.inverse_transform(kmeans_2.cluster_centers_)  # Convert centroids back to original scale
cluster_data_2 = pd.DataFrame(cluster_centers_2, columns=selected_features_2)
print("\nCluster Data for Second Group:")
print(cluster_data_2)

