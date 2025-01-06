mport pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv("data.csv")  # Replace with the actual file name

# Separate features (gene expressions) from labels
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Labels (tumor types)

# 1. Exploratory Data Analysis (EDA)

# Basic statistics
print(X.describe())

# Histograms of gene expression distributions
for i in range(0, X.shape[1], 1000):  # Sample some genes
    plt.hist(X.iloc[:, i], bins=50)
    plt.title(f"Gene {X.columns[i]} Expression Distribution")
    plt.show()

# Correlation heatmap (sample a subset of genes for visualization)
corr_matrix = X.iloc[:, :1000].corr()  # Sample for performance
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Correlation Heatmap (Subset of Genes)")
plt.show()

# 2. Dimensionality Reduction

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=50)  # Choose number of components
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Plot explained variance
plt.plot(range(1, 51), pca.explained_variance_ratio_.cumsum())
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.show()

# t-SNE (for visualization in 2D)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Plot t-SNE
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.astype('category').cat.codes)
plt.title("t-SNE Visualization of Gene Expression Data")
plt.show()

# 3. Clustering (K-Means)

# Determine optimal number of clusters (using the elbow method)
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Apply K-Means with the chosen k
kmeans = KMeans(n_clusters=5, random_state=42)  # Example: 5 clusters
clusters = kmeans.fit_predict(X_pca)

# Visualize clusters with t-SNE
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters)
plt.title("K-Means Clustering of Gene Expression Data")
plt.show()

# 4. Further Analysis

# Analyze cluster characteristics
for i in range(5):  # For each cluster
    cluster_data = X[clusters == i]
    print(f"Cluster {i}:")
    print(cluster_data.describe())

    # Check the distribution of tumor types within each cluster
    print(y[clusters == i].value_counts())

# ... (Further analysis, e.g., feature selection, classification, etc.)
