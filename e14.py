import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def run_experiment():
    df = pd.read_csv('mall_customers.csv')
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    X_scaled = StandardScaler().fit_transform(X)

    # K-means
    km = KMeans(n_clusters=5, n_init='auto').fit(X_scaled)
    # Hierarchical
    hc = AgglomerativeClustering(n_clusters=5).fit(X_scaled)

    print(f"K-means Silhouette: {silhouette_score(X_scaled, km.labels_):.2f}")
    print(f"Hierarchical Silhouette: {silhouette_score(X_scaled, hc.labels_):.2f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=km.labels_, cmap='viridis')
    plt.title('K-means Clustering')
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')

    plt.subplot(1, 2, 2)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=hc.labels_, cmap='viridis')
    plt.title('Hierarchical Clustering')
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    
    plt.tight_layout()
    plt.savefig('e14.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()
