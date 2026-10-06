import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_experiment():
    df = pd.read_csv('digits.csv')
    X = df.drop('target', axis=1)
    
    inertias = []
    silhouettes = []
    ks = range(2, 16)
    
    for k in ks:
        km = KMeans(n_clusters=k, n_init='auto').fit(X)
        inertia = km.inertia_
        sil = silhouette_score(X, km.labels_)
        inertias.append(inertia)
        silhouettes.append(sil)
        print(f"K={k} Inertia: {inertia:.2f}, Silhouette: {sil:.2f}")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ks, inertias, 'bo-')
    plt.title('Elbow Method')
    plt.xlabel('K')
    plt.ylabel('Inertia')
    
    plt.subplot(1, 2, 2)
    plt.plot(ks, silhouettes, 'ro-')
    plt.title('Silhouette Scores')
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.savefig('e15.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()
