# --> BEGINNING OF: importing libraries
import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
# <-- END OF: importing libraries

# --> BEGINNING OF: function to plot scatter plot of income vs spending score
def scatter_plot(df):
    plt.figure(figsize=(7,5))
    
    # specifying data to plot
    plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], alpha=0.7)
    
    # x- and y- labels
    plt.xlabel("Annual Income (1000$)")
    plt.ylabel("Spending Score")
    
    # title
    plt.title("Income vs Spending Score")
    
    plt.grid(True)
    plt.show()
# <-- END OF: function to plot scatter plot of income vs spending score

# --> BEGINNING OF: function to run k-means w/ different k values and compare elbow method vs silhouette scores
def elbow_silhouette(scaled_df, k_range=range(2,11)):
    # initialise list to store elbow method metric for all values of k
    inertias = []
    
    # initialise list to store silhouette score for each k
    silhouettes = []
    
    for k in k_range:
        # create k means model
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_df)
        
        # append list of elbow method metrics
        inertias.append(km.inertia_)
        
        # compute silhouette score
        try:
            sil = silhouette_score(scaled_df, labels)
            
        # exception handling in case of failure
        except Exception:
            sil = np.nan
            
        # append list of silhouette scores
        silhouettes.append(sil)
        
    # create two plots side by side
    fig, ax = plt.subplots(1,2, figsize=(14,5))
    
    # --> BEGINNING OF: elbow plot
    ax[0].plot(k_range, inertias, '-o')
    ax[0].set_title("Elbow Method")
    ax[0].set_xlabel("Number of clusters (k)")
    ax[0].set_ylabel("Inertia")
    ax[0].grid(True)
    # <-- END OF: elbow plot
    
    # --> BEGINNING OF: silhouette plot
    ax[1].plot(k_range, silhouettes, '-o', color='orange')
    ax[1].set_title("Silhouette Score vs k")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Silhouette Score")
    ax[1].grid(True)
    # <-- END OF: silhouette plot
    
    plt.tight_layout()
    plt.show()
# <-- END OF: function to run k-means w/ different k values and compare elbow method vs silhouette scores

# --> BEGINNING OF: function to plot distinct clusters
def plot_clusters(df, labels, centers, title):
    plt.figure(figsize=(8,6))
    plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                c=labels, cmap='tab10', s=50, edgecolor='k')
    
    # mark centers of clusters if relevant
    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1], s=200, c='black', marker='X', label='Centers')
    
    # x- and y- labels
    plt.xlabel("Annual Income (1000$)")
    plt.ylabel("Spending Score (1-100)")
    
    # title
    plt.title(title)
    
    plt.legend()
    plt.grid(True)
    plt.show()
# <-- END OF: function to plot distinct clusters

# --> BEGINNING OF: main function
def main():
    # download Kaggle dataset
    path = kagglehub.dataset_download("shwetabh123/mall-customers")

    # find dataset
    csv_path = os.path.join(path, "Mall_Customers.csv")
    
    # error handling
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found in {path}")

    # read data
    df = pd.read_csv(csv_path)

    # create scatter plot of income vs spending score
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].copy()
    scatter_plot(X)

    # scale dataframe
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # create plots using elbow method & silhouette scores to find the best k
    elbow_silhouette(X_scaled)

    # fit KMeans with best silhouette heuristic
    best_k, best_sil = None, -1
    for k in range(2,11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        try:
            sil = silhouette_score(X_scaled, labels)
        except Exception:
            sil = -1
        if sil > best_sil:
            best_k, best_sil = k, sil
    print("----------------------")
    print(f"Best k by silhouette: {best_k} at score = {best_sil:.3f})")
    print("----------------------")

    # ready parameters for plotting clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    centers_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)

    # plot clusters
    plot_clusters(X, labels, centers=centers_unscaled, title=f"KMeans Clusters (k={best_k})")

    # print average spending per cluster
    df['Cluster'] = labels
    print("----------------------")
    print("- Average Spending per Cluster:")
    print(df.groupby('Cluster')['Spending Score (1-100)'].mean().sort_values(ascending=False))
    print("----------------------")

    # run DBSCAN
    db = DBSCAN(eps=0.5, min_samples=5)
    db_labels = db.fit_predict(X_scaled)
    print("----------------------")
    print("DBSCAN cluster counts:", pd.Series(db_labels).value_counts())
    print("----------------------")
    plot_clusters(X, db_labels, centers=None, title="DBSCAN Clusters")
# <-- END OF: main function

# run program
if __name__ == "__main__":
    main()
