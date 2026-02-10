import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
import config

def visualize():
    embeddings_path = os.path.join(config.DATA_FOLDER, 'face_embeddings.pkl')
    clusters_path = os.path.join(config.DATA_FOLDER, 'face_clusters.pkl')

    print(f"Loading data from '{config.DATA_FOLDER}'...")

    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        with open(clusters_path, 'rb') as f:
            labels = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find data files in '{config.DATA_FOLDER}'.")
        print("Please run 'main.py' first to generate the data!")
        return

    print("Generating visualization...")

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(embeddings)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:
            color = 'k'
            marker = 'x'
            label_name = "Noise / Unknown"
            alpha = 0.3
        else:
            color = None 
            marker = 'o'
            label_name = f"Person {label}"
            alpha = 0.8

        mask = [l == label for l in labels]
        
        plt.scatter(
            reduced_data[mask, 0], 
            reduced_data[mask, 1], 
            c=color, 
            marker=marker, 
            label=label_name, 
            alpha=alpha
        )

    plt.title("AI Face Clusters Visualization")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize()