import os
import shutil
import pickle
import numpy as np
from sklearn.cluster import DBSCAN
import config
from processor import extract_embeddings

def main():
    embeddings, file_paths = extract_embeddings(config.INPUT_FOLDER)

    if not embeddings:
        print("No faces found to cluster.")
        return

    print(f"Clustering {len(embeddings)} faces...")
    clt = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES, metric='euclidean')
    labels = clt.fit_predict(embeddings)

    print("Sorting photos into folders...")
    
    if not os.path.exists(config.OUTPUT_FOLDER):
        os.makedirs(config.OUTPUT_FOLDER)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} unique people.")

    for label, src_path in zip(labels, file_paths):
        if label == -1:
            dir_name = "Uncategorized"
        else:
            dir_name = f"Person_{label}"       
        target_dir = os.path.join(config.OUTPUT_FOLDER, dir_name)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)        
        filename = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, filename)
        
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

    if not os.path.exists(config.DATA_FOLDER):
        os.makedirs(config.DATA_FOLDER)   
    clusters_path = os.path.join(config.DATA_FOLDER, 'face_clusters.pkl')
    embeddings_path = os.path.join(config.DATA_FOLDER, 'face_embeddings.pkl')

    with open(clusters_path, 'wb') as f:
        pickle.dump(labels, f)
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Done! Photos organized in '{config.OUTPUT_FOLDER}'")

if __name__ == "__main__":
    main()