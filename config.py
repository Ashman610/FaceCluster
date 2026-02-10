INPUT_FOLDER = 'in_pic'     # Uncategorized input images go here. You can create subfolders for better organization, but all images will be processed together.
OUTPUT_FOLDER = 'out_pic'   # Organized output images with bounding boxes and cluster labels will be saved here. Subfolders will be created for each cluster.
DATA_FOLDER = 'data'        # Intermediate data (face embeddings and cluster labels) will be stored here. This allows you to run the visualization separately without reprocessing the images.

IMAGE_SIZE = 160 
# Lower = stricter (more small clusters). Higher = looser (fewer large clusters).
DBSCAN_EPS = 0.9 
DBSCAN_MIN_SAMPLES = 2 # Minimum number of faces to define a cluster.