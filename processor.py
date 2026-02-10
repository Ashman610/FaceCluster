import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import sys
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm
import config

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def extract_embeddings(image_folder):
    print(f"Loading models...")
    embedder = FaceNet()
    detector = MTCNN()
    
    embeddings = []
    file_paths = []
    
    if not os.path.exists(image_folder):
        print(f"Error: Input folder '{image_folder}' not found.")
        return [], []
        
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images...")

    for image_name in tqdm(image_files, desc="Scanning Faces"):
        image_path = os.path.join(image_folder, image_name)
        try:
            stream = open(image_path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            stream.close()
        except Exception:
            image = cv2.imread(image_path)
        
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with SuppressOutput():
            faces = detector.detect_faces(image_rgb)

        for face in faces:
            x, y, width, height = face['box']
            x, y = max(0, x), max(0, y)
            
            face_image = image_rgb[y:y+height, x:x+width]

            if face_image.size > 0:
                try:
                    face_image_resized = cv2.resize(face_image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                    sample = np.expand_dims(face_image_resized, axis=0)
                    with SuppressOutput():
                        embedding = embedder.embeddings(sample)[0]
                    
                    embeddings.append(embedding)
                    file_paths.append(image_path)
                except Exception as e:
                    pass
                    
    return embeddings, file_paths