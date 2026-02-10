# 📸 FaceCluster: Automated AI Photo Organizer
### Intelligent Unsupervised Face Clustering & Sorting

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://opensource.org/licenses/MIT)

## 🚀 Overview
**FaceCluster** is an automated pipeline that organizes massive, chaotic photo collections into structured folders based on human identity.

Unlike traditional face recognition systems that require you to "train" the model on specific people beforehand, this tool uses **Unsupervised Learning (DBSCAN)**. It blindly analyzes a batch of photos, detects unique faces, and groups them together without ever knowing who the people are.

## 🛠️ The Tech Stack
This project integrates three state-of-the-art technologies into a cohesive pipeline:

1.  **Face Detection (MTCNN):** Uses *Multi-task Cascaded Convolutional Networks* to locate faces in images, handling various angles and lighting conditions.
2.  **Feature Extraction (FaceNet):** Converts cropped faces into **128-dimensional embeddings** (numerical vectors). These vectors represent the unique features of a face.
3.  **Clustering (DBSCAN):** *Density-Based Spatial Clustering of Applications with Noise*. It groups the vectors based on Euclidean distance.
    * **Advantage:** Unlike K-Means, DBSCAN does not require knowing the number of people ($k$) in advance.
    * **Noise Handling:** Automatically rejects blurry or non-human "faces" as noise.

---

## 💡 Use Cases (Beyond Photography)
This concept is applicable to many domains:

### 1. 🛡️ Security & Surveillance
* **Repeat Visitor Tracking:** Analyze hours of CCTV frames to identify distinct individuals who visited a location, even if their identity is unknown.
* **Threat Detection:** Isolate "unknown" faces from a database of authorized personnel.

### 2. 🎞️ Media & Journalism
* **Archive Management:** Rapidly sort thousands of press photos by politician, celebrity, or athlete without manual tagging.
* **Video Indexing:** Extract frames from long interviews or movies and group them by the actor on screen.

### 3. 🏠 Personal Digital Archiving
* **Family Albums:** Organize decades of mixed family photos (digital or scanned) into folders for each family member.

### 4. 🎓 Education & Yearbooks
* **Student Sorting:** Automatically group thousands of unlabelled student candids for yearbook layouts.

---

## ⚙️ Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Muhammad-Hassan12/FaceCluster.git](https://github.com/Muhammad-Hassan12/FaceCluster.git)
    cd FaceCluster
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment (Conda/venv).
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Data**
    * Create a folder named `in_pic`.
    * Drop your raw, unsorted images (`.jpg`, `.png`) into it.

---

## 🏃‍♂️ Usage

### 1. Run the Organizer
This script detects faces, calculates embeddings, and physically sorts the files.
```bash
python main.py
```
* Output: Organized folders will appear in out_pic/.
* Data: Embeddings and cluster labels are saved in data/.

### 2. Visualize the Clusters
Generate a 2D scatter plot to see how the AI separated the identities.

* Output: A T-SNE/PCA plot showing distinct clusters for each person.

---

## 📂 Project Structure
```bash
FaceCluster/
│
├── config.py           # Configuration (Paths, DBSCAN epsilon, Image size)
├── processor.py        # Core Engine (MTCNN detection & FaceNet embeddings)
├── main.py             # Main Pipeline (Orchestrates clustering & sorting)
├── visualize.py        # Analytics (Generates 2D cluster plots)
│
├── in_pic/             # [Input] Place raw images here
├── out_pic/            # [Output] Organized folders appear here
└── data/               # [System] Stores serialized embeddings (.pkl)
```
#### *Reminder!*
Dont forget to create "in_pic", "out_pic", and "data" folders!!!

---

### 🔧 Configuration (config.py)
You can tweak the clustering sensitivity in config.py:
* DBSCAN_EPS: Controls how strict the matching is. Lower = fewer matches, higher precision. Higher = more matches, potential mix-ups.
* DBSCAN_MIN_SAMPLES: The minimum number of photos required to create a "Person" folder.

---

## 👨‍💻 Authors & Acknowledgments

### **Lead Architect:** "Syed Muhammad Hassan" (https://github.com/Muhammad-Hassan12)
*Initial concept, algorithm implementation, and core logic.*

### **Maintained by:** "AgenticEra Systems"

---

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License
Distributed under the "MIT License". See "LICENSE" for more information.

---

*Built with ❤️ in Python. If you find this tool useful, please star the repo!*
