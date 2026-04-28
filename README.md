# 🛰️ SIFT-Based Satellite Image Matching & Runway Detection

### 🚀 Repository Description
Scale- and rotation-invariant satellite image matching system using SIFT, feature filtering, and RANSAC-based geometric verification, with an interactive Streamlit interface for real-time analysis.

---

## 📌 Problem Statement

Satellite images of the same location often vary due to:
- Rotation (different viewing angles)  
- Scale (zoom levels)  
- Image degradation (blur, lighting, noise)  

Traditional pixel-based methods fail under these conditions.  
This project solves the problem using **scale-invariant feature detection and robust geometric matching**.

---

## 🧠 Methodology

### Pipeline Overview

1. **Preprocessing**
   - Resize images (standard resolution normalization)
   - Convert to grayscale
   - Apply CLAHE for contrast enhancement  

2. **Feature Extraction**
   - SIFT keypoint detection  
   - 128-dimensional descriptor generation  

3. **Feature Matching**
   - Brute Force Matcher (L2 norm)  
   - Cross-check filtering (mutual matches)  
   - Lowe’s Ratio Test for ambiguity rejection  

4. **Geometric Verification**
   - Homography estimation  
   - RANSAC to remove outliers  
   - Extraction of inlier correspondences  

5. **Evaluation Metric**
   - **Inlier Ratio = RANSAC Inliers / Total Matches**  
   - Measures geometric consistency of matches  

---

## 🛠️ Tech Stack

- Python  
- OpenCV  
- NumPy  
- Streamlit  
- Matplotlib  

---

## 🖥️ Features

- Upload and compare two satellite images  
- Simulate transformations:
  - Rotation  
  - Scaling  
  - Gaussian Blur  
- Visualize:
  - Keypoints  
  - Feature matches  
  - RANSAC inliers  
- Display real-time metrics:
  - Number of keypoints  
  - Match count  
  - Inlier ratio  

---

## 📊 Results

- ~4000+ keypoints detected per image  
- ~1500+ reliable matches after filtering  
- Strong geometric consistency using RANSAC  
- Robust performance across:
  - Rotation: 0° → 90°  
  - Scale: 0.3 → 1.0  
  - Blur: Gaussian σ up to 5  

---

## 📈 Robustness Analysis

Evaluated system performance under controlled variations:

- **Rotation Analysis** → gradual decrease in inlier ratio  
- **Scale Analysis** → strong invariance across zoom levels  
- **Blur Analysis** → noticeable degradation in feature detection  

See `/experiments` for detailed analysis.

---

## 📂 Project Structure


sift-satellite-matching/
│
├── app/ # Streamlit UI
├── core/ # CV pipeline modules
├── experiments/ # Analysis notebooks
├── assets/ # Sample images
├── results/ # Output plots
├── requirements.txt
└── README.md


---

## ▶️ Installation & Usage

```bash
git clone https://github.com/yourusername/sift-satellite-matching.git
cd sift-satellite-matching

pip install -r requirements.txt
streamlit run app/ui_v3.py
```

Keypoint visualization
Match correspondence
UI dashboard
###🔍 Key Insights
Geometric verification (RANSAC) is critical for reliable matching
Cross-check + ratio test significantly reduces false matches
SIFT is robust to scale and rotation but sensitive to blur
Preprocessing improves feature detection in low-contrast regions
###🚀 Future Work
Compare with ORB, SuperPoint, SuperGlue
Integrate deep learning-based feature matching
Extend to UAV navigation and landing systems
Deploy as a web API for geospatial matching
📜 License

MIT License

###👤 Author

Ayush Bhaskar
Computer Vision | Robotics | Aeromodelling
