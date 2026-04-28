📄 README.md
🛰️ SIFT-Based Satellite Image Matching & Runway Detection

A computer vision system designed to reliably detect and match airport runways in satellite imagery, robust to variations in scale, orientation, and image quality.

This project combines feature-based matching with geometric verification to ensure high-confidence correspondences between images.

📌 Problem Statement

Satellite images of the same location often differ due to:

Rotation (different viewing angles)
Scale (zoom levels)
Image degradation (blur, lighting, noise)

Traditional pixel-based methods fail under these conditions.
This project addresses the problem using scale-invariant feature detection and robust matching.

🧠 Methodology
Pipeline Overview
Preprocessing
Image resizing (resolution normalization)
Grayscale conversion
CLAHE enhancement for low-contrast regions
Feature Extraction
SIFT keypoint detection
128-dimensional descriptor generation
Feature Matching
Brute Force Matcher (L2 norm)
Cross-check filtering (mutual matches)
Lowe’s Ratio Test (ambiguity rejection)
Geometric Verification
Homography estimation
RANSAC to remove outliers
Inlier-based validation
Evaluation Metric
Inlier Ratio = RANSAC Inliers / Total Matches
Measures geometric consistency of matches
🛠️ Tech Stack
Python
OpenCV
NumPy
Streamlit
Matplotlib
🖥️ Interactive Application

The project includes a Streamlit-based UI with:

Dual image upload for matching
Transformation simulation:
Rotation
Scaling
Gaussian blur
Visualization of:
Keypoints
Feature correspondences
RANSAC inliers
Real-time metrics:
Keypoints detected
Match count
Inlier ratio
📊 Results
~4000+ keypoints detected per image
~1500+ robust matches after filtering
High geometric consistency validated using RANSAC
Stable performance across:
Rotation: 0° → 90°
Scale: 0.3 → 1.0
Blur: Gaussian σ up to 5
📈 Robustness Analysis

Systematically evaluated performance under controlled transformations:

Rotation Analysis → gradual degradation in inlier ratio
Scale Analysis → strong invariance across zoom levels
Blur Analysis → significant impact on feature detectability

All experiments available in /experiments.

📂 Project Structure
sift-satellite-matching/
│
├── app/                # Streamlit UI
├── core/               # Modular CV pipeline
├── experiments/        # Notebooks for analysis
├── assets/             # Sample images
├── results/            # Output plots
├── requirements.txt
└── README.md
▶️ Installation & Usage
git clone https://github.com/yourusername/sift-satellite-matching.git
cd sift-satellite-matching

pip install -r requirements.txt
streamlit run app/ui_v3.py
📸 Sample Outputs

(Add screenshots here — keypoints, matches, UI dashboard)

🔍 Key Insights
Raw feature matching is insufficient without geometric verification
Cross-check + RANSAC significantly improves match reliability
SIFT is robust to scale and rotation, but sensitive to blur
Preprocessing (CLAHE + normalization) improves feature stability
🚀 Future Work
Replace SIFT with ORB / SuperPoint / SuperGlue for performance comparison
Integrate deep learning-based feature matching
Extend to real-time UAV navigation / landing alignment systems
Deploy as a web-based API for geospatial matching
📜 License

MIT License

👤 Author

Ayush Bhaskar
Aeromodelling | Computer Vision | Robotics
