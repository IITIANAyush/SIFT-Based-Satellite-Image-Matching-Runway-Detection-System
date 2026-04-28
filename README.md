sift-satellite-matching/
│
├── app/
│   └── ui_v3.py                # Streamlit app
│
├── core/
│   ├── preprocessing.py        # CLAHE, resize, grayscale
│   ├── sift_pipeline.py        # keypoints + descriptors
│   ├── matcher.py              # BFMatcher + ratio + cross-check
│   ├── ransac.py               # homography + inliers
│
├── experiments/
│   └── robustness_analysis.ipynb
│
├── assets/
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── demo_output.png
│
├── results/
│   └── plots/
│
├── requirements.txt
├── README.md
└── LICENSE
