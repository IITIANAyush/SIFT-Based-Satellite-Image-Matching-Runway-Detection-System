import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(page_title="Satellite Image Matcher Pro", layout="wide")

st.title("🛰️ SIFT Feature Detection & Geometric Matching")
st.write("Using Cross-Check & RANSAC to verify geographic similarity.")

# --- SIDEBAR & MODES ---
mode = st.sidebar.radio("Analysis Mode", ["Upload Two Images", "Simulate Transformation"])
st.sidebar.markdown("---")

img1_raw, img2_raw = None, None

# ==========================================
# 1. IMAGE ACQUISITION
# ==========================================
if mode == "Upload Two Images":
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        img_file1 = st.file_uploader("Upload Target Image", type=["png","jpg","jpeg"])
    with col_u2:
        img_file2 = st.file_uploader("Upload Reference Image", type=["png","jpg","jpeg"])

    if img_file1 and img_file2:
        img1_raw = cv2.imdecode(np.frombuffer(img_file1.read(), np.uint8), 1)
        img2_raw = cv2.imdecode(np.frombuffer(img_file2.read(), np.uint8), 1)

else:
    img_file = st.file_uploader("Upload Base Satellite Image", type=["png","jpg","jpeg"])
    st.sidebar.subheader("Simulation Parameters")
    angle = st.sidebar.slider("Rotation (deg)", 0, 90, 30)
    scale = st.sidebar.slider("Scaling Factor", 0.3, 1.0, 0.7)
    blur = st.sidebar.slider("Gaussian Blur (Sigma)", 0, 5, 0)
    
    # NEW: Gaussian Noise Slider
    noise_sigma = st.sidebar.slider("Gaussian Noise (Intensity)", 0, 100, 0)

    if img_file:
        img1_raw = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 1)
        h, w = img1_raw.shape[:2]
        
        # Apply Geometric Transformation
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        img2_raw = cv2.warpAffine(img1_raw, M, (w, h))
        
        # Apply Blur if selected
        if blur > 0:
            img2_raw = cv2.GaussianBlur(img2_raw, (0,0), blur)
            
        # ==========================================
        # NEW: Apply Gaussian Noise
        # ==========================================
        if noise_sigma > 0:
            # Generate random noise from a normal distribution
            # we convert to float32 to prevent wrap-around errors (0-255)
            noise = np.random.normal(0, noise_sigma, img2_raw.shape).astype(np.float32)
            img2_noisy = img2_raw.astype(np.float32) + noise
            
            # Clip values to stay within [0, 255] and convert back to uint8
            img2_raw = np.clip(img2_noisy, 0, 255).astype(np.uint8)

# ==========================================
# 2. ENHANCED PROCESSING (Notebook Methodology)
# ==========================================
if img1_raw is not None and img2_raw is not None:
    # ACCURACY FIX: Always resize to a standard resolution (800x600).
    img1 = cv2.resize(img1_raw, (800, 600))
    img2 = cv2.resize(img2_raw, (800, 600))

    if st.button("🚀 Run SIFT Feature Matching"):
        with st.spinner("Analyzing geometric consistency..."):
            
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Use CLAHE for better feature extraction in satellite shadows
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray1 = clahe.apply(gray1)
            gray2 = clahe.apply(gray2)

            # Detect SIFT features
            sift = cv2.SIFT_create(nfeatures=0)
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)

            # ==========================================
            # ROBUST MATCHING
            # ==========================================
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches_knn_1 = bf.knnMatch(des1, des2, k=2)
            matches_knn_2 = bf.knnMatch(des2, des1, k=2)

            def ratio_test(matches, ratio=0.75):
                good = []
                for m, n in matches:
                    if m.distance < ratio * n.distance:
                        good.append(m)
                return good

            good_1 = ratio_test(matches_knn_1)
            good_2 = ratio_test(matches_knn_2)

            # Symmetric Matching
            good_matches = []
            matches_2_dict = {(m.queryIdx, m.trainIdx) for m in good_2}
            for m in good_1:
                if (m.trainIdx, m.queryIdx) in matches_2_dict:
                    good_matches.append(m)

            matches = sorted(good_matches, key=lambda x: x.distance)
            
            # RANSAC Filtering
            inliers = 0
            matches_mask = None
            if len(matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(
                    src_pts, dst_pts,
                    cv2.RANSAC,
                    4.0,
                    maxIters=5000,
                    confidence=0.995
                )                
                if mask is not None:
                    matches_mask = mask.ravel().tolist()
                    inliers = np.sum(matches_mask)

            # ==========================================
            # 3. RESULTS DISPLAY
            # ==========================================
            if len(matches) > 0:
                match_quality = (inliers / len(matches)) * 100
                coverage = (inliers / min(len(kp1), len(kp2))) * 100
            else:
                match_quality = 0
                coverage = 0
         
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Img 1 Keypoints", len(kp1))
            c2.metric("Img 2 Keypoints", len(kp2))
            c3.metric("Verified Inliers", int(inliers))
            
            color_q = "green" if match_quality > 70 else "orange" if match_quality > 40 else "red"
            color_c = "green" if coverage > 30 else "orange" if coverage > 15 else "red"

            c4.markdown(f"### Match Quality: :{color_q}[{match_quality:.1f}%]")
            st.markdown(f"### Image Coverage: :{color_c}[{coverage:.1f}%]")

            if matches_mask is not None:
                inlier_matches = [matches[i] for i in range(len(matches)) if matches_mask[i] == 1]
                match_viz = cv2.drawMatches(
                    img1, kp1, img2, kp2, 
                    inlier_matches[:30], None, 
                    matchColor=(0, 255, 0), 
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
    
                st.subheader("🖼️ Visual Correspondence (Top 30 Inliers)")
                st.image(cv2.cvtColor(match_viz, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                st.write(
                    f"""
                    **Analysis Summary:**
                    - Total Matches: {len(matches)}
                    - Geometric Inliers: {int(inliers)}
                    - Match Quality: {match_quality:.2f}%
                    - Image Coverage: {coverage:.2f}%
                    """
                )          
            else:
                st.error("Could not find a valid geometric transformation.")

else:
    st.info("Upload imagery to begin feature analysis.")
