# FaceCompareSimplified_dlib.py
import streamlit as st # type: ignore
from PIL import Image, ImageOps # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore
import pandas as pd # type: ignore

# Import your simplified Anthropometer functions
from FaceMetry.Anthropometer import (
    get_landmarks as get_dlib_landmarks,
    calculate_measurements,
    normalize_by_eye_width,
    compare_measurements
)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Face Comparison Tool", layout="wide")
st.title("🧠 Face Verification & Anthropometric Analysis (dlib only)")

# --- Upload Images ---
col1, col2 = st.columns(2)
img1_file = col1.file_uploader("Upload Quotient", type=["jpg", "jpeg", "png"])
img2_file = col2.file_uploader("Upload Sample", type=["jpg", "jpeg", "png"])

# Convert uploaded files to PIL Images
img1, img2 = None, None
if img1_file:
    img1 = Image.open(img1_file).convert("RGB")
if img2_file:
    img2 = Image.open(img2_file).convert("RGB")

def display_image(col, img, label):
    if img:
        w, h = img.size
        col.image(ImageOps.pad(img.copy(), (300, 300)), caption=f"{label}\n({w}×{h}px)")
        col.markdown(f"**Size:** `{w}×{h}` pixels")

display_image(col1, img1, "Quotient")
display_image(col2, img2, "Sample")

# --- Comparison Logic ---
st.subheader("🔍 Compare Faces (dlib)")

if st.button("Run Comparison"):
    if not img1 or not img2:
        st.warning("Please upload both images.")
    else:
        # Convert PIL images to OpenCV format
        img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
        img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

        # Extract landmarks
        lm1 = get_dlib_landmarks(img1_cv)
        lm2 = get_dlib_landmarks(img2_cv)

        if not lm1 or not lm2:
            st.error("⚠️ Unable to detect landmarks in one or both images.")
        else:
            # --- Calculate measurements & normalized ratios ---
            m1 = calculate_measurements(lm1)
            m2 = calculate_measurements(lm2)
            r1 = normalize_by_eye_width(m1)
            r2 = normalize_by_eye_width(m2)

            # --- Simple matching metric ---
            ratio_diff = sum(abs(r1[k] - r2.get(k, 0)) for k in r1) / len(r1)
            threshold = 0.2
            is_match = ratio_diff <= threshold
            confidence = round(max(0, 1 - ratio_diff / threshold) * 100, 2)

            st.metric("Match Verdict", "✅ SAME PERSON" if is_match else "❌ DIFFERENT PERSON")
            st.write(f"Average Ratio Difference: {ratio_diff:.4f} | Threshold: {threshold:.4f} | Confidence: {confidence}%")

            # --- Display normalized ratios table ---
            st.subheader("📏 Normalized Feature Ratios")
            df = pd.DataFrame(compare_measurements(r1, r2))
            st.table(df)
