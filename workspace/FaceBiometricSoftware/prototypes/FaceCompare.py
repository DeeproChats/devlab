# FaceCompare.py
import streamlit as st #type: ignore
from PIL import Image, ImageOps   #type: ignore
from deepface import DeepFace  #type: ignore
import tempfile, os  #type: ignore
import pandas as pd  #type: ignore
import cv2, numpy as np  #type: ignore

from FaceMetry.Anthropometer import (
    get_dlib_landmarks, calculate_measurements,
    normalize_by_eye_distance, normalize_by_image_diag,
    compare_ratios, plot_measurements_comparison,
    draw_landmarks_with_letter_labels, calculate_golden_ratios
)

# === UI CONFIG ===
st.set_page_config(page_title="Face Comparison Tool", layout="wide")
st.title("🧠 Face Verification & Anthropometric Analysis")

# === Uploads ===
col1, col2 = st.columns(2)
img1_file = col1.file_uploader("Upload Quotient", type=["jpg", "jpeg", "png"])
img2_file = col2.file_uploader("Upload Sample", type=["jpg", "jpeg", "png"])

img1, img2 = None, None
if img1_file: img1 = Image.open(img1_file)
if img2_file: img2 = Image.open(img2_file)
if img1:
    width1, height1 = img1.size
    col1.image(ImageOps.pad(img1.copy(), (300, 300)), caption=f"Quotient\n({width1} × {height1} pixels)")
    col1.markdown(f"**Quotient Size:** `{width1} × {height1}` pixels")

if img2:
    width2, height2 = img2.size
    col2.image(ImageOps.pad(img2.copy(), (300, 300)), caption=f"Sample\n({width2} × {height2} pixels)")
    col2.markdown(f"**Sample Size:** `{width2} × {height2}` pixels")


# === DeepFace Settings ===
st.subheader("🔧 DeepFace Settings")
model_name = st.selectbox("Model", ["Facenet", "ArcFace", "VGG-Face", "DeepFace"])
distance_metric = st.selectbox("Distance Metric", ["cosine", "euclidean", "euclidean_l2"])
custom_threshold = st.slider("Custom Threshold", 0.2, 1.0, 0.4, 0.01)
accessories = st.checkbox("Wearing accessories?")

# === Main Logic ===
st.subheader("🔍 Compare Faces")
if st.button("Run Comparison"):
    if not img1 or not img2:
        st.warning("Please upload both images.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp1:
            img1.convert("RGB").save(tmp1.name)
            path1 = tmp1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
            img2.convert("RGB").save(tmp2.name)
            path2 = tmp2.name

        try:
            with st.spinner("Analyzing faces..."):
                result = DeepFace.verify(
                    img1_path=path1,
                    img2_path=path2,
                    model_name=model_name,
                    distance_metric=distance_metric,
                    enforce_detection=True
                )

                distance = result["distance"]
                buffer = 0.05 if accessories else 0
                adjusted_threshold = custom_threshold + buffer
                is_match = distance <= adjusted_threshold
                confidence = round(max(0, 1 - (distance / adjusted_threshold)) * 100, 2)

                st.subheader("🧠 DeepFace Result")
                st.metric("Match Verdict", "✅ SAME PERSON" if is_match else "❌ DIFFERENT PERSON")
                st.write(f"**Distance:** {distance:.4f}")
                st.write(f"**Threshold Used:** {adjusted_threshold:.4f}")
                st.write(f"**Confidence Estimate:** {confidence}%")

                with st.expander("📌 Raw Embedding Output"):
                    st.json(result.get("embedding", result))

            with st.expander("📐 Calculation Method & Interpretation"):
                st.markdown(f"""
                ### 🔍 Matching Formula:
                The DeepFace model computes a distance metric between two facial embeddings.
                This distance represents how similar the two faces are in terms of their features.
                It compares facial embeddings using this rule:

                ```
                if distance <= threshold → verified = True
                else → verified = False
                ```
                the threshold is a value used to determine the similarity between two face embeddings

                ### 📊 Step-by-Step Calculation:
                - **Computed embedding Distance**: `{distance:.4f}`
                - **Custom Threshold**: `{custom_threshold:.4f}`
                - **Accessories Buffer**: `{buffer:.4f}`
                - **Adjusted Threshold**: `{adjusted_threshold:.4f}`
                - **Final Decision**: {"✅ MATCH" if is_match else "❌ NO MATCH"}

                ### 🧠 Confidence Score:
                We estimate a normalized similarity confidence as:

                ```
                confidence = 1 - (distance / adjusted_threshold)
                ```
                - **Resulting Confidence**: `{confidence}%`
                ### 📚 Interpretation:
                - Lower distance = higher similarity
                - The threshold is a model-specific boundary (empirically derived)
                - When faces are occluded or partially obstructed, the system may show higher distances but still within tolerance
                - Confidence % offers a soft measure of similarity — not a legal metric, but helpful for interpretation
                """)

            # === Anthropometry Analysis ===
            img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
            lm1 = get_dlib_landmarks(img1_cv)
            lm2 = get_dlib_landmarks(img2_cv)

            if not lm1 or not lm2:
                st.error("⚠️ Unable to detect landmarks in one or both images.")
            else:
                m1 = calculate_measurements(lm1)
                m2 = calculate_measurements(lm2)

                r1_eye = normalize_by_eye_distance(m1)
                r2_eye = normalize_by_eye_distance(m2)

                r1_diag = normalize_by_image_diag(m1, img1_cv.shape)
                r2_diag = normalize_by_image_diag(m2, img2_cv.shape)

                # === Landmark + Measurements Visualization ===
                st.subheader("📌 Landmark Mapping with Letter Annotations")
                annotated1, legend1, segs1 = draw_landmarks_with_letter_labels(
                    img1_cv.copy(), lm1, normalized=True, reference_value=m1["Eye Width (outer)"]
                )
                annotated2, legend2, segs2 = draw_landmarks_with_letter_labels(
                    img2_cv.copy(), lm2, normalized=True, reference_value=m2["Eye Width (outer)"]
                )
                st.image([annotated1, annotated2], caption=["Quotient Annotated", "Sample Annotated"], width=320)

                with st.expander("🔠 Landmark Letter Legend"):
                    st.write("**Quotient Landmarks:**")
                    st.json(legend1)
                    st.write("**Sample Landmarks:**")
                    st.json(legend2)

                with st.expander("📏 Measurement Segment Distances"):
                    st.write("**Quotient Distances:**")
                    st.json(segs1)
                    st.write("**Sample Distances:**")
                    st.json(segs2)

                with st.expander("📚 Anthropometric Measurement Explanation"):
                    st.markdown(f"""
                    ### Landmark Coordinates (Quotient & Sample)

                    | Label | Landmark             | Quotient (x, y)       | Sample (x, y)         |
                    |-------|----------------------|-----------------------|-----------------------|
                    | A     | left_eye             | {lm1['left_eye']}     | {lm2['left_eye']}     |
                    | B     | right_eye            | {lm1['right_eye']}    | {lm2['right_eye']}    |
                    | C     | nose_tip             | {lm1['nose_tip']}     | {lm2['nose_tip']}     |
                    | D     | mouth_center         | {lm1['mouth_center']} | {lm2['mouth_center']} |
                    | E     | left_ear             | {lm1['left_ear']}     | {lm2['left_ear']}     |
                    | F     | right_ear            | {lm1['right_ear']}    | {lm2['right_ear']}    |
                    | G     | chin                 | {lm1['chin']}         | {lm2['chin']}         |
                    | H     | forehead_glabella    | {lm1['forehead_glabella']} | {lm2['forehead_glabella']} |
                    | I     | jaw_left             | {lm1['jaw_left']}     | {lm2['jaw_left']}     |
                    | J     | jaw_right            | {lm1['jaw_right']}    | {lm2['jaw_right']}    |

                    ### Formulas for Facial Measurements

                    | Feature              | Formula                                                                 |
                    |----------------------|-------------------------------------------------------------------------|
                    | Eye Width (outer)    | \( \sqrt{{(x_B - x_A)^2 + (y_B - y_A)^2}} \)                              |
                    | Nose Width           | \( \sqrt{{(x_J - x_I)^2 + (y_J - y_I)^2}} \)                              |
                    | Nose to Chin         | \( \sqrt{{(x_G - x_C)^2 + (y_G - y_C)^2}} \)                              |
                    | Nose to Mouth        | \( \sqrt{{(x_D - x_C)^2 + (y_D - y_C)^2}} \)                              |
                    | Hairline to Nose     | \( \sqrt{{(x_H - x_C)^2 + (y_H - y_C)^2}} \)                              |
                    | Hairline to Chin     | \( \sqrt{{(x_H - x_G)^2 + (y_H - y_G)^2}} \)                              |
                    | Pupil Line to Chin   | \( \sqrt{{(x_A - x_G)^2 + (y_A - y_G)^2}} \)                              |
                    | Eye to Nose Base     | \( \sqrt{{(x_A - x_C)^2 + (y_A - y_C)^2}} \)                              |
                    | Mouth Width          | \( \sqrt{{(x_I - x_J)^2 + (y_I - y_J)^2}} \)                              |

                    These formulas use landmark coordinates to calculate distances between features using Euclidean geometry.

                    ### Example Distance Calculation (Image 1)

                    ```python
                    A = lm1['left_eye']
                    B = lm1['right_eye']
                    distance_AB = euclidean_distance(A, B)
                    ```

                    This uses the provided `euclidean_distance` utility function for measurement computation.
                    """)

                # === Ratio Tables ===
                st.subheader("📏 Facial Feature Ratios (Normalized by Eye Width)")
                df_eye = pd.DataFrame(compare_ratios(r1_eye, r2_eye)["rows"])
                st.table(df_eye)
              
                with st.expander("📏 Why Normalize by Eye Distance?"):
                    st.markdown("""
                    When comparing faces, we can't rely on raw pixel distances because:

                    - Different images may have **different resolutions**, zoom levels, or head sizes.  
                    - A 100px nose in a zoomed-in image is **not** the same as a 100px nose in a distant image.

                    Hence, we normalize the measurements using a consistent facial anchor:

                    | **Normalization Method**     | **What It Does**                                                                                                           |
                    |------------------------------|----------------------------------------------------------------------------------------------------------------------------|
                    | 👁️ **Eye-to-Eye Distance**   | Uses the distance between eye corners as a stable reference. Eye landmarks are typically visible and less affected by pose. |

                    ---

                    ✅ This ensures that **shape** — not **size** — becomes the basis for comparing facial geometry.
                    """)

                # ➕ Add this
                with st.expander("🧠 Math Behind Anthropometry"):
                    st.markdown("""
                    | **Math Chapter**         | **Application in Facial Measurement**                                   |
                    |--------------------------|-------------------------------------------------------------------------|
                    | 🔢 **Ratio & Proportion**        | Used to normalize features like `Nose→Mouth / Eye→Eye`. Allows size-independent comparison. |
                    | 📐 **Similarity of Triangles**   | Facial structure remains proportional even when image scale changes — like similar triangles. |
                    | 📍 **Coordinate Geometry**       | Calculates exact distances between points using `(x, y)` positions from facial landmarks. |
                    | 📈 **Trigonometry (advanced)**   | Useful in 3D analysis: head tilt, yaw, pitch, roll — not yet implemented here. |
                    | 🧮 **Linear Algebra (DeepFace)** | DeepFace uses high-dimensional **vectors** to encode facial identity and compute similarity. |
                    """)

                
                # === Golden Ratio Section ===
                st.subheader("📐 Golden Ratio Comparisons (for reference)")
                golden1 = calculate_golden_ratios(m1)
                golden2 = calculate_golden_ratios(m2)
                df_golden = pd.DataFrame({
                    "Ratio": list(golden1.keys()),
                    "Quotient": list(golden1.values()),
                    "Sample": [golden2.get(k, 0.0) for k in golden1.keys()],
                    "Ideal": [1.618 if "Hairline" in k else "Varies" for k in golden1.keys()]
                })
                st.table(df_golden)

                # === Pixel Comparison Bar Chart ===
                st.subheader("📊 Raw Measurement Comparison (in Pixels)")
                fig = plot_measurements_comparison(m1, m2)
                st.pyplot(fig)

                
                
        except Exception as e:
            st.error("❌ Error during analysis.")
            st.exception(e)

        finally:
            os.remove(path1)
            os.remove(path2)