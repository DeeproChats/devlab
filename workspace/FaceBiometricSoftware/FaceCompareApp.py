# FaceCompare.py
import streamlit as st #type: ignore
from PIL import Image, ImageOps   #type: ignore
from deepface import DeepFace  #type: ignore
import tempfile, os  #type: ignore
import pandas as pd  #type: ignore
import cv2, numpy as np  #type: ignore
import matplotlib.pyplot as plt  #type: ignore
from concurrent.futures import ThreadPoolExecutor
import time

from FaceMetry.Anthropometer import (
    get_dlib_landmarks, calculate_measurements,
    normalize_by_eye_distance, normalize_by_image_diag,
    compare_ratios, plot_measurements_comparison,
    draw_landmarks_with_letter_labels, calculate_golden_ratios
)

# === PERFORMANCE OPTIMIZATION ===
def compute_histogram_data(img1_np, img2_np):
    """Optimized histogram computation"""
    stats = {"Quotient": {}, "Sample": {}}
    color_names = ['Red Channel', 'Green Channel', 'Blue Channel']
    
    for idx, name in enumerate(color_names):
        channel1 = img1_np[:, :, idx]
        channel2 = img2_np[:, :, idx]
        
        stats["Quotient"][name] = {
            "mean": float(np.mean(channel1)),
            "std": float(np.std(channel1)),
            "median": float(np.median(channel1)),
            "hist": np.histogram(channel1.flatten(), bins=256, range=(0, 256))[0]
        }
        
        stats["Sample"][name] = {
            "mean": float(np.mean(channel2)),
            "std": float(np.std(channel2)),
            "median": float(np.median(channel2)),
            "hist": np.histogram(channel2.flatten(), bins=256, range=(0, 256))[0]
        }
    
    return stats

def render_histogram_fast(stats):
    """Fast matplotlib rendering"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    colors = ['#ff4757', '#2ed573', '#1e90ff']
    color_names = ['Red Channel', 'Green Channel', 'Blue Channel']
    
    for idx, (color, name) in enumerate(zip(colors, color_names)):
        # Quotient
        ax1 = axes[0, idx]
        hist1 = stats["Quotient"][name]["hist"]
        ax1.fill_between(range(256), hist1, color=color, alpha=0.8)
        ax1.set_title(f'{name} - Quotient', color='#ecf0f1', fontsize=11, fontweight='bold')
        ax1.set_facecolor('#1a1d29')
        ax1.tick_params(colors='#7f8c8d', labelsize=8)
        ax1.grid(True, alpha=0.15, color='#34495e')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Sample
        ax2 = axes[1, idx]
        hist2 = stats["Sample"][name]["hist"]
        ax2.fill_between(range(256), hist2, color=color, alpha=0.8)
        ax2.set_title(f'{name} - Sample', color='#ecf0f1', fontsize=11, fontweight='bold')
        ax2.set_facecolor('#1a1d29')
        ax2.tick_params(colors='#7f8c8d', labelsize=8)
        ax2.grid(True, alpha=0.15, color='#34495e')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    
    fig.patch.set_facecolor('#0e1117')
    plt.tight_layout()
    return fig

# === PREMIUM UI CONFIG ===
st.set_page_config(
    page_title="Forensic Face Analysis Suite", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === STUNNING MODERN CSS ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d35 100%);
    }
    
    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 1.5rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 600;
        color: #a0aec0 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.6);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.02);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        border-left: 4px solid #667eea;
        font-weight: 600;
        color: #e2e8f0;
    }
    
    /* Tables */
    .dataframe {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    
    .dataframe th {
        background: rgba(102, 126, 234, 0.2) !important;
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    
    .dataframe td {
        color: #cbd5e0 !important;
    }
    
    /* Upload box */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d35 0%, #0a0e27 100%);
    }
    
    /* Success/Error */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Smooth animations */
    * {
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# === HERO HEADER ===
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🔬 Forensic Face Analysis Suite</h1>
    <p class="hero-subtitle">Enterprise-Grade Biometric Verification | Built for Real-World Forensic Conditions</p>
</div>
""", unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_name = st.selectbox("Recognition Model", ["Facenet", "ArcFace", "VGG-Face", "DeepFace"], index=1)
    distance_metric = st.selectbox("Distance Metric", ["cosine", "euclidean", "euclidean_l2"], index=0)
    custom_threshold = st.slider("Threshold", 0.2, 1.0, 0.4, 0.01)
    accessories = st.checkbox("Accessories Worn")
    
    st.markdown("---")
    st.markdown("### 🔧 Advanced Options")
    enforce_detection = st.checkbox("Strict Face Detection", value=False, 
                                    help="Disable for low-quality/partial faces")
    detector_backend = st.selectbox("Detector", ["opencv", "retinaface", "mtcnn", "ssd"], 
                                   index=0, help="opencv = fastest, retinaface = most accurate")
    
    st.markdown("---")
    st.markdown("### 📊 Modules")
    show_histogram = st.checkbox("RGB Analysis", value=True)
    show_anthropometry = st.checkbox("Anthropometry", value=True)
    show_golden = st.checkbox("Golden Ratios", value=False)

# === IMAGE UPLOAD ===
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### 📸 Quotient Image")
    img1_file = st.file_uploader("Upload Reference", type=["jpg", "jpeg", "png"], key="img1", label_visibility="collapsed")
    
with col2:
    st.markdown("#### 📸 Sample Image")
    img2_file = st.file_uploader("Upload Query", type=["jpg", "jpeg", "png"], key="img2", label_visibility="collapsed")

img1, img2 = None, None
if img1_file: img1 = Image.open(img1_file)
if img2_file: img2 = Image.open(img2_file)

if img1:
    with col1:
        st.image(img1, use_container_width=True, caption=f"Resolution: {img1.size[0]}×{img1.size[1]}px")

if img2:
    with col2:
        st.image(img2, use_container_width=True, caption=f"Resolution: {img2.size[0]}×{img2.size[1]}px")

# === ANALYSIS BUTTON ===
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 RUN ANALYSIS", use_container_width=True, type="primary"):
    if not img1 or not img2:
        st.error("⚠️ Please upload both images")
    else:
        # Save temp files with error handling
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp1:
                img1.convert("RGB").save(tmp1.name, quality=95)
                path1 = tmp1.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
                img2.convert("RGB").save(tmp2.name, quality=95)
                path2 = tmp2.name
        except Exception as e:
            st.error(f"❌ Error saving images: {str(e)}")
            st.stop()

        # === ANALYSIS CONTAINER ===
        analysis_container = st.container()
        
        with analysis_container:
            progress_bar = st.progress(0, text="Initializing forensic analysis...")
            
            # === HISTOGRAM ANALYSIS (Always runs) ===
            hist_stats = None
            if show_histogram:
                try:
                    progress_bar.progress(10, text="Computing RGB histograms...")
                    img1_np = np.array(img1)
                    img2_np = np.array(img2)
                    hist_stats = compute_histogram_data(img1_np, img2_np)
                    progress_bar.progress(30, text="Histogram analysis complete ✓")
                except Exception as e:
                    st.warning(f"⚠️ Histogram analysis failed: {str(e)}")
            
            # === DEEPFACE VERIFICATION (Robust) ===
            deepface_result = None
            deepface_error = None
            
            try:
                progress_bar.progress(40, text="Running DeepFace verification...")
                
                # Try with relaxed settings first
                deepface_result = DeepFace.verify(
                    img1_path=path1,
                    img2_path=path2,
                    model_name=model_name,
                    distance_metric=distance_metric,
                    enforce_detection=enforce_detection,
                    detector_backend=detector_backend,
                    align=True
                )
                progress_bar.progress(70, text="DeepFace verification complete ✓")
                
            except Exception as e:
                deepface_error = str(e)
                st.warning(f"⚠️ DeepFace failed with current settings. Trying alternative approach...")
                
                # Fallback: Try different detector
                try:
                    deepface_result = DeepFace.verify(
                        img1_path=path1,
                        img2_path=path2,
                        model_name=model_name,
                        distance_metric=distance_metric,
                        enforce_detection=False,
                        detector_backend="opencv",
                        align=False
                    )
                    st.success("✓ Analysis completed with relaxed detection mode")
                except Exception as e2:
                    st.error(f"❌ DeepFace verification failed: {str(e2)}")
                    st.info("💡 **Possible reasons:**\n- Face not clearly visible\n- Poor image quality\n- Extreme angle/occlusion\n\n**Solution:** Try different detector in sidebar or upload clearer images")
            
            # === ANTHROPOMETRY (Robust) ===
            anthropometry_data = None
            
            if show_anthropometry:
                try:
                    progress_bar.progress(80, text="Extracting facial landmarks...")
                    img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
                    img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
                    lm1 = get_dlib_landmarks(img1_cv)
                    lm2 = get_dlib_landmarks(img2_cv)
                    
                    if lm1 and lm2:
                        anthropometry_data = {
                            'lm1': lm1, 'lm2': lm2,
                            'img1_cv': img1_cv, 'img2_cv': img2_cv,
                            'm1': calculate_measurements(lm1),
                            'm2': calculate_measurements(lm2)
                        }
                        progress_bar.progress(95, text="Anthropometry complete ✓")
                    else:
                        st.warning("⚠️ Landmark detection failed - skipping anthropometry")
                except Exception as e:
                    st.warning(f"⚠️ Anthropometry failed: {str(e)}")
            
            progress_bar.progress(100, text="Analysis complete!")
            time.sleep(0.3)
            progress_bar.empty()
            
            # === RESULTS DISPLAY ===
            tab1, tab2, tab3 = st.tabs(["🧠 Verification", "📊 RGB Analysis", "📐 Anthropometry"])
            
            # === TAB 1: DEEPFACE ===
            with tab1:
                if deepface_result:
                    distance = deepface_result["distance"]
                    buffer = 0.05 if accessories else 0
                    adjusted_threshold = custom_threshold + buffer
                    is_match = distance <= adjusted_threshold
                    confidence = round(max(0, 1 - (distance / adjusted_threshold)) * 100, 2)
                    
                    m1, m2, m3, m4 = st.columns(4)
                    
                    with m1:
                        st.metric("Verdict", "✅ MATCH" if is_match else "❌ NO MATCH")
                    with m2:
                        st.metric("Distance", f"{distance:.4f}")
                    with m3:
                        st.metric("Threshold", f"{adjusted_threshold:.4f}")
                    with m4:
                        st.metric("Confidence", f"{confidence}%")
                    
                    with st.expander("📐 Calculation Details"):
                        st.markdown(f"""
                        **Formula:** `distance ≤ threshold → Match`
                        
                        - Computed Distance: `{distance:.4f}`
                        - Base Threshold: `{custom_threshold:.4f}`
                        - Accessories Buffer: `{buffer:.4f}`
                        - Final Threshold: `{adjusted_threshold:.4f}`
                        - **Result:** {"✅ MATCH" if is_match else "❌ NO MATCH"}
                        
                        **Confidence:** `{confidence}%` = `(1 - distance/threshold) × 100`
                        
                        **Model:** {model_name} | **Metric:** {distance_metric} | **Detector:** {detector_backend}
                        """)
                else:
                    st.error("❌ DeepFace verification could not complete")
                    if deepface_error:
                        st.code(deepface_error)
                    st.info("""
                    **Troubleshooting Steps:**
                    1. Disable "Strict Face Detection" in sidebar
                    2. Try different detector (opencv is most forgiving)
                    3. Ensure face is visible and not heavily occluded
                    4. Try a different model (ArcFace works better for low-quality images)
                    """)
            
            # === TAB 2: HISTOGRAM ===
            with tab2:
                if hist_stats:
                    fig = render_histogram_fast(hist_stats)
                    st.pyplot(fig, use_container_width=True)
                    
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.markdown("##### 📈 Quotient Stats")
                        for ch, stats in hist_stats["Quotient"].items():
                            st.markdown(f"**{ch}** • Mean: `{stats['mean']:.1f}` • Std: `{stats['std']:.1f}`")
                    
                    with c2:
                        st.markdown("##### 📈 Sample Stats")
                        for ch, stats in hist_stats["Sample"].items():
                            st.markdown(f"**{ch}** • Mean: `{stats['mean']:.1f}` • Std: `{stats['std']:.1f}`")
                    
                    with st.expander("🔍 Forensic Interpretation"):
                        st.markdown("""
                        ### What RGB Histograms Reveal
                        
                        **Histogram = Color Fingerprint** showing pixel intensity distribution (0-255) across Red, Green, Blue channels.
                        
                        #### Processing Steps:
                        1. **Decompose** image into R, G, B channels
                        2. **Count** frequency of each intensity (0-255)
                        3. **Calculate** statistical properties
                        
                        #### Forensic Applications:
                        - **Lighting Consistency**: Similar peaks = same environment
                        - **Authenticity**: Unnatural gaps indicate editing
                        - **Camera Matching**: Same device = similar patterns
                        - **Compression**: Spikes reveal over-compression
                        
                        #### Interpretation:
                        - Mean diff >30 → Different lighting
                        - Std diff >20 → Different contrast
                        - Similar patterns → Likely same conditions
                        
                        ⚠️ **Legal Note**: Supplementary evidence only
                        """)
                else:
                    st.info("Histogram analysis not performed")
            
            # === TAB 3: ANTHROPOMETRY ===
            with tab3:
                if anthropometry_data:
                    lm1 = anthropometry_data['lm1']
                    lm2 = anthropometry_data['lm2']
                    m1 = anthropometry_data['m1']
                    m2 = anthropometry_data['m2']
                    
                    annotated1, legend1, segs1 = draw_landmarks_with_letter_labels(
                        anthropometry_data['img1_cv'].copy(), lm1, normalized=True, 
                        reference_value=m1["Eye Width (outer)"]
                    )
                    annotated2, legend2, segs2 = draw_landmarks_with_letter_labels(
                        anthropometry_data['img2_cv'].copy(), lm2, normalized=True, 
                        reference_value=m2["Eye Width (outer)"]
                    )
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(annotated1, caption="Quotient Landmarks", use_container_width=True)
                    with c2:
                        st.image(annotated2, caption="Sample Landmarks", use_container_width=True)
                    
                    r1_eye = normalize_by_eye_distance(m1)
                    r2_eye = normalize_by_eye_distance(m2)
                    
                    st.markdown("##### 📏 Normalized Ratios")
                    df_eye = pd.DataFrame(compare_ratios(r1_eye, r2_eye)["rows"])
                    st.dataframe(df_eye, use_container_width=True)
                    
                    if show_golden:
                        golden1 = calculate_golden_ratios(m1)
                        golden2 = calculate_golden_ratios(m2)
                        df_golden = pd.DataFrame({
                            "Ratio": list(golden1.keys()),
                            "Quotient": list(golden1.values()),
                            "Sample": [golden2.get(k, 0.0) for k in golden1.keys()],
                            "Ideal": [1.618 if "Hairline" in k else "Varies" for k in golden1.keys()]
                        })
                        st.dataframe(df_golden, use_container_width=True)
                    
                    fig = plot_measurements_comparison(m1, m2)
                    st.pyplot(fig, use_container_width=True)
                    
                    with st.expander("📚 Measurement Formulas"):
                        st.markdown(f"""
                        ### Landmark Coordinates
                        
                        | Label | Point | Quotient | Sample |
                        |-------|-------|----------|--------|
                        | A | Left Eye | {lm1['left_eye']} | {lm2['left_eye']} |
                        | B | Right Eye | {lm1['right_eye']} | {lm2['right_eye']} |
                        | C | Nose Tip | {lm1['nose_tip']} | {lm2['nose_tip']} |
                        | G | Chin | {lm1['chin']} | {lm2['chin']} |
                        
                        **Distance:** `d = √[(x₂-x₁)² + (y₂-y₁)²]`
                        
                        **Normalization:** All divided by eye distance for scale independence
                        """)
                else:
                    st.info("Anthropometry analysis not available - landmark detection failed")
        
        # Cleanup
        try:
            os.remove(path1)
            os.remove(path2)
        except:
            pass

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; font-size: 0.875rem;'>
    Forensic Face Analysis Suite v2.1 | Built for Real-World Forensic Conditions
</div>
""", unsafe_allow_html=True)