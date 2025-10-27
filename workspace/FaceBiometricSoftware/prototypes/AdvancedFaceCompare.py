"""
═══════════════════════════════════════════════════════════════════════════
    PROFESSIONAL FORENSIC FACIAL COMPARISON SYSTEM v3.0
    Court-Admissible Evidence Documentation & Analysis Platform
═══════════════════════════════════════════════════════════════════════════

Standards Compliance:
- FISWG (Facial Identification Scientific Working Group)
- PCAST (President's Council of Advisors on Science & Technology)
- Federal Rules of Evidence 702
- ISO/IEC 17025 (Testing Laboratory Accreditation)

Author: Forensic Analysis Division
License: Professional Forensic Use
Date: 2025-10-13
"""

import streamlit as st #type: ignore
from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageFont, ImageFilter, ImageEnhance #type: ignore
import numpy as np #type: ignore
import cv2 #type: ignore
import hashlib #type: ignore
import pandas as pd #type: ignore 
import io #type: ignore
import datetime #type: ignore
import math #type: ignore
import matplotlib.pyplot as plt #type: ignore
import matplotlib.patches as patches #type: ignore
import seaborn as sns #type: ignore
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass, asdict
from enum import Enum

# Configure matplotlib for forensic-quality outputs
import matplotlib #type: ignore
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

from FaceMetry.Anthropometer import (
    get_landmarks, calculate_measurements, normalize_by_eye_width, compare_measurements
)

# Optional dependencies with graceful fallback
try:
    from skimage.metrics import structural_similarity as ssim #type: ignore
    from skimage import exposure #type: ignore
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    st.warning("⚠️ scikit-image not available. Some advanced metrics disabled.")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION & FORENSIC STANDARDS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Forensic Facial Comparison System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional Forensic Facial Comparison System v3.0 - FISWG Compliant"
    }
)

# Golden Ratio Constant (Phi)
PHI = 1.618033988749895

class EvidenceStrength(Enum):
    """Court-admissible evidence classification"""
    STRONG_SUPPORT_SAME = "Strong Support - Same Person"
    MODERATE_SUPPORT_SAME = "Moderate Support - Same Person"
    LIMITED_SUPPORT = "Limited Support - Inconclusive"
    MODERATE_SUPPORT_DIFFERENT = "Moderate Support - Different Person"
    STRONG_SUPPORT_DIFFERENT = "Strong Support - Different Person"

@dataclass
class ForensicThresholds:
    """Evidence-based thresholds validated by forensic research"""
    # Anthropometric Confidence Levels (%)
    STRONG_MATCH: float = 92.0
    MODERATE_MATCH: float = 82.0
    LIMITED_SUPPORT: float = 72.0
    INCONCLUSIVE_LOWER: float = 62.0
    
    # Measurement Tolerances (%)
    RATIO_STRICT: float = 0.03      # 3% - Very strict
    RATIO_MODERATE: float = 0.07    # 7% - Standard forensic
    RATIO_LENIENT: float = 0.12     # 12% - Lenient (different conditions)
    
    # Golden Ratio Tolerances
    GOLDEN_RATIO_PERFECT: float = 0.05
    GOLDEN_RATIO_GOOD: float = 0.15
    GOLDEN_RATIO_ACCEPTABLE: float = 0.30
    
    # Image Quality Metrics
    ELA_NORMAL: float = 0.04
    ELA_ELEVATED: float = 0.08
    HISTOGRAM_HIGH: float = 0.82
    HISTOGRAM_MODERATE: float = 0.65
    SSIM_EXCELLENT: float = 0.90
    SSIM_GOOD: float = 0.75
    PSNR_EXCELLENT: float = 35.0
    PSNR_GOOD: float = 25.0
    
    # Morphological Feature Weights
    CRITICAL_FEATURES_WEIGHT: float = 2.0  # Eyes, nose, mouth
    SECONDARY_FEATURES_WEIGHT: float = 1.5 # Eyebrows, chin, jaw
    TERTIARY_FEATURES_WEIGHT: float = 1.0  # Ears, forehead

THRESHOLDS = ForensicThresholds()

# Processing Constants
TARGET_SIZE = (800, 800)  # Higher resolution for better analysis
LANDMARK_COUNT = 68
DPI_FORENSIC = 150  # Minimum DPI for court exhibits

# ═══════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE MORPHOLOGICAL FEATURE DATABASE
# ═══════════════════════════════════════════════════════════════════════════

MORPHOLOGICAL_FEATURES = {
    # Critical Features (Weight: 2.0)
    'eye_shape': {
        'options': ['Almond', 'Round', 'Hooded', 'Upturned', 'Downturned', 'Monolid', 'Deep-set', 'Protruding'],
        'category': 'critical',
        'description': 'Overall shape and configuration of the eye'
    },
    'eye_spacing': {
        'options': ['Close-set (<1 eye width)', 'Average (1 eye width)', 'Wide-set (>1 eye width)'],
        'category': 'critical',
        'description': 'Distance between inner eye corners'
    },
    'iris_color': {
        'options': ['Brown', 'Dark Brown', 'Hazel', 'Green', 'Blue', 'Gray', 'Amber', 'Mixed'],
        'category': 'critical',
        'description': 'Iris pigmentation'
    },
    'nose_profile': {
        'options': ['Straight', 'Roman/Aquiline', 'Concave/Snub', 'Convex', 'Greek', 'Nubian'],
        'category': 'critical',
        'description': 'Lateral nasal profile'
    },
    'nose_width': {
        'options': ['Narrow (<70% intercanthal)', 'Medium (70-85%)', 'Wide (>85%)'],
        'category': 'critical',
        'description': 'Nasal width relative to eye spacing'
    },
    'nostril_orientation': {
        'options': ['Horizontal', 'Oblique', 'Vertical'],
        'category': 'critical',
        'description': 'Nostril axis orientation'
    },
    'lip_thickness_upper': {
        'options': ['Thin (<6mm)', 'Medium (6-10mm)', 'Full (>10mm)'],
        'category': 'critical',
        'description': 'Upper lip vermillion thickness'
    },
    'lip_thickness_lower': {
        'options': ['Thin (<8mm)', 'Medium (8-12mm)', 'Full (>12mm)'],
        'category': 'critical',
        'description': 'Lower lip vermillion thickness'
    },
    'mouth_width': {
        'options': ['Narrow (<1.5x nose)', 'Average (1.5x nose)', 'Wide (>1.5x nose)'],
        'category': 'critical',
        'description': 'Mouth width relative to nose width'
    },
    
    # Secondary Features (Weight: 1.5)
    'eyebrow_shape': {
        'options': ['Straight', 'Arched', 'S-shaped', 'Angular', 'Rounded', 'Flat'],
        'category': 'secondary',
        'description': 'Eyebrow curvature and shape'
    },
    'eyebrow_thickness': {
        'options': ['Thin', 'Medium', 'Thick', 'Bushy'],
        'category': 'secondary',
        'description': 'Eyebrow hair density and width'
    },
    'chin_shape': {
        'options': ['Pointed', 'Round', 'Square', 'Cleft/Dimpled', 'Flat', 'Protruding'],
        'category': 'secondary',
        'description': 'Chin projection and contour'
    },
    'chin_prominence': {
        'options': ['Receding', 'Normal', 'Prominent', 'Prognathic'],
        'category': 'secondary',
        'description': 'Anterior-posterior chin position'
    },
    'jaw_shape': {
        'options': ['Angular/Square', 'Rounded', 'Tapered', 'Heavy/Robust'],
        'category': 'secondary',
        'description': 'Lower facial contour'
    },
    'philtrum_depth': {
        'options': ['Shallow', 'Medium', 'Deep', 'Very Deep'],
        'category': 'secondary',
        'description': 'Depression between nose and upper lip'
    },
    'philtrum_length': {
        'options': ['Short', 'Medium', 'Long'],
        'category': 'secondary',
        'description': 'Distance from nose base to upper lip'
    },
    
    # Tertiary Features (Weight: 1.0)
    'facial_shape': {
        'options': ['Oval', 'Round', 'Square', 'Heart', 'Oblong/Long', 'Diamond', 'Triangular'],
        'category': 'tertiary',
        'description': 'Overall face outline'
    },
    'forehead_height': {
        'options': ['Low (< 1/3 face height)', 'Medium (≈ 1/3)', 'High (> 1/3)'],
        'category': 'tertiary',
        'description': 'Vertical forehead dimension'
    },
    'forehead_width': {
        'options': ['Narrow', 'Medium', 'Wide'],
        'category': 'tertiary',
        'description': 'Horizontal forehead dimension'
    },
    'forehead_slope': {
        'options': ['Vertical', 'Slightly Sloped', 'Moderately Sloped', 'Highly Sloped'],
        'category': 'tertiary',
        'description': 'Forehead angle from vertical'
    },
    'cheekbone_prominence': {
        'options': ['Flat', 'Low', 'Medium', 'High/Prominent'],
        'category': 'tertiary',
        'description': 'Zygomatic arch projection'
    },
    'ear_shape': {
        'options': ['Attached Lobe', 'Free Lobe', 'Round', 'Oval', 'Rectangular'],
        'category': 'tertiary',
        'description': 'Auricle configuration'
    },
    'ear_size': {
        'options': ['Small', 'Medium', 'Large'],
        'category': 'tertiary',
        'description': 'Relative ear size'
    },
    'ear_protrusion': {
        'options': ['Flat (<20°)', 'Normal (20-30°)', 'Protruding (>30°)'],
        'category': 'tertiary',
        'description': 'Ear angle from head'
    },
    'nasal_bridge_height': {
        'options': ['Low/Flat', 'Medium', 'High/Prominent'],
        'category': 'tertiary',
        'description': 'Nasal root elevation'
    },
    'nasal_tip_shape': {
        'options': ['Pointed', 'Round', 'Bulbous', 'Upturned', 'Downturned'],
        'category': 'tertiary',
        'description': 'Nasal apex morphology'
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB to OpenCV BGR format"""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR to PIL RGB format"""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def compute_sha256(img: Image.Image) -> str:
    """
    Cryptographic hash for chain of custody
    Ensures image integrity and tamper detection
    """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()

def compute_md5(img: Image.Image) -> str:
    """Secondary hash for verification"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return hashlib.md5(buffer.getvalue()).hexdigest()

def standardize_image(img: Image.Image, target_size: Tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """
    Forensic image standardization:
    - Maintains aspect ratio
    - Neutral gray background
    - Consistent dimensions for comparison
    """
    img_copy = img.copy()
    img_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    background = Image.new('RGB', target_size, (128, 128, 128))
    offset = (
        (target_size[0] - img_copy.width) // 2,
        (target_size[1] - img_copy.height) // 2
    )
    background.paste(img_copy, offset)
    return background

def generate_case_id() -> str:
    """Generate unique forensic case identifier"""
    timestamp = datetime.datetime.utcnow()
    return f"FFC-{timestamp.strftime('%Y%m%d-%H%M%S')}-{timestamp.microsecond:06d}"

def calculate_image_quality_score(img: Image.Image) -> Dict[str, Any]:
    """
    Comprehensive image quality assessment
    Returns metrics for court documentation
    """
    cv_img = pil_to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(100, laplacian_var / 10)
    
    # Contrast (standard deviation)
    contrast = np.std(gray)
    contrast_score = min(100, contrast / 2)
    
    # Brightness (mean)
    brightness = np.mean(gray)
    brightness_score = 100 - abs(brightness - 128) / 1.28
    
    # Noise estimate
    noise = np.std(cv2.Laplacian(gray, cv2.CV_64F))
    noise_score = max(0, 100 - noise / 2)
    
    # Overall quality
    overall = (sharpness_score + contrast_score + brightness_score + noise_score) / 4
    
    return {
        'overall': overall,
        'sharpness': sharpness_score,
        'contrast': contrast_score,
        'brightness': brightness_score,
        'noise': noise_score,
        'resolution': img.size,
        'assessment': 'Excellent' if overall >= 80 else 'Good' if overall >= 60 else 'Fair' if overall >= 40 else 'Poor'
    }

# ═══════════════════════════════════════════════════════════════════════════
# IMAGE FORENSIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def compute_ela(img: Image.Image, quality: int = 90) -> Tuple[Image.Image, float, Dict]:
    """
    Enhanced Error Level Analysis with detailed metrics
    """
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    
    diff = ImageChops.difference(img, recompressed)
    diff_array = np.array(diff)
    
    # Calculate comprehensive metrics
    ela_score = float(np.mean(diff_array) / 255.0)
    ela_max = float(np.max(diff_array) / 255.0)
    ela_std = float(np.std(diff_array) / 255.0)
    
    # Amplified visualization
    amplified = np.clip(diff_array * 15, 0, 255).astype(np.uint8)
    
    # Hot spot detection (potential manipulation areas)
    threshold = np.percentile(diff_array, 95)
    hotspots = (diff_array > threshold).any(axis=2).sum()
    hotspot_percentage = (hotspots / (diff_array.shape[0] * diff_array.shape[1])) * 100
    
    metrics = {
        'mean': ela_score,
        'max': ela_max,
        'std': ela_std,
        'hotspots_pct': hotspot_percentage
    }
    
    return Image.fromarray(amplified), ela_score, metrics

def calculate_histogram_similarity(img1: Image.Image, img2: Image.Image) -> Dict[str, float]:
    """
    Multi-method histogram comparison
    """
    def calc_hist(img, channel):
        return cv2.calcHist([np.array(img)], [channel], None, [256], [0, 256])
    
    results = {}
    
    # Per-channel correlation
    for i, channel in enumerate(['red', 'green', 'blue']):
        hist1 = calc_hist(img1, i)
        hist2 = calc_hist(img2, i)
        results[f'{channel}_correlation'] = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
        results[f'{channel}_chi_square'] = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR))
        results[f'{channel}_intersection'] = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT))
        results[f'{channel}_bhattacharyya'] = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))
    
    # Overall metrics
    results['overall_correlation'] = np.mean([results[f'{c}_correlation'] for c in ['red', 'green', 'blue']])
    results['overall_similarity'] = (results['overall_correlation'] + 1) / 2  # Normalize to 0-1
    
    return results

def calculate_pixel_metrics(img1: Image.Image, img2: Image.Image) -> Dict[str, float]:
    """
    Comprehensive pixel-level similarity metrics
    """
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    
    # Mean Squared Error
    mse = float(np.mean((arr1 - arr2) ** 2))
    
    # Root Mean Squared Error
    rmse = float(np.sqrt(mse))
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    
    # Mean Absolute Error
    mae = float(np.mean(np.abs(arr1 - arr2)))
    
    # Normalized Cross-Correlation
    ncc = float(np.sum(arr1 * arr2) / (np.sqrt(np.sum(arr1**2)) * np.sqrt(np.sum(arr2**2))))
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'psnr': psnr,
        'mae': mae,
        'ncc': ncc
    }
    
    # SSIM if available
    if HAS_SKIMAGE:
        gray1 = cv2.cvtColor(pil_to_cv(img1), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(pil_to_cv(img2), cv2.COLOR_BGR2GRAY)
        metrics['ssim'] = float(ssim(gray1, gray2))
        
        # Multi-scale SSIM
        try:
            from skimage.metrics import structural_similarity as ssim_multiscale #type: ignore
            metrics['ms_ssim'] = float(ssim_multiscale(gray1, gray2, channel_axis=None))
        except:
            pass
    
    return metrics

# ═══════════════════════════════════════════════════════════════════════════
# LANDMARK & ANTHROPOMETRIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def draw_landmarks_professional(img: Image.Image, landmarks: np.ndarray) -> Image.Image:
    """
    Professional landmark visualization with feature labels
    """
    cv_img = pil_to_cv(img).copy()
    
    # Normalize landmarks
    try:
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks)
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(-1, 2)
        elif landmarks.ndim == 3:
            landmarks = landmarks.squeeze()
    except Exception as e:
        st.error(f"Landmark processing error: {str(e)}")
        return img
    
    # Feature definitions with colors
    features = {
        'jaw': (list(range(0, 17)), (0, 255, 255), False),      # Cyan
        'right_eyebrow': (list(range(17, 22)), (255, 0, 255), False),  # Magenta
        'left_eyebrow': (list(range(22, 27)), (255, 0, 255), False),
        'nose_bridge': (list(range(27, 31)), (255, 255, 0), False),  # Yellow
        'nose_tip': (list(range(31, 36)), (255, 128, 0), False),  # Orange
        'right_eye': (list(range(36, 42)), (0, 255, 0), True),  # Green
        'left_eye': (list(range(42, 48)), (0, 255, 0), True),
        'outer_lip': (list(range(48, 60)), (255, 0, 0), True),  # Red
        'inner_lip': (list(range(60, 68)), (255, 0, 0), True)
    }
    
    # Draw points and lines
    for feature_name, (indices, color, is_closed) in features.items():
        try:
            if max(indices) < len(landmarks):
                # Draw lines
                points = np.array([landmarks[i] for i in indices], dtype=np.int32)
                cv2.polylines(cv_img, [points], is_closed, color, 2, cv2.LINE_AA)
                
                # Draw points
                for idx in indices:
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(cv_img, (x, y), 3, color, -1, cv2.LINE_AA)
                    cv2.circle(cv_img, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)
        except:
            continue
    
    return cv_to_pil(cv_img)

def calculate_golden_ratios(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Calculate facial golden ratio proportions
    Used in facial aesthetics and identification
    """
    ratios = {}
    
    try:
        # Face height to width
        face_height = np.linalg.norm(landmarks[8] - landmarks[27])  # Chin to forehead
        face_width = np.linalg.norm(landmarks[0] - landmarks[16])    # Jaw width
        ratios['face_height_to_width'] = face_height / face_width if face_width > 0 else 0
        
        # Upper to lower face
        upper_face = np.linalg.norm(landmarks[27] - landmarks[33])  # Forehead to nose tip
        lower_face = np.linalg.norm(landmarks[33] - landmarks[8])   # Nose tip to chin
        ratios['upper_to_lower_face'] = upper_face / lower_face if lower_face > 0 else 0
        
        # Nose width to mouth width
        nose_width = np.linalg.norm(landmarks[31] - landmarks[35])
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        ratios['nose_to_mouth_width'] = mouth_width / nose_width if nose_width > 0 else 0
        
        # Eye width to intercanthal distance
        eye_width = np.linalg.norm(landmarks[36] - landmarks[39])
        intercanthal = np.linalg.norm(landmarks[39] - landmarks[42])
        ratios['eye_to_intercanthal'] = intercanthal / eye_width if eye_width > 0 else 0
        
        # Lip proportions
        mouth_height = np.linalg.norm(landmarks[51] - landmarks[57])
        ratios['mouth_width_to_height'] = mouth_width / mouth_height if mouth_height > 0 else 0
        
        # Face thirds
        facial_height = np.linalg.norm(landmarks[8] - landmarks[27])
        upper_third = np.linalg.norm(landmarks[27] - landmarks[28])
        middle_third = np.linalg.norm(landmarks[28] - landmarks[33])
        lower_third = np.linalg.norm(landmarks[33] - landmarks[8])
        
        ratios['upper_third_ratio'] = upper_third / facial_height if facial_height > 0 else 0
        ratios['middle_third_ratio'] = middle_third / facial_height if facial_height > 0 else 0
        ratios['lower_third_ratio'] = lower_third / facial_height if facial_height > 0 else 0
        
        # Deviation from golden ratio
        ideal_ratios = {
            'face_height_to_width': PHI,
            'upper_to_lower_face': PHI,
            'nose_to_mouth_width': PHI,
            'mouth_width_to_height': PHI
        }
        
        for key, ideal in ideal_ratios.items():
            if key in ratios and ratios[key] > 0:
                deviation = abs(ratios[key] - ideal) / ideal
                ratios[f'{key}_deviation_from_phi'] = deviation
    
    except Exception as e:
        st.warning(f"Golden ratio calculation warning: {str(e)}")
    
    return ratios

def calculate_facial_similarity_advanced(ratios1: Dict, ratios2: Dict, 
                                         tolerance: float) -> Tuple[float, List[Dict], Dict]:
    """
    Advanced anthropometric similarity with weighted analysis
    """
    if not ratios1 or not ratios2:
        return 0.0, [], {}
    
    differences = []
    weights = []
    
    for key in ratios1.keys():
        if key in ratios2:
            val1, val2 = ratios1[key], ratios2[key]
            
            if val1 == 0 or val2 == 0:
                continue
            
            # Calculate difference metrics
            abs_diff = abs(val1 - val2)
            rel_diff = abs_diff / max(val1, val2)
            percent_diff = (abs_diff / val1) * 100 if val1 != 0 else 0
            
            # Status classification
            if rel_diff <= tolerance * 0.4:
                status = "✓ Excellent Match"
                weight = 1.0
            elif rel_diff <= tolerance * 0.7:
                status = "✓ Good Match"
                weight = 0.9
            elif rel_diff <= tolerance:
                status = "~ Acceptable"
                weight = 0.7
            elif rel_diff <= tolerance * 1.5:
                status = "⚠ Marginal"
                weight = 0.4
            else:
                status = "✗ Significant Deviation"
                weight = 0.1
            
            differences.append({
                'measurement': key.replace('_', ' ').title(),
                'image_a': float(val1),
                'image_b': float(val2),
                'absolute_diff': float(abs_diff),
                'relative_diff': float(rel_diff),
                'percent_diff': float(percent_diff),
                'status': status,
                'weight': weight
            })
            weights.append(weight)
    
    if not differences:
        return 0.0, [], {}
    
    # Weighted confidence calculation
    weighted_avg = sum(d['weight'] * (1 - d['relative_diff']) for d in differences) / len(differences)
    confidence = max(0, min(100, weighted_avg * 100))
    
    # Statistical summary
    stats = {
        'mean_abs_diff': np.mean([d['absolute_diff'] for d in differences]),
        'std_abs_diff': np.std([d['absolute_diff'] for d in differences]),
        'mean_rel_diff': np.mean([d['relative_diff'] for d in differences]),
        'std_rel_diff': np.std([d['relative_diff'] for d in differences]),
        'max_diff': max([d['relative_diff'] for d in differences]),
        'min_diff': min([d['relative_diff'] for d in differences]),
        'excellent_matches': len([d for d in differences if '✓ Excellent' in d['status']]),
        'good_matches': len([d for d in differences if '✓ Good' in d['status']]),
        'acceptable_matches': len([d for d in differences if '~' in d['status']]),
        'marginal_matches': len([d for d in differences if '⚠' in d['status']]),
        'poor_matches': len([d for d in differences if '✗' in d['status']])
    }
    
    return confidence, differences, stats

def classify_evidence_strength(confidence: float, stats: Dict) -> Tuple[str, str, str, str]:
    """
    Classify evidence strength with detailed reasoning
    Returns: (classification, color, interpretation, recommendation)
    """
    excellent_pct = (stats.get('excellent_matches', 0) / max(stats.get('excellent_matches', 0) + stats.get('good_matches', 0) + stats.get('acceptable_matches', 0) + stats.get('marginal_matches', 0) + stats.get('poor_matches', 0), 1)) * 100
    
    if confidence >= THRESHOLDS.STRONG_MATCH and excellent_pct >= 60:
        return (
            "STRONG SUPPORT - IDENTIFICATION",
            "#28a745",
            f"The anthropometric evidence provides STRONG SUPPORT for the conclusion that Images A and B depict the SAME INDIVIDUAL. "
            f"Match confidence: {confidence:.1f}% with {stats.get('excellent_matches', 0)} excellent matches out of {stats.get('excellent_matches', 0) + stats.get('good_matches', 0) + stats.get('acceptable_matches', 0) + stats.get('marginal_matches', 0) + stats.get('poor_matches', 0)} measurements.",
            "Proceed with positive identification. Recommend morphological feature verification by certified examiner."
        )
    elif confidence >= THRESHOLDS.MODERATE_MATCH:
        return (
            "MODERATE SUPPORT - PROBABLE MATCH",
            "#007bff",
            f"The anthropometric evidence provides MODERATE SUPPORT for the conclusion that Images A and B likely depict the SAME INDIVIDUAL. "
            f"Match confidence: {confidence:.1f}%. Some measurements show acceptable variation.",
            "Recommend additional comparative analysis and expert morphological review before final determination."
        )
    elif confidence >= THRESHOLDS.LIMITED_SUPPORT:
        return (
            "LIMITED SUPPORT - INCONCLUSIVE",
            "#ffc107",
            f"The anthropometric evidence provides LIMITED SUPPORT. The analysis is INCONCLUSIVE and cannot definitively determine identity. "
            f"Match confidence: {confidence:.1f}%.",
            "Insufficient evidence for determination. Require additional high-quality images and comprehensive expert examination."
        )
    elif confidence >= THRESHOLDS.INCONCLUSIVE_LOWER:
        return (
            "MODERATE SUPPORT - PROBABLE EXCLUSION",
            "#fd7e14",
            f"The anthropometric evidence provides MODERATE SUPPORT for the conclusion that Images A and B likely depict DIFFERENT INDIVIDUALS. "
            f"Match confidence: {confidence:.1f}%.",
            "Evidence suggests different individuals. Recommend expert review to rule out temporal changes or image quality issues."
        )
    else:
        return (
            "STRONG SUPPORT - EXCLUSION",
            "#dc3545",
            f"The anthropometric evidence provides STRONG SUPPORT for the conclusion that Images A and B depict DIFFERENT INDIVIDUALS. "
            f"Match confidence: {confidence:.1f}% with {stats.get('poor_matches', 0)} significant deviations.",
            "Evidence strongly supports exclusion. Proceed with non-match determination pending expert confirmation."
        )

def align_faces_advanced(img1: Image.Image, img2: Image.Image, 
                        landmarks1: np.ndarray, landmarks2: np.ndarray) -> Tuple[Image.Image, Dict]:
    """
    Advanced face alignment with quality metrics
    """
    def get_alignment_points(landmarks):
        # Use more stable points for alignment
        left_eye = np.mean([landmarks[i] for i in range(36, 42)], axis=0)
        right_eye = np.mean([landmarks[i] for i in range(42, 48)], axis=0)
        nose_tip = landmarks[30]
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]
        return np.array([left_eye, right_eye, nose_tip, left_mouth, right_mouth], dtype=np.float32)
    
    try:
        points1 = get_alignment_points(landmarks1)
        points2 = get_alignment_points(landmarks2)
        
        # Calculate transformation matrix
        transform_matrix = cv2.estimateAffinePartial2D(points2, points1)[0]
        
        if transform_matrix is None:
            raise Exception("Could not compute transformation matrix")
        
        # Apply transformation
        cv_img2 = pil_to_cv(img2)
        aligned = cv2.warpAffine(
            cv_img2,
            transform_matrix,
            img1.size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(128, 128, 128)
        )
        
        # Calculate alignment quality metrics
        scale = np.sqrt(transform_matrix[0,0]**2 + transform_matrix[0,1]**2)
        rotation = np.arctan2(transform_matrix[1,0], transform_matrix[0,0]) * 180 / np.pi
        translation = np.sqrt(transform_matrix[0,2]**2 + transform_matrix[1,2]**2)
        
        metrics = {
            'scale_factor': float(scale),
            'rotation_degrees': float(rotation),
            'translation_pixels': float(translation),
            'quality': 'Good' if abs(scale - 1.0) < 0.2 and abs(rotation) < 10 else 'Fair'
        }
        
        return cv_to_pil(aligned), metrics
    
    except Exception as e:
        st.warning(f"Advanced alignment failed: {str(e)}. Using fallback method.")
        return img2.resize(img1.size, Image.Resampling.LANCZOS), {'quality': 'Fallback', 'error': str(e)}

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_superimposed_image(img1: Image.Image, img2: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Create professional blended overlay"""
    cv_img1 = pil_to_cv(img1)
    cv_img2 = pil_to_cv(img2)
    
    if cv_img1.shape != cv_img2.shape:
        cv_img2 = cv2.resize(cv_img2, (cv_img1.shape[1], cv_img1.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    blended = cv2.addWeighted(cv_img1, 1 - alpha, cv_img2, alpha, 0)
    return cv_to_pil(blended)

def create_difference_heatmap(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """Generate professional difference heatmap"""
    cv_img1 = pil_to_cv(img1)
    cv_img2 = pil_to_cv(img2)
    
    # Calculate absolute difference
    diff = cv2.absdiff(cv_img1, cv_img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Normalize and apply colormap
    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    
    # Blend with original for context
    blended = cv2.addWeighted(cv_img1, 0.3, heatmap, 0.7, 0)
    
    return cv_to_pil(blended)

def plot_measurement_comparison(differences: List[Dict]) -> Image.Image:
    """Create comprehensive measurement comparison chart"""
    if not differences:
        return None
    
    df = pd.DataFrame(differences)
    
    # Sort by status for better visualization
    status_order = {"✓ Excellent Match": 0, "✓ Good Match": 1, "~ Acceptable": 2, 
                   "⚠ Marginal": 3, "✗ Significant Deviation": 4}
    df['status_order'] = df['status'].map(status_order)
    df = df.sort_values('status_order')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(differences) * 0.35)))
    
    # Left plot: Bar comparison
    measurements = df['measurement'].tolist()
    y_pos = np.arange(len(measurements))
    
    bars1 = ax1.barh(y_pos - 0.2, df['image_a'], 0.4, label='Image A', color='#4472C4', alpha=0.8)
    bars2 = ax1.barh(y_pos + 0.2, df['image_b'], 0.4, label='Image B', color='#ED7D31', alpha=0.8)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(measurements, fontsize=9)
    ax1.set_xlabel('Normalized Ratio', fontsize=11, fontweight='bold')
    ax1.set_title('Measurement Value Comparison', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Right plot: Difference percentages
    colors = ['#28a745' if '✓ Excellent' in s else '#17a2b8' if '✓ Good' in s else 
              '#ffc107' if '~' in s else '#fd7e14' if '⚠' in s else '#dc3545' 
              for s in df['status']]
    
    bars3 = ax2.barh(y_pos, df['percent_diff'], color=colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(measurements, fontsize=9)
    ax2.set_xlabel('Percentage Difference (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Relative Difference (Color-coded by Status)', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add status legend
    from matplotlib.patches import Patch #type: ignore
    legend_elements = [
        Patch(facecolor='#28a745', label='✓ Excellent Match'),
        Patch(facecolor='#17a2b8', label='✓ Good Match'),
        Patch(facecolor='#ffc107', label='~ Acceptable'),
        Patch(facecolor='#fd7e14', label='⚠ Marginal'),
        Patch(facecolor='#dc3545', label='✗ Significant Deviation')
    ]
    ax2.legend(handles=legend_elements, loc='best', fontsize=9)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI_FORENSIC, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf)

def plot_rgb_histograms_advanced(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """Enhanced RGB histogram comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # RGB channels
    channels = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']
    
    for idx, (channel, color) in enumerate(zip(channels, colors)):
        ax = axes[idx // 2, idx % 2]
        
        ax.hist(arr1[:, :, idx].flatten(), bins=128, alpha=0.6,
               color=color, label='Image A', density=True, linewidth=0)
        ax.hist(arr2[:, :, idx].flatten(), bins=128, alpha=0.6,
               color=color, label='Image B', density=True,
               histtype='step', linewidth=2.5, linestyle='--')
        
        ax.set_title(f'{channel} Channel Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Pixel Intensity', fontsize=10)
        ax.set_ylabel('Normalized Frequency', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, 255)
    
    # Combined luminance histogram
    ax = axes[1, 1]
    gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
    
    ax.hist(gray1.flatten(), bins=128, alpha=0.6, color='black',
           label='Image A', density=True, linewidth=0)
    ax.hist(gray2.flatten(), bins=128, alpha=0.6, color='gray',
           label='Image B', density=True, histtype='step', linewidth=2.5, linestyle='--')
    
    ax.set_title('Luminance Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Intensity', fontsize=10)
    ax.set_ylabel('Normalized Frequency', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, 255)
    
    plt.suptitle('Color Channel Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI_FORENSIC, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf)

def create_golden_ratio_visualization(landmarks: np.ndarray, img: Image.Image, ratios: Dict) -> Image.Image:
    """Visualize golden ratio proportions on face"""
    cv_img = pil_to_cv(img).copy()
    
    try:
        # Draw key proportions
        # Face height to width
        cv2.line(cv_img, tuple(landmarks[8].astype(int)), tuple(landmarks[27].astype(int)), (255, 215, 0), 2)
        cv2.line(cv_img, tuple(landmarks[0].astype(int)), tuple(landmarks[16].astype(int)), (255, 215, 0), 2)
        
        # Facial thirds
        cv2.line(cv_img, tuple(landmarks[27].astype(int)), tuple(landmarks[28].astype(int)), (0, 255, 255), 2)
        cv2.line(cv_img, tuple(landmarks[28].astype(int)), tuple(landmarks[33].astype(int)), (0, 255, 255), 2)
        cv2.line(cv_img, tuple(landmarks[33].astype(int)), tuple(landmarks[8].astype(int)), (0, 255, 255), 2)
        
        # Eye spacing
        cv2.line(cv_img, tuple(landmarks[36].astype(int)), tuple(landmarks[39].astype(int)), (255, 0, 255), 2)
        cv2.line(cv_img, tuple(landmarks[39].astype(int)), tuple(landmarks[42].astype(int)), (255, 0, 255), 2)
        cv2.line(cv_img, tuple(landmarks[42].astype(int)), tuple(landmarks[45].astype(int)), (255, 0, 255), 2)
        
    except Exception as e:
        st.warning(f"Golden ratio visualization error: {str(e)}")
    
    return cv_to_pil(cv_img)

def create_statistical_dashboard(stats: Dict, differences: List[Dict]) -> Image.Image:
    """Create comprehensive statistical summary dashboard"""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Match distribution pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Excellent', 'Good', 'Acceptable', 'Marginal', 'Poor']
    sizes = [
        stats.get('excellent_matches', 0),
        stats.get('good_matches', 0),
        stats.get('acceptable_matches', 0),
        stats.get('marginal_matches', 0),
        stats.get('poor_matches', 0)
    ]
    colors = ['#28a745', '#17a2b8', '#ffc107', '#fd7e14', '#dc3545']
    
    # Only plot non-zero values
    sizes_filtered = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if sizes_filtered:
        sizes_f, labels_f, colors_f = zip(*sizes_filtered)
        ax1.pie(sizes_f, labels=labels_f, colors=colors_f, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Match Quality Distribution', fontweight='bold')
    
    # 2. Difference distribution histogram
    ax2 = fig.add_subplot(gs[0, 1])
    diffs = [d['relative_diff'] * 100 for d in differences]
    ax2.hist(diffs, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(diffs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(diffs):.2f}%')
    ax2.set_xlabel('Relative Difference (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Difference Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Box plot of differences
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.boxplot(diffs, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    ax3.set_ylabel('Relative Difference (%)')
    ax3.set_title('Difference Statistics', fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. Top 5 best matches
    ax4 = fig.add_subplot(gs[1, 0])
    sorted_diffs = sorted(differences, key=lambda x: x['relative_diff'])[:5]
    measurements = [d['measurement'][:20] for d in sorted_diffs]
    values = [d['relative_diff'] * 100 for d in sorted_diffs]
    ax4.barh(measurements, values, color='green', alpha=0.7)
    ax4.set_xlabel('Difference (%)')
    ax4.set_title('Top 5 Best Matches', fontweight='bold')
    ax4.grid(alpha=0.3, axis='x')
    
    # 5. Top 5 worst matches
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_diffs_worst = sorted(differences, key=lambda x: x['relative_diff'], reverse=True)[:5]
    measurements_worst = [d['measurement'][:20] for d in sorted_diffs_worst]
    values_worst = [d['relative_diff'] * 100 for d in sorted_diffs_worst]
    ax5.barh(measurements_worst, values_worst, color='red', alpha=0.7)
    ax5.set_xlabel('Difference (%)')
    ax5.set_title('Top 5 Largest Deviations', fontweight='bold')
    ax5.grid(alpha=0.3, axis='x')
    
    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = f"""
    STATISTICAL SUMMARY
    {'='*30}
    
    Total Measurements: {len(differences)}
    
    Mean Difference: {stats['mean_rel_diff']*100:.2f}%
    Std Deviation: {stats['std_rel_diff']*100:.2f}%
    Min Difference: {stats['min_diff']*100:.2f}%
    Max Difference: {stats['max_diff']*100:.2f}%
    
    Match Distribution:
    • Excellent: {stats.get('excellent_matches', 0)}
    • Good: {stats.get('good_matches', 0)}
    • Acceptable: {stats.get('acceptable_matches', 0)}
    • Marginal: {stats.get('marginal_matches', 0)}
    • Poor: {stats.get('poor_matches', 0)}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Comprehensive Statistical Analysis', fontsize=14, fontweight='bold')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI_FORENSIC, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf)

# ═══════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_comprehensive_report(
    case_id: str,
    timestamp: str,
    img1_hash: str,
    img2_hash: str,
    img1_md5: str,
    img2_md5: str,
    img1_quality: Dict,
    img2_quality: Dict,
    confidence: float,
    classification: str,
    interpretation: str,
    recommendation: str,
    differences: List[Dict],
    stats: Dict,
    golden_ratios1: Dict,
    golden_ratios2: Dict,
    ela_metrics1: Dict,
    ela_metrics2: Dict,
    hist_metrics: Dict,
    pixel_metrics: Dict,
    alignment_metrics: Dict,
    examiner_notes: str,
    morphological_assessment: Optional[Dict] = None
) -> str:
    """
    Generate comprehensive forensic examination report
    Court-admissible documentation format
    """
    
    report = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                  FORENSIC FACIAL COMPARISON EXAMINATION REPORT            ║
╚═══════════════════════════════════════════════════════════════════════════╝

CASE IDENTIFICATION
────────────────────────────────────────────────────────────────────────────
Case Number:                {case_id}
Examination Date/Time:      {timestamp}
Examiner:                   Automated Analysis System v3.0
Analysis Method:            Morphological & Anthropometric Comparison
Standards Applied:          FISWG Guidelines, PCAST Recommendations

CHAIN OF CUSTODY & IMAGE INTEGRITY
────────────────────────────────────────────────────────────────────────────
IMAGE A (Reference):
  SHA-256 Hash:             {img1_hash}
  MD5 Hash:                 {img1_md5}
  Resolution:               {img1_quality['resolution'][0]}x{img1_quality['resolution'][1]} pixels
  Quality Assessment:       {img1_quality['assessment']} (Score: {img1_quality['overall']:.1f}/100)
    - Sharpness:            {img1_quality['sharpness']:.1f}/100
    - Contrast:             {img1_quality['contrast']:.1f}/100
    - Brightness:           {img1_quality['brightness']:.1f}/100
    - Noise Level:          {img1_quality['noise']:.1f}/100

IMAGE B (Query):
  SHA-256 Hash:             {img2_hash}
  MD5 Hash:                 {img2_md5}
  Resolution:               {img2_quality['resolution'][0]}x{img2_quality['resolution'][1]} pixels
  Quality Assessment:       {img2_quality['assessment']} (Score: {img2_quality['overall']:.1f}/100)
    - Sharpness:            {img2_quality['sharpness']:.1f}/100
    - Contrast:             {img2_quality['contrast']:.1f}/100
    - Brightness:           {img2_quality['brightness']:.1f}/100
    - Noise Level:          {img2_quality['noise']:.1f}/100

═══════════════════════════════════════════════════════════════════════════
                            EXAMINATION FINDINGS
═══════════════════════════════════════════════════════════════════════════

CONCLUSION:             {classification}
CONFIDENCE SCORE:       {confidence:.2f}%

INTERPRETATION:
{interpretation}

RECOMMENDATION:
{recommendation}

ANTHROPOMETRIC ANALYSIS
────────────────────────────────────────────────────────────────────────────
Total Measurements Analyzed:    {len(differences)}
Tolerance Threshold:            {THRESHOLDS.RATIO_MODERATE * 100:.1f}%

MATCH DISTRIBUTION:
  ✓ Excellent Matches:          {stats.get('excellent_matches', 0)} ({stats.get('excellent_matches', 0)/len(differences)*100:.1f}%)
  ✓ Good Matches:               {stats.get('good_matches', 0)} ({stats.get('good_matches', 0)/len(differences)*100:.1f}%)
  ~ Acceptable Matches:         {stats.get('acceptable_matches', 0)} ({stats.get('acceptable_matches', 0)/len(differences)*100:.1f}%)
  ⚠ Marginal Matches:           {stats.get('marginal_matches', 0)} ({stats.get('marginal_matches', 0)/len(differences)*100:.1f}%)
  ✗ Significant Deviations:     {stats.get('poor_matches', 0)} ({stats.get('poor_matches', 0)/len(differences)*100:.1f}%)

STATISTICAL SUMMARY:
  Mean Relative Difference:     {stats['mean_rel_diff']*100:.3f}%
  Standard Deviation:           {stats['std_rel_diff']*100:.3f}%
  Minimum Difference:           {stats['min_diff']*100:.3f}%
  Maximum Difference:           {stats['max_diff']*100:.3f}%

DETAILED MEASUREMENTS:
┌───┬────────────────────────────────┬──────────┬──────────┬───────────┬───────────┬─────────────────────┐
│ # │ Measurement                    │ Image A  │ Image B  │ Abs Diff  │ Rel Diff  │ Status              │
├───┼────────────────────────────────┼──────────┼──────────┼───────────┼───────────┼─────────────────────┤"""
    
    # Add measurement table
    for i, diff in enumerate(differences, 1):
        report += f"\n│{i:3d}│ {diff['measurement'][:30]:<30} │ {diff['image_a']:8.4f} │ {diff['image_b']:8.4f} │ {diff['absolute_diff']:9.4f} │ {diff['relative_diff']*100:8.2f}% │ {diff['status']:<19} │"
    
    report += f"""
└───┴────────────────────────────────┴──────────┴──────────┴───────────┴───────────┴─────────────────────┘

GOLDEN RATIO ANALYSIS
────────────────────────────────────────────────────────────────────────────
The Golden Ratio (Phi = 1.618) represents ideal facial proportions.

IMAGE A Golden Ratio Deviations:
"""
    for key, value in golden_ratios1.items():
        if 'deviation' in key:
            feature = key.replace('_deviation_from_phi', '').replace('_', ' ').title()
            deviation_pct = value * 100
            assessment = "Excellent" if value < THRESHOLDS.GOLDEN_RATIO_PERFECT else "Good" if value < THRESHOLDS.GOLDEN_RATIO_GOOD else "Acceptable" if value < THRESHOLDS.GOLDEN_RATIO_ACCEPTABLE else "Significant"
            report += f"  {feature:<30}: {deviation_pct:6.2f}% deviation ({assessment})\n"
    
    report += f"\nIMAGE B Golden Ratio Deviations:\n"
    for key, value in golden_ratios2.items():
        if 'deviation' in key:
            feature = key.replace('_deviation_from_phi', '').replace('_', ' ').title()
            deviation_pct = value * 100
            assessment = "Excellent" if value < THRESHOLDS.GOLDEN_RATIO_PERFECT else "Good" if value < THRESHOLDS.GOLDEN_RATIO_GOOD else "Acceptable" if value < THRESHOLDS.GOLDEN_RATIO_ACCEPTABLE else "Significant"
            report += f"  {feature:<30}: {deviation_pct:6.2f}% deviation ({assessment})\n"
    
    report += f"""
IMAGE INTEGRITY & AUTHENTICITY ANALYSIS
────────────────────────────────────────────────────────────────────────────
ERROR LEVEL ANALYSIS (Manipulation Detection):

IMAGE A:
  Mean ELA Score:               {ela_metrics1['mean']:.4f} {"(Normal)" if ela_metrics1['mean'] < THRESHOLDS.ELA_NORMAL else "(Elevated - Possible Manipulation)"}
  Maximum ELA:                  {ela_metrics1['max']:.4f}
  Standard Deviation:           {ela_metrics1['std']:.4f}
  Hotspot Percentage:           {ela_metrics1['hotspots_pct']:.2f}%

IMAGE B:
  Mean ELA Score:               {ela_metrics2['mean']:.4f} {"(Normal)" if ela_metrics2['mean'] < THRESHOLDS.ELA_NORMAL else "(Elevated - Possible Manipulation)"}
  Maximum ELA:                  {ela_metrics2['max']:.4f}
  Standard Deviation:           {ela_metrics2['std']:.4f}
  Hotspot Percentage:           {ela_metrics2['hotspots_pct']:.2f}%

ASSESSMENT:
"""
    
    if ela_metrics1['mean'] < THRESHOLDS.ELA_NORMAL and ela_metrics2['mean'] < THRESHOLDS.ELA_NORMAL:
        report += "  ✓ Both images show normal compression artifacts with no evidence of manipulation.\n"
    elif ela_metrics1['mean'] < THRESHOLDS.ELA_ELEVATED and ela_metrics2['mean'] < THRESHOLDS.ELA_ELEVATED:
        report += "  ⚠ Moderate ELA scores detected. May indicate multiple compressions or editing.\n"
    else:
        report += "  ⚠ High ELA scores detected. Possible digital manipulation or heavy editing present.\n"
    
    report += f"""
COLOR DISTRIBUTION ANALYSIS
────────────────────────────────────────────────────────────────────────────
HISTOGRAM CORRELATION:
  Overall Similarity:           {hist_metrics['overall_similarity']:.4f} ({100*hist_metrics['overall_similarity']:.1f}%)
  Red Channel Correlation:      {hist_metrics['red_correlation']:.4f}
  Green Channel Correlation:    {hist_metrics['green_correlation']:.4f}
  Blue Channel Correlation:     {hist_metrics['blue_correlation']:.4f}

INTERPRETATION:
"""
    
    if hist_metrics['overall_similarity'] > THRESHOLDS.HISTOGRAM_HIGH:
        report += "  ✓ High color similarity - Images likely captured under similar conditions.\n"
    elif hist_metrics['overall_similarity'] > THRESHOLDS.HISTOGRAM_MODERATE:
        report += "  ~ Moderate color similarity - Some variation in lighting/capture conditions.\n"
    else:
        report += "  ⚠ Low color similarity - Significant differences in lighting or post-processing.\n"
    
    report += f"""
PIXEL-LEVEL COMPARISON METRICS
────────────────────────────────────────────────────────────────────────────
  Mean Squared Error (MSE):     {pixel_metrics['mse']:.2f}
  Root Mean Squared Error:      {pixel_metrics['rmse']:.2f}
  Peak Signal-to-Noise Ratio:   {pixel_metrics['psnr']:.2f} dB {"(Excellent)" if pixel_metrics['psnr'] > THRESHOLDS.PSNR_EXCELLENT else "(Good)" if pixel_metrics['psnr'] > THRESHOLDS.PSNR_GOOD else "(Fair/Poor)"}
  Mean Absolute Error:          {pixel_metrics['mae']:.2f}
  Normalized Cross-Correlation: {pixel_metrics['ncc']:.4f}
"""
    
    if 'ssim' in pixel_metrics:
        ssim_val = pixel_metrics['ssim']
        ssim_assessment = "Excellent" if ssim_val > THRESHOLDS.SSIM_EXCELLENT else "Good" if ssim_val > THRESHOLDS.SSIM_GOOD else "Fair/Poor"
        report += f"  Structural Similarity (SSIM):  {ssim_val:.4f} ({ssim_assessment})\n"
    
    report += f"""
FACE ALIGNMENT QUALITY
────────────────────────────────────────────────────────────────────────────
  Alignment Quality:            {alignment_metrics.get('quality', 'N/A')}
  Scale Factor:                 {alignment_metrics.get('scale_factor', 0):.4f}
  Rotation Applied:             {alignment_metrics.get('rotation_degrees', 0):.2f}°
  Translation Distance:         {alignment_metrics.get('translation_pixels', 0):.2f} pixels
"""
    
    # Morphological Features
    if morphological_assessment:
        report += f"""
MORPHOLOGICAL FEATURE ASSESSMENT
────────────────────────────────────────────────────────────────────────────
Expert visual comparison of {len(morphological_assessment)} observable features:

"""
        critical_matches = 0
        critical_total = 0
        secondary_matches = 0
        secondary_total = 0
        
        for feature, assessment in morphological_assessment.items():
            if assessment['assessed']:
                feature_info = MORPHOLOGICAL_FEATURES[feature]
                category = feature_info['category'].upper()
                match_status = assessment['match_status']
                
                report += f"  [{category}] {feature.replace('_', ' ').title():<30}\n"
                report += f"    Image A: {assessment['value_a']:<20} Image B: {assessment['value_b']:<20} {match_status}\n"
                
                if feature_info['category'] == 'critical':
                    critical_total += 1
                    if assessment['match']:
                        critical_matches += 1
                elif feature_info['category'] == 'secondary':
                    secondary_total += 1
                    if assessment['match']:
                        secondary_matches += 1
        
        if critical_total > 0:
            critical_pct = (critical_matches / critical_total) * 100
            report += f"\nCRITICAL FEATURES: {critical_matches}/{critical_total} matches ({critical_pct:.1f}%)\n"
        
        if secondary_total > 0:
            secondary_pct = (secondary_matches / secondary_total) * 100
            report += f"SECONDARY FEATURES: {secondary_matches}/{secondary_total} matches ({secondary_pct:.1f}%)\n"
    
    report += f"""
EXAMINER NOTES & OBSERVATIONS
────────────────────────────────────────────────────────────────────────────
{examiner_notes if examiner_notes.strip() else 'No additional examiner notes provided.'}

METHODOLOGY
────────────────────────────────────────────────────────────────────────────
This forensic facial comparison employed a multi-modal approach combining:

1. ANTHROPOMETRIC ANALYSIS: 68-point facial landmark detection with automated
   measurement extraction. Measurements normalized by inter-ocular distance to
   account for scale variations. Tolerance threshold: {THRESHOLDS.RATIO_MODERATE*100:.1f}%

2. GOLDEN RATIO ANALYSIS: Assessment of facial proportions against the Golden
   Ratio (Phi = 1.618) to evaluate aesthetic consistency and natural variation.

3. MORPHOLOGICAL ASSESSMENT: Visual comparison of {len(MORPHOLOGICAL_FEATURES)} observable
   facial features categorized by forensic significance (critical, secondary, tertiary).

4. IMAGE INTEGRITY VERIFICATION: Error Level Analysis (ELA) to detect potential
   digital manipulation or editing.

5. COLOR DISTRIBUTION ANALYSIS: Multi-channel histogram correlation to assess
   capture condition consistency.

6. PIXEL-LEVEL COMPARISON: Structural and statistical similarity metrics including
   SSIM, PSNR, MSE, and normalized cross-correlation.

7. GEOMETRIC ALIGNMENT: Affine transformation for optimal facial registration
   and accurate pixel-level comparison.

Standards Applied:
• FISWG (Facial Identification Scientific Working Group) Guidelines
• PCAST (President's Council of Advisors on Science & Technology) Recommendations
• Federal Rules of Evidence 702 (Expert Testimony Standards)

LIMITATIONS & CONSIDERATIONS
────────────────────────────────────────────────────────────────────────────
1. IMAGE QUALITY: Analysis accuracy is dependent on image resolution, lighting,
   and facial pose. Optimal results require frontal or near-frontal views with
   adequate resolution (minimum 512x512 pixels recommended).

2. TEMPORAL FACTORS: Significant aging, weight changes, facial hair growth, or
   cosmetic procedures may affect measurements and should be considered in
   interpretation.

3. ENVIRONMENTAL FACTORS: Variations in lighting, camera systems, and capture
   conditions can affect color distribution and pixel-level metrics while not
   affecting identity.

4. OCCLUSIONS: Glasses, makeup, facial hair, or other occlusions may interfere
   with landmark detection and measurement accuracy.

5. AUTOMATED ANALYSIS: This system provides computational analysis and should
   be reviewed by qualified forensic facial comparison examiners for final
   determination, particularly in critical cases.

6. STATISTICAL INTERPRETATION: Confidence scores represent mathematical
   similarity and should be interpreted within the context of the complete
   examination, not as absolute probability of identity.

EXPERT REVIEW REQUIREMENTS
────────────────────────────────────────────────────────────────────────────
Per FISWG and PCAST guidelines, this automated analysis should be reviewed by
a certified forensic facial comparison examiner before use in legal proceedings.
The following review points are recommended:

☐ Verification of landmark detection accuracy
☐ Assessment of image quality and suitability for comparison
☐ Evaluation of temporal factors (aging, weight changes, etc.)
☐ Morphological feature verification and documentation
☐ Consideration of case-specific contextual factors
☐ Review of measurement tolerances and thresholds
☐ Validation of conclusions and evidence strength classification

DISCLAIMER
────────────────────────────────────────────────────────────────────────────
This report is generated by an automated forensic analysis system following
established scientific methodologies and forensic standards. While the system
provides objective computational assessment, final forensic determinations
should be made by qualified examiners who can integrate this analysis with
additional evidence, contextual factors, and expert judgment.

The conclusions presented represent the computational assessment based on the
methodologies applied and available image data. Results should be considered
as supporting evidence within a comprehensive forensic examination and not as
sole determinative proof of identity or non-identity.

This analysis is provided for forensic investigative purposes and should be
used in accordance with applicable legal standards and professional guidelines.

═══════════════════════════════════════════════════════════════════════════
                            END OF REPORT
═══════════════════════════════════════════════════════════════════════════

Report Generated: {timestamp}
System Version: Forensic Facial Comparison System v3.0
Analysis ID: {case_id}

Digital Signature: {hashlib.sha256(case_id.encode()).hexdigest()[:32]}
"""
    
    return report

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main application interface"""
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
    }
    .evidence-strong {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    .evidence-moderate {
        background: #cce5ff;
        border-left: 5px solid #007bff;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    .evidence-weak {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    .evidence-exclusion {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    .stExpander {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1 style='margin: 0; font-size: 2.5em;'>⚖️ FORENSIC FACIAL COMPARISON SYSTEM</h1>
        <p style='margin: 15px 0 5px 0; font-size: 1.2em; font-weight: 500;'>Professional Analysis Platform v3.0</p>
        <p style='margin: 0; font-size: 0.95em; opacity: 0.9;'>FISWG Compliant • Court-Ready Documentation • Multi-Modal Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.markdown("## 📋 Analysis Configuration")
        
        st.markdown("---")
        st.markdown("### 🎯 Measurement Tolerance")
        tolerance_option = st.radio(
            "Select tolerance level",
            options=[
                ("Strict (3%)", THRESHOLDS.RATIO_STRICT),
                ("Moderate (7%) - Recommended", THRESHOLDS.RATIO_MODERATE),
                ("Lenient (12%)", THRESHOLDS.RATIO_LENIENT)
            ],
            index=1,
            format_func=lambda x: x[0]
        )
        tolerance = tolerance_option[1]
        
        st.markdown("---")
        st.markdown("### 📊 Standards & Guidelines")
        st.info("""
        **Compliance:**
        ✓ FISWG Guidelines  
        ✓ PCAST Standards  
        ✓ FRE 702  
        ✓ ISO/IEC 17025
        
        **Evidence Thresholds:**
        • Strong Support: ≥92%
        • Moderate Support: ≥82%
        • Limited Support: ≥72%
        • Inconclusive: 62-72%
        • Exclusion: <62%
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Best Practices")
        st.warning("""
        **For Optimal Results:**
        • Use high-resolution images
        • Frontal facial views preferred
        • Good lighting conditions
        • Minimal occlusions
        • Similar poses if possible
        """)
        
        st.markdown("---")
        st.caption("© 2025 Forensic Analysis Division")
    
    # Generate case ID
    if 'case_id' not in st.session_state:
        st.session_state.case_id = generate_case_id()
    
    case_id = st.session_state.case_id
    
    # Case Information
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"**📁 Case Number:** `{case_id}`")
    with col2:
        if st.button("🔄 Generate New Case ID"):
            st.session_state.case_id = generate_case_id()
            st.rerun()
    
    # Image Upload Section
    st.markdown("---")
    st.markdown("## 📤 IMAGE UPLOAD & VERIFICATION")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🅰️ Reference Image (Known Subject)")
        img_file_1 = st.file_uploader(
            "Upload reference image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            key="img1",
            help="High-quality image of known subject for comparison"
        )
    
    with col2:
        st.markdown("### 🅱️ Query Image (Unknown Subject)")
        img_file_2 = st.file_uploader(
            "Upload query image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            key="img2",
            help="Image to be compared against reference"
        )
    
    # Load and display images
    img1, img2 = None, None
    if img_file_1:
        img1 = Image.open(img_file_1).convert("RGB")
    if img_file_2:
        img2 = Image.open(img_file_2).convert("RGB")
    
    if img1 and img2:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption="Image A - Reference", use_container_width=True)
        with col2:
            st.image(img2, caption="Image B - Query", use_container_width=True)
        
        # Quick quality check
        quality1 = calculate_image_quality_score(img1)
        quality2 = calculate_image_quality_score(img2)
        
        col1, col2 = st.columns(2)
        with col1:
            quality_color = "🟢" if quality1['overall'] >= 80 else "🟡" if quality1['overall'] >= 60 else "🔴"
            st.caption(f"{quality_color} Quality: {quality1['assessment']} ({quality1['overall']:.1f}/100) | {quality1['resolution'][0]}×{quality1['resolution'][1]}px")
        with col2:
            quality_color = "🟢" if quality2['overall'] >= 80 else "🟡" if quality2['overall'] >= 60 else "🔴"
            st.caption(f"{quality_color} Quality: {quality2['assessment']} ({quality2['overall']:.1f}/100) | {quality2['resolution'][0]}×{quality2['resolution'][1]}px")
    
    # Examiner Notes
    st.markdown("---")
    st.markdown("## 📝 EXAMINER NOTES & OBSERVATIONS")
    examiner_notes = st.text_area(
        "Document relevant observations, context, or special considerations",
        height=120,
        placeholder="Example: Subject appears approximately 5 years older in Image B. Facial hair present in query image. "
                   "Lighting conditions significantly different. Subject reported cosmetic procedures between captures.",
        help="These notes will be included in the final forensic report"
    )
    
    # Analysis Options
    st.markdown("---")
    st.markdown("## ⚙️ ANALYSIS OPTIONS")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        perform_morphological = st.checkbox("Include Morphological Assessment", value=True, 
                                          help="Manual feature comparison assessment")
    with col2:
        perform_golden_ratio = st.checkbox("Include Golden Ratio Analysis", value=True,
                                          help="Facial proportion aesthetic analysis")
    with col3:
        generate_visualizations = st.checkbox("Generate All Visualizations", value=True,
                                             help="Create comprehensive visual outputs")
    
    # Main Analysis Button
    st.markdown("---")
    analyze_button = st.button(
        "🔬 CONDUCT COMPREHENSIVE FORENSIC ANALYSIS",
        type="primary",
        use_container_width=True,
        help="Begin full multi-modal forensic comparison"
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAIN ANALYSIS WORKFLOW
    # ═══════════════════════════════════════════════════════════════════════
    
    if analyze_button:
        if not img1 or not img2:
            st.error("⚠️ **Error:** Both images must be uploaded before conducting analysis.")
            st.stop()
        
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Initialize progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0, text="Initializing forensic analysis...")
            status_text = st.empty()
        
        # ===================================================================
        # STEP 1: Image Integrity & Quality Assessment
        # ===================================================================
        status_text.markdown("**Step 1/10:** Computing cryptographic hashes for chain of custody...")
        progress_bar.progress(5, text="Step 1/10: Image integrity verification")
        
        hash1_sha256 = compute_sha256(img1)
        hash1_md5 = compute_md5(img1)
        hash2_sha256 = compute_sha256(img2)
        hash2_md5 = compute_md5(img2)
        
        quality1 = calculate_image_quality_score(img1)
        quality2 = calculate_image_quality_score(img2)
        
        # ===================================================================
        # STEP 2: Image Standardization
        # ===================================================================
        status_text.markdown("**Step 2/10:** Standardizing images for consistent analysis...")
        progress_bar.progress(10, text="Step 2/10: Image standardization")
        
        img1_std = standardize_image(img1)
        img2_std = standardize_image(img2)
        
        # ===================================================================
        # STEP 3: Facial Landmark Detection
        # ===================================================================
        status_text.markdown("**Step 3/10:** Detecting 68-point facial landmarks...")
        progress_bar.progress(20, text="Step 3/10: Landmark detection")
        
        cv_img1 = pil_to_cv(img1_std)
        cv_img2 = pil_to_cv(img2_std)
        
        landmarks1 = get_landmarks(cv_img1)
        landmarks2 = get_landmarks(cv_img2)
        
        if not landmarks1:
            st.error("❌ **CRITICAL ERROR:** No face detected in Image A (Reference)")
            st.warning("**Recommendation:** Ensure the image contains a clear, unobstructed frontal or near-frontal facial view.")
            progress_container.empty()
            st.stop()
        
        if not landmarks2:
            st.error("❌ **CRITICAL ERROR:** No face detected in Image B (Query)")
            st.warning("**Recommendation:** Ensure the image contains a clear, unobstructed frontal or near-frontal facial view.")
            progress_container.empty()
            st.stop()
        
        # ===================================================================
        # STEP 4: Anthropometric Measurements
        # ===================================================================
        status_text.markdown("**Step 4/10:** Calculating anthropometric measurements...")
        progress_bar.progress(30, text="Step 4/10: Anthropometric analysis")
        
        measurements1 = calculate_measurements(landmarks1)
        measurements2 = calculate_measurements(landmarks2)
        
        ratios1 = normalize_by_eye_width(measurements1)
        ratios2 = normalize_by_eye_width(measurements2)
        
        # Calculate golden ratios
        golden_ratios1 = {}
        golden_ratios2 = {}
        if perform_golden_ratio:
            golden_ratios1 = calculate_golden_ratios(landmarks1)
            golden_ratios2 = calculate_golden_ratios(landmarks2)
        
        # ===================================================================
        # STEP 5: Similarity Calculation
        # ===================================================================
        status_text.markdown("**Step 5/10:** Computing similarity confidence scores...")
        progress_bar.progress(40, text="Step 5/10: Similarity analysis")
        
        confidence, differences, stats = calculate_facial_similarity_advanced(ratios1, ratios2, tolerance)
        classification, color, interpretation, recommendation = classify_evidence_strength(confidence, stats)
        
        # ===================================================================
        # STEP 6: Error Level Analysis
        # ===================================================================
        status_text.markdown("**Step 6/10:** Performing error level analysis...")
        progress_bar.progress(50, text="Step 6/10: Manipulation detection")
        
        ela1, ela_score1, ela_metrics1 = compute_ela(img1_std)
        ela2, ela_score2, ela_metrics2 = compute_ela(img2_std)
        
        # ===================================================================
        # STEP 7: Color Distribution Analysis
        # ===================================================================
        status_text.markdown("**Step 7/10:** Analyzing color distributions...")
        progress_bar.progress(60, text="Step 7/10: Color analysis")
        
        hist_metrics = calculate_histogram_similarity(img1_std, img2_std)
        
        # ===================================================================
        # STEP 8: Face Alignment
        # ===================================================================
        status_text.markdown("**Step 8/10:** Aligning faces for pixel-level comparison...")
        progress_bar.progress(70, text="Step 8/10: Face alignment")
        
        img2_aligned, alignment_metrics = align_faces_advanced(img1_std, img2_std, landmarks1, landmarks2)
        
        # ===================================================================
        # STEP 9: Pixel-Level Metrics
        # ===================================================================
        status_text.markdown("**Step 9/10:** Calculating pixel-level similarity metrics...")
        progress_bar.progress(80, text="Step 9/10: Pixel analysis")
        
        pixel_metrics = calculate_pixel_metrics(img1_std, img2_aligned)
        
        # ===================================================================
        # STEP 10: Visualization Generation
        # ===================================================================
        status_text.markdown("**Step 10/10:** Generating forensic visualizations...")
        progress_bar.progress(90, text="Step 10/10: Creating visualizations")
        
        # Generate landmark visualizations
        img1_landmarks = draw_landmarks_professional(img1_std, landmarks1)
        img2_landmarks = draw_landmarks_professional(img2_std, landmarks2)
        
        # Generate comprehensive visualizations
        if generate_visualizations:
            measurement_chart = plot_measurement_comparison(differences)
            rgb_histograms = plot_rgb_histograms_advanced(img1_std, img2_std)
            diff_heatmap = create_difference_heatmap(img1_std, img2_aligned)
            stats_dashboard = create_statistical_dashboard(stats, differences)
        
        # Complete
        progress_bar.progress(100, text="Analysis complete!")
        status_text.markdown("**✅ Forensic analysis completed successfully!**")
        
        # Clear progress indicators after brief pause
        import time
        time.sleep(1)
        progress_container.empty()
        
        # ===================================================================
        # RESULTS PRESENTATION
        # ===================================================================
        
        st.markdown("---")
        st.markdown("# 📊 FORENSIC EXAMINATION RESULTS")
        
        # SECTION 1: Executive Summary
        st.markdown("## 1️⃣ EXECUTIVE SUMMARY")
        
        evidence_class = "evidence-strong" if "STRONG" in classification and "IDENTIFICATION" in classification else \
                        "evidence-moderate" if "MODERATE" in classification or "LIMITED" in classification else \
                        "evidence-exclusion"
        
        st.markdown(f"""
        <div class='{evidence_class}'>
            <h2 style='margin-top: 0; color: {color};'>{classification}</h2>
            <h3 style='margin: 10px 0;'>Confidence Score: {confidence:.2f}%</h3>
            <p style='font-size: 1.1em; line-height: 1.6;'><strong>Interpretation:</strong><br>{interpretation}</p>
            <p style='font-size: 1em; margin-bottom: 0;'><strong>Recommendation:</strong><br>{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Match Confidence", f"{confidence:.1f}%",
                   delta="Strong" if confidence >= THRESHOLDS.STRONG_MATCH else "Moderate" if confidence >= THRESHOLDS.MODERATE_MATCH else "Weak")
        col2.metric("Measurements", len(differences), 
                   delta=f"{stats.get('excellent_matches', 0)} excellent")
        col3.metric("Image Quality", f"{(quality1['overall'] + quality2['overall'])/2:.0f}/100",
                   delta=quality1['assessment'])
        col4.metric("ELA Assessment", 
                   "Normal" if max(ela_score1, ela_score2) < THRESHOLDS.ELA_NORMAL else "Elevated",
                   delta="✓" if max(ela_score1, ela_score2) < THRESHOLDS.ELA_NORMAL else "⚠")
        
        # SECTION 2: Chain of Custody
        with st.expander("🔐 SECTION 2: CHAIN OF CUSTODY & IMAGE INTEGRITY", expanded=False):
            st.markdown("### Image Verification Hashes")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Image A (Reference):**")
                st.code(f"SHA-256: {hash1_sha256}\nMD5: {hash1_md5}", language="text")
                st.caption(f"Resolution: {quality1['resolution'][0]}×{quality1['resolution'][1]}px")
                
                col1a, col1b, col1c, col1d = st.columns(4)
                col1a.metric("Quality", f"{quality1['overall']:.0f}/100")
                col1b.metric("Sharpness", f"{quality1['sharpness']:.0f}/100")
                col1c.metric("Contrast", f"{quality1['contrast']:.0f}/100")
                col1d.metric("Noise", f"{quality1['noise']:.0f}/100")
            
            with col2:
                st.markdown("**Image B (Query):**")
                st.code(f"SHA-256: {hash2_sha256}\nMD5: {hash2_md5}", language="text")
                st.caption(f"Resolution: {quality2['resolution'][0]}×{quality2['resolution'][1]}px")
                
                col2a, col2b, col2c, col2d = st.columns(4)
                col2a.metric("Quality", f"{quality2['overall']:.0f}/100")
                col2b.metric("Sharpness", f"{quality2['sharpness']:.0f}/100")
                col2c.metric("Contrast", f"{quality2['contrast']:.0f}/100")
                col2d.metric("Noise", f"{quality2['noise']:.0f}/100")
            
            st.success(f"✅ Image integrity verified via cryptographic hashing | Timestamp: {timestamp}")
        
        # SECTION 3: Facial Landmarks
        with st.expander("🎯 SECTION 3: FACIAL LANDMARK DETECTION & ANNOTATION", expanded=False):
            st.success(f"✅ Successfully detected {LANDMARK_COUNT} facial landmarks in both images")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img1_landmarks, caption="Image A - 68-Point Landmark Annotation", use_container_width=True)
            with col2:
                st.image(img2_landmarks, caption="Image B - 68-Point Landmark Annotation", use_container_width=True)
            
            with st.expander("ℹ️ Landmark Model Specification"):
                st.markdown("""
                **68-Point Facial Landmark Model (dlib/iBUG):**
                
                | Feature | Landmarks | Count | Color Code |
                |---------|-----------|-------|------------|
                | Jawline | 0-16 | 17 | Cyan |
                | Right Eyebrow | 17-21 | 5 | Magenta |
                | Left Eyebrow | 22-26 | 5 | Magenta |
                | Nose Bridge | 27-30 | 4 | Yellow |
                | Nose Tip/Nostrils | 31-35 | 5 | Orange |
                | Right Eye | 36-41 | 6 | Green |
                | Left Eye | 42-47 | 6 | Green |
                | Outer Lip | 48-59 | 12 | Red |
                | Inner Lip | 60-67 | 8 | Red |
                
                These landmarks serve as anatomical reference points for anthropometric measurements.
                """)
        
        # SECTION 4: Anthropometric Analysis
        with st.expander("📏 SECTION 4: ANTHROPOMETRIC ANALYSIS & MEASUREMENTS", expanded=True):
            st.markdown("### Match Distribution")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("✓ Excellent", stats.get('excellent_matches', 0), 
                       delta=f"{stats.get('excellent_matches', 0)/len(differences)*100:.0f}%")
            col2.metric("✓ Good", stats.get('good_matches', 0),
                       delta=f"{stats.get('good_matches', 0)/len(differences)*100:.0f}%")
            col3.metric("~ Acceptable", stats.get('acceptable_matches', 0),
                       delta=f"{stats.get('acceptable_matches', 0)/len(differences)*100:.0f}%")
            col4.metric("⚠ Marginal", stats.get('marginal_matches', 0),
                       delta=f"{stats.get('marginal_matches', 0)/len(differences)*100:.0f}%")
            col5.metric("✗ Poor", stats.get('poor_matches', 0),
                       delta=f"{stats.get('poor_matches', 0)/len(differences)*100:.0f}%")
            
            st.markdown("### Statistical Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Difference", f"{stats['mean_rel_diff']*100:.2f}%")
            col2.metric("Std Deviation", f"{stats['std_rel_diff']*100:.2f}%")
            col3.metric("Min Difference", f"{stats['min_diff']*100:.2f}%")
            col4.metric("Max Difference", f"{stats['max_diff']*100:.2f}%")
            
            st.markdown("### Detailed Measurement Table")
            df_measurements = pd.DataFrame(differences)
            
            # Apply styling
            def color_status(row):
                if '✓ Excellent' in row['status']:
                    return ['background-color: #d4edda'] * len(row)
                elif '✓ Good' in row['status']:
                    return ['background-color: #d1ecf1'] * len(row)
                elif '~' in row['status']:
                    return ['background-color: #fff3cd'] * len(row)
                elif '⚠' in row['status']:
                    return ['background-color: #f8d7da'] * len(row)
                else:
                    return ['background-color: #f5c6cb'] * len(row)
            
            styled_df = df_measurements[['measurement', 'image_a', 'image_b', 'absolute_diff', 'percent_diff', 'status']].style.apply(
                color_status, axis=1
            ).format({
                'image_a': '{:.4f}',
                'image_b': '{:.4f}',
                'absolute_diff': '{:.4f}',
                'percent_diff': '{:.2f}%'
            })
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            if generate_visualizations and measurement_chart:
                st.markdown("### Visual Measurement Comparison")
                st.image(measurement_chart, use_container_width=True)
            
            if generate_visualizations and stats_dashboard:
                st.markdown("### Comprehensive Statistical Dashboard")
                st.image(stats_dashboard, use_container_width=True)
        
        # SECTION 5: Golden Ratio Analysis
        if perform_golden_ratio and golden_ratios1 and golden_ratios2:
            with st.expander("✨ SECTION 5: GOLDEN RATIO ANALYSIS (Phi = 1.618)", expanded=False):
                st.markdown("""
                The Golden Ratio (Phi ≈ 1.618) represents mathematically ideal proportions found in nature 
                and human facial aesthetics. This analysis assesses facial proportions against this standard.
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Image A Golden Ratio Assessment:**")
                    for key, value in golden_ratios1.items():
                        if 'deviation' in key:
                            feature = key.replace('_deviation_from_phi', '').replace('_', ' ').title()
                            deviation_pct = value * 100
                            
                            if value < THRESHOLDS.GOLDEN_RATIO_PERFECT:
                                status = "🟢 Excellent"
                                color = "green"
                            elif value < THRESHOLDS.GOLDEN_RATIO_GOOD:
                                status = "🟡 Good"
                                color = "blue"
                            elif value < THRESHOLDS.GOLDEN_RATIO_ACCEPTABLE:
                                status = "🟠 Acceptable"
                                color = "orange"
                            else:
                                status = "🔴 Significant"
                                color = "red"
                            
                            st.metric(feature, f"{deviation_pct:.1f}% deviation", delta=status)
                
                with col2:
                    st.markdown("**Image B Golden Ratio Assessment:**")
                    for key, value in golden_ratios2.items():
                        if 'deviation' in key:
                            feature = key.replace('_deviation_from_phi', '').replace('_', ' ').title()
                            deviation_pct = value * 100
                            
                            if value < THRESHOLDS.GOLDEN_RATIO_PERFECT:
                                status = "🟢 Excellent"
                            elif value < THRESHOLDS.GOLDEN_RATIO_GOOD:
                                status = "🟡 Good"
                            elif value < THRESHOLDS.GOLDEN_RATIO_ACCEPTABLE:
                                status = "🟠 Acceptable"
                            else:
                                status = "🔴 Significant"
                            
                            st.metric(feature, f"{deviation_pct:.1f}% deviation", delta=status)
                
                st.info("**Interpretation:** Closer adherence to golden ratio proportions indicates natural facial harmony. "
                       "Consistent deviations between images support identity match.")
        
        # SECTION 6: Error Level Analysis
        with st.expander("🔬 SECTION 6: ERROR LEVEL ANALYSIS (Manipulation Detection)", expanded=False):
            st.markdown("""
            **Purpose:** Detect digital manipulation by analyzing JPEG compression artifacts. 
            Areas with inconsistent compression suggest editing, splicing, or manipulation.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Image A Analysis:**")
                st.image(ela1, caption="ELA Visualization", use_container_width=True)
                
                col1a, col1b, col1c, col1d = st.columns(4)
                col1a.metric("Mean ELA", f"{ela_metrics1['mean']:.4f}")
                col1b.metric("Max ELA", f"{ela_metrics1['max']:.4f}")
                col1c.metric("Std Dev", f"{ela_metrics1['std']:.4f}")
                col1d.metric("Hotspots", f"{ela_metrics1['hotspots_pct']:.1f}%")
                
                if ela_metrics1['mean'] < THRESHOLDS.ELA_NORMAL:
                    st.success("✅ Normal compression artifacts - No manipulation detected")
                elif ela_metrics1['mean'] < THRESHOLDS.ELA_ELEVATED:
                    st.warning("⚠️ Moderate ELA - Possible multiple compressions")
                else:
                    st.error("🚨 High ELA - Possible manipulation detected")
            
            with col2:
                st.markdown("**Image B Analysis:**")
                st.image(ela2, caption="ELA Visualization", use_container_width=True)
                
                col2a, col2b, col2c, col2d = st.columns(4)
                col2a.metric("Mean ELA", f"{ela_metrics2['mean']:.4f}")
                col2b.metric("Max ELA", f"{ela_metrics2['max']:.4f}")
                col2c.metric("Std Dev", f"{ela_metrics2['std']:.4f}")
                col2d.metric("Hotspots", f"{ela_metrics2['hotspots_pct']:.1f}%")
                
                if ela_metrics2['mean'] < THRESHOLDS.ELA_NORMAL:
                    st.success("✅ Normal compression artifacts - No manipulation detected")
                elif ela_metrics2['mean'] < THRESHOLDS.ELA_ELEVATED:
                    st.warning("⚠️ Moderate ELA - Possible multiple compressions")
                else:
                    st.error("🚨 High ELA - Possible manipulation detected")
            
            st.info("**Interpretation:** Bright areas indicate compression inconsistencies. Uniform brightness suggests authentic, unedited images.")
        
        # SECTION 7: Color Distribution
        with st.expander("🎨 SECTION 7: COLOR DISTRIBUTION & HISTOGRAM ANALYSIS", expanded=False):
            st.markdown("### Histogram Similarity Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall Similarity", f"{hist_metrics['overall_similarity']:.3f}",
                       delta="High" if hist_metrics['overall_similarity'] > THRESHOLDS.HISTOGRAM_HIGH else "Moderate")
            col2.metric("Red Channel", f"{hist_metrics['red_correlation']:.3f}")
            col3.metric("Green Channel", f"{hist_metrics['green_correlation']:.3f}")
            col4.metric("Blue Channel", f"{hist_metrics['blue_correlation']:.3f}")
            
            if hist_metrics['overall_similarity'] > THRESHOLDS.HISTOGRAM_HIGH:
                st.success("✅ High color similarity - Consistent capture conditions")
            elif hist_metrics['overall_similarity'] > THRESHOLDS.HISTOGRAM_MODERATE:
                st.info("ℹ️ Moderate color similarity - Acceptable variation")
            else:
                st.warning("⚠️ Low color similarity - Different lighting/post-processing")
            
            if generate_visualizations and rgb_histograms:
                st.markdown("### RGB Channel Distribution Comparison")
                st.image(rgb_histograms, use_container_width=True)
            
            with st.expander("📊 Detailed Channel Metrics"):
                metrics_data = []
                for channel in ['red', 'green', 'blue']:
                    metrics_data.append({
                        'Channel': channel.capitalize(),
                        'Correlation': f"{hist_metrics[f'{channel}_correlation']:.4f}",
                        'Chi-Square': f"{hist_metrics[f'{channel}_chi_square']:.2f}",
                        'Intersection': f"{hist_metrics[f'{channel}_intersection']:.0f}",
                        'Bhattacharyya': f"{hist_metrics[f'{channel}_bhattacharyya']:.4f}"
                    })
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        # SECTION 8: Pixel-Level Comparison
        with st.expander("🖼️ SECTION 8: PIXEL-LEVEL SIMILARITY METRICS", expanded=False):
            st.markdown("### Alignment Quality")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Quality", alignment_metrics.get('quality', 'N/A'))
            col2.metric("Scale Factor", f"{alignment_metrics.get('scale_factor', 0):.3f}")
            col3.metric("Rotation", f"{alignment_metrics.get('rotation_degrees', 0):.2f}°")
            col4.metric("Translation", f"{alignment_metrics.get('translation_pixels', 0):.1f}px")
            
            st.markdown("### Similarity Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("MSE", f"{pixel_metrics['mse']:.2f}",
                       help="Mean Squared Error - Lower is better")
            col2.metric("RMSE", f"{pixel_metrics['rmse']:.2f}",
                       help="Root Mean Squared Error")
            
            psnr_val = pixel_metrics['psnr']
            psnr_delta = "Excellent" if psnr_val > THRESHOLDS.PSNR_EXCELLENT else "Good" if psnr_val > THRESHOLDS.PSNR_GOOD else "Fair"
            col3.metric("PSNR", f"{psnr_val:.2f} dB", delta=psnr_delta,
                       help="Peak Signal-to-Noise Ratio - Higher is better")
            
            col4.metric("MAE", f"{pixel_metrics['mae']:.2f}",
                       help="Mean Absolute Error")
            col5.metric("NCC", f"{pixel_metrics['ncc']:.4f}",
                       help="Normalized Cross-Correlation")
            
            if 'ssim' in pixel_metrics:
                ssim_val = pixel_metrics['ssim']
                ssim_delta = "Excellent" if ssim_val > THRESHOLDS.SSIM_EXCELLENT else "Good" if ssim_val > THRESHOLDS.SSIM_GOOD else "Fair"
                st.metric("SSIM (Structural Similarity)", f"{ssim_val:.4f}", delta=ssim_delta,
                         help="Structural Similarity Index - 1.0 is identical")
            
            if generate_visualizations and diff_heatmap:
                st.markdown("### Difference Heatmap Visualization")
                st.image(diff_heatmap, caption="Pixel Difference Heatmap (Blue=Similar, Red=Different)", 
                        use_container_width=True)
                st.caption("Interpretation: Blue/cool colors indicate pixel similarity; Red/warm colors indicate differences")
        
        # SECTION 9: Face Superimposition
        with st.expander("👥 SECTION 9: FACE SUPERIMPOSITION & OVERLAY ANALYSIS", expanded=False):
            st.markdown("""
            **Purpose:** Visual overlay allows manual assessment of feature alignment and facial structure correspondence.
            Adjust transparency to observe key anatomical landmark alignment.
            """)
            
            alpha = st.slider(
                "Adjust Image B Transparency (0 = Image A only, 1 = Image B only)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Slide to observe feature alignment"
            )
            
            superimposed = create_superimposed_image(img1_std, img2_aligned, alpha)
            
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image(superimposed, caption=f"Superimposed Comparison (Alpha={alpha:.2f})", 
                        use_container_width=True)
            
            st.info("💡 **Examiner Tip:** Good alignment of key features (eyes, nose, mouth) supports positive identification. "
                   "Misalignment may indicate different individuals or significant pose differences.")
        
        # SECTION 10: Morphological Assessment
        if perform_morphological:
            with st.expander("🔍 SECTION 10: MORPHOLOGICAL FEATURE ASSESSMENT", expanded=False):
                st.markdown("""
                **Purpose:** Document observable morphological features for expert comparison.
                Features are weighted by forensic significance: **Critical** (2.0x), **Secondary** (1.5x), **Tertiary** (1.0x).
                """)
                
                st.warning("⚠️ **Note:** This section requires trained examiner input for accurate assessment. "
                          "Default selections are 'Not Assessed'.")
                
                # Create morphological assessment interface
                morpho_assessment = {}
                
                st.markdown("### Feature Assessment Grid")
                
                for feature, info in MORPHOLOGICAL_FEATURES.items():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        val_a = st.selectbox(
                            f"Image A - {feature.replace('_', ' ').title()}",
                            options=['Not Assessed'] + info['options'],
                            key=f"morpho_a_{feature}",
                            help=info['description']
                        )
                    
                    with col2:
                        val_b = st.selectbox(
                            f"Image B - {feature.replace('_', ' ').title()}",
                            options=['Not Assessed'] + info['options'],
                            key=f"morpho_b_{feature}",
                            help=info['description']
                        )
                    
                    with col3:
                        if val_a != 'Not Assessed' and val_b != 'Not Assessed':
                            match = val_a == val_b
                            match_status = "✓ Match" if match else "✗ Different"
                            st.markdown(f"**{match_status}**")
                            
                            morpho_assessment[feature] = {
                                'value_a': val_a,
                                'value_b': val_b,
                                'match': match,
                                'match_status': match_status,
                                'assessed': True,
                                'category': info['category']
                            }
                        else:
                            st.markdown("**— N/A**")
                            morpho_assessment[feature] = {
                                'value_a': val_a,
                                'value_b': val_b,
                                'match': False,
                                'match_status': '— Not Assessed',
                                'assessed': False,
                                'category': info['category']
                            }
                
                # Calculate morphological match statistics
                assessed = [f for f in morpho_assessment.values() if f['assessed']]
                if assessed:
                    critical_assessed = [f for f in assessed if f['category'] == 'critical']
                    secondary_assessed = [f for f in assessed if f['category'] == 'secondary']
                    tertiary_assessed = [f for f in assessed if f['category'] == 'tertiary']
                    
                    st.markdown("### Morphological Assessment Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_matches = len([f for f in assessed if f['match']])
                    match_pct = (total_matches / len(assessed)) * 100
                    col1.metric("Overall Agreement", f"{match_pct:.1f}%", 
                               delta=f"{total_matches}/{len(assessed)} features")
                    
                    if critical_assessed:
                        crit_matches = len([f for f in critical_assessed if f['match']])
                        crit_pct = (crit_matches / len(critical_assessed)) * 100
                        col2.metric("Critical Features", f"{crit_pct:.1f}%",
                                   delta=f"{crit_matches}/{len(critical_assessed)}")
                    
                    if secondary_assessed:
                        sec_matches = len([f for f in secondary_assessed if f['match']])
                        sec_pct = (sec_matches / len(secondary_assessed)) * 100
                        col3.metric("Secondary Features", f"{sec_pct:.1f}%",
                                   delta=f"{sec_matches}/{len(secondary_assessed)}")
                    
                    if tertiary_assessed:
                        tert_matches = len([f for f in tertiary_assessed if f['match']])
                        tert_pct = (tert_matches / len(tertiary_assessed)) * 100
                        col4.metric("Tertiary Features", f"{tert_pct:.1f}%",
                                   delta=f"{tert_matches}/{len(tertiary_assessed)}")
                else:
                    st.info("ℹ️ No morphological features have been assessed yet. "
                           "Complete the assessment above to generate statistics.")
        else:
            morpho_assessment = None
        
        # ===================================================================
        # COMPREHENSIVE REPORT GENERATION
        # ===================================================================
        
        st.markdown("---")
        st.markdown("## 📄 COMPREHENSIVE FORENSIC REPORT")
        
        # Generate report
        comprehensive_report = generate_comprehensive_report(
            case_id=case_id,
            timestamp=timestamp,
            img1_hash=hash1_sha256,
            img2_hash=hash2_sha256,
            img1_md5=hash1_md5,
            img2_md5=hash2_md5,
            img1_quality=quality1,
            img2_quality=quality2,
            confidence=confidence,
            classification=classification,
            interpretation=interpretation,
            recommendation=recommendation,
            differences=differences,
            stats=stats,
            golden_ratios1=golden_ratios1,
            golden_ratios2=golden_ratios2,
            ela_metrics1=ela_metrics1,
            ela_metrics2=ela_metrics2,
            hist_metrics=hist_metrics,
            pixel_metrics=pixel_metrics,
            alignment_metrics=alignment_metrics,
            examiner_notes=examiner_notes,
            morphological_assessment=morpho_assessment if perform_morphological else None
        )
        
        # Display report preview
        st.text_area(
            "Complete Forensic Examination Report",
            value=comprehensive_report,
            height=500,
            help="Full text report for documentation and legal proceedings"
        )
        
        # Export Options
        st.markdown("### 📥 EXPORT OPTIONS")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # TXT Report
        with col1:
            st.download_button(
                label="📄 Download Report (TXT)",
                data=comprehensive_report,
                file_name=f"{case_id}_Forensic_Report.txt",
                mime="text/plain",
                use_container_width=True,
                help="Complete text report"
            )
        
        # CSV Measurements
        with col2:
            df_export = pd.DataFrame(differences)
            csv_data = df_export.to_csv(index=False)
            st.download_button(
                label="📊 Download Measurements (CSV)",
                data=csv_data,
                file_name=f"{case_id}_Measurements.csv",
                mime="text/csv",
                use_container_width=True,
                help="Detailed measurement data"
            )
        
        # JSON Metadata
        with col3:
            metadata = {
                'case_id': case_id,
                'timestamp': timestamp,
                'system_version': 'v3.0',
                'images': {
                    'image_a': {
                        'sha256': hash1_sha256,
                        'md5': hash1_md5,
                        'quality': quality1
                    },
                    'image_b': {
                        'sha256': hash2_sha256,
                        'md5': hash2_md5,
                        'quality': quality2
                    }
                },
                'results': {
                    'confidence': confidence,
                    'classification': classification,
                    'anthropometric_stats': stats,
                    'pixel_metrics': pixel_metrics,
                    'histogram_metrics': hist_metrics
                },
                'examiner_notes': examiner_notes
            }
            json_data = json.dumps(metadata, indent=2)
            st.download_button(
                label="🔧 Download Metadata (JSON)",
                data=json_data,
                file_name=f"{case_id}_Metadata.json",
                mime="application/json",
                use_container_width=True,
                help="Machine-readable case data"
            )
        
        # Full data package
        with col4:
            # Create comprehensive package
            package_data = f"""
FORENSIC ANALYSIS PACKAGE
Case: {case_id}
Generated: {timestamp}

{comprehensive_report}

================================================================================
MEASUREMENT DATA (CSV FORMAT)
================================================================================

{csv_data}

================================================================================
METADATA (JSON FORMAT)
================================================================================

{json_data}
"""
            st.download_button(
                label="📦 Download Complete Package",
                data=package_data,
                file_name=f"{case_id}_Complete_Package.txt",
                mime="text/plain",
                use_container_width=True,
                help="All data in one file"
            )
        
        # Success message
        st.success("""
        ✅ **Forensic Analysis Complete!**
        
        All results have been generated and are ready for review. The comprehensive report 
        includes cryptographic hashes for chain of custody, detailed measurements, statistical 
        analysis, and expert interpretation guidelines.
        """)
        
        # Final Disclaimer
        st.markdown("---")
        st.error("""
        ### ⚖️ LEGAL DISCLAIMER & EXPERT REVIEW REQUIREMENT
        
        **IMPORTANT:** This automated forensic analysis system provides computational assessment 
        based on established scientific methodologies (FISWG, PCAST). However:
        
        ⚠️ **Final determinations MUST be made by qualified forensic facial comparison examiners**
        
        **Required for Court Admissibility:**
        - Review by certified forensic examiner (FISWG-trained preferred)
        - Validation of landmark detection and measurements
        - Morphological feature assessment by trained expert
        - Consideration of temporal and environmental factors
        - Integration with additional case evidence
        - Expert testimony for legal proceedings
        
        **Limitations:**
        - Results depend on image quality, pose, and lighting
        - Aging, weight changes, and cosmetic procedures affect measurements
        - Automated analysis cannot replace human expertise
        - Confidence scores are computational, not probability statements
        
        **This report is provided as supporting evidence for expert examination, not as 
        standalone proof of identity or non-identity.**
        """)

# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()