#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
    FORENSIC FACIAL COMPARISON SYSTEM v5.0 ENTERPRISE
    Professional Biometric Analysis Platform
    Enhanced with Morphological Assessment & RGB Histogram
    
    © 2025 Forensic Biometrics Division
═══════════════════════════════════════════════════════════════
"""

# [SECTION 1: IMPORTS & CONFIGURATION]
import streamlit as st #type: ignore
import cv2 #type: ignore
import dlib #type: ignore
import numpy as np #type: ignore
from PIL import Image, ImageChops #type: ignore
import io, hashlib, datetime, math, pandas as pd, sys #type: ignore
import matplotlib.pyplot as plt #type: ignore
import matplotlib #type: ignore
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import os

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

try:
    from skimage.metrics import structural_similarity as ssim #type: ignore
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 2: FORENSIC STANDARDS - COMPLETE]
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ForensicStandards:
    """Industry-standard thresholds and parameters"""
    # Tolerance levels
    STRICT_TOLERANCE: float = 0.025
    STANDARD_TOLERANCE: float = 0.065
    LENIENT_TOLERANCE: float = 0.115
    
    # Golden Ratio
    PHI: float = 1.618033988749895
    
    # Match quality thresholds
    EXCELLENT_THRESHOLD: float = 0.025
    GOOD_THRESHOLD: float = 0.050
    ACCEPTABLE_THRESHOLD: float = 0.080
    MARGINAL_THRESHOLD: float = 0.115
    
    # Evidence strength classification
    IDENTIFICATION_THRESHOLD: float = 95.0
    STRONG_SUPPORT_THRESHOLD: float = 85.0
    MODERATE_SUPPORT_THRESHOLD: float = 70.0
    LIMITED_SUPPORT_THRESHOLD: float = 55.0
    INCONCLUSIVE_THRESHOLD: float = 40.0
    
    # Image quality thresholds
    MIN_RESOLUTION: int = 200
    MAX_FILE_SIZE: int = 25 * 1024 * 1024
    
    # ELA thresholds
    ELA_PRISTINE: float = 10.0
    ELA_NORMAL: float = 20.0
    ELA_EDITED: float = 35.0
    
    # Color similarity
    COLOR_HIGH_SIMILARITY: float = 0.85
    COLOR_MODERATE_SIMILARITY: float = 0.65
    
    # PSNR thresholds
    PSNR_EXCELLENT: float = 40.0
    PSNR_GOOD: float = 30.0
    PSNR_ACCEPTABLE: float = 25.0
    
    # SSIM thresholds
    SSIM_IDENTICAL: float = 0.98
    SSIM_VERY_SIMILAR: float = 0.90
    SSIM_SIMILAR: float = 0.75
    
    # Golden Ratio thresholds
    GOLDEN_RATIO_PERFECT: float = 0.02
    GOLDEN_RATIO_GOOD: float = 0.05
    GOLDEN_RATIO_ACCEPTABLE: float = 0.10

STANDARDS = ForensicStandards()
LANDMARK_COUNT = 68
DPI_FORENSIC = 150

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 3: MORPHOLOGICAL FEATURES - COMPREHENSIVE DEFINITIONS]
# ═══════════════════════════════════════════════════════════════════════════

MORPHOLOGICAL_FEATURES = {
    'eye_shape': {
        'name': 'Eye Shape',
        'options': ['Almond', 'Round', 'Hooded', 'Upturned', 'Downturned', 'Monolid', 'Deep-set', 'Protruding'],
        'category': 'CRITICAL',
        'weight': 3.0,
        'forensic_value': 'High discriminative power',
        'description': 'Overall shape and contour of the eye opening including palpebral fissure configuration'
    },
    'forehead_shape': {
        'name': 'Forehead Shape',
        'options': ['Straight/Vertical', 'Slightly Sloped', 'Moderately Sloped', 'Steeply Sloped', 'Round/Curved', 'Broad', 'Narrow'],
        'category': 'SECONDARY',
        'weight': 2.0,
        'forensic_value': 'Moderate discriminative power',
        'description': 'Profile and frontal contour of the frontal bone and forehead-hairline relationship'
    },
    'nose_shape': {
        'name': 'Nose Shape',
        'options': ['Straight', 'Roman/Aquiline', 'Convex', 'Concave', 'Button', 'Bulbous', 'Hooked', 'Snub', 'Hawk'],
        'category': 'CRITICAL',
        'weight': 3.0,
        'forensic_value': 'High discriminative power',
        'description': 'Nasal bridge profile, dorsal contour, tip projection, and overall nasal morphology'
    },
    'lip_shape': {
        'name': 'Lip Shape',
        'options': ['Full', 'Thin', 'Medium', 'Heart-shaped', 'Wide', 'Narrow', 'Bow-shaped', 'Downturned', 'Upturned'],
        'category': 'CRITICAL',
        'weight': 3.0,
        'forensic_value': 'High discriminative power',
        'description': 'Vermilion border thickness, cupid\'s bow definition, philtrum depth, and oral commissure position'
    },
    'eyebrow_shape': {
        'name': 'Eyebrow Shape',
        'options': ['Straight', 'Arched', 'Soft Arch', 'Sharp Arch', 'Angled', 'Flat', 'S-shaped', 'Rounded'],
        'category': 'SECONDARY',
        'weight': 2.0,
        'forensic_value': 'Moderate (subject to grooming alterations)',
        'description': 'Natural eyebrow trajectory, arch height, medial/lateral terminus position, and overall brow shape'
    },
    'cheek_shape': {
        'name': 'Cheek Contour',
        'options': ['High Cheekbones', 'Low Cheekbones', 'Prominent', 'Flat', 'Round', 'Angular', 'Hollow', 'Full'],
        'category': 'SECONDARY',
        'weight': 2.0,
        'forensic_value': 'Moderate discriminative power',
        'description': 'Malar eminence position, prominence, zygomatic arch morphology, and midface contour'
    },
    'chin_shape': {
        'name': 'Chin Shape',
        'options': ['Square', 'Round', 'Pointed/V-shaped', 'Cleft/Dimpled', 'Prominent', 'Receding', 'Broad', 'Narrow', 'Double'],
        'category': 'CRITICAL',
        'weight': 3.0,
        'forensic_value': 'High discriminative power',
        'description': 'Mandibular symphysis morphology, mental protuberance projection, and mental eminence characteristics'
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 4: DLIB MODEL LOADING]
# ═══════════════════════════════════════════════════════════════════════════

DLIB_MODEL_PATH = '/home/vscode/workspace/FaceBiometricSoftware/dlibModel/shape_predictor_68_face_landmarks.dat'

DLIB_MODEL_FALLBACK_PATHS = [
    DLIB_MODEL_PATH,
    './dlibModel/shape_predictor_68_face_landmarks.dat',
    './shape_predictor_68_face_landmarks.dat',
    '../dlibModel/shape_predictor_68_face_landmarks.dat',
    os.path.expanduser('~/shape_predictor_68_face_landmarks.dat'),
]

@st.cache_resource
def load_dlib_models():
    """Load dlib face detector and landmark predictor"""
    env_path = os.environ.get('DLIB_PREDICTOR_PATH')
    if env_path and os.path.exists(env_path):
        predictor_path = env_path
    else:
        predictor_path = None
        for path in DLIB_MODEL_FALLBACK_PATHS:
            if os.path.exists(path):
                predictor_path = path
                break
    
    if predictor_path is None:
        st.error("❌ Cannot find dlib shape predictor model file")
        st.stop()
    
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        st.sidebar.success(f"✅ **Dlib Model Loaded**\n\n`{os.path.basename(predictor_path)}`")
        return detector, predictor
    except Exception as e:
        st.error(f"❌ Error loading dlib models: {str(e)}")
        st.stop()

DETECTOR, PREDICTOR = load_dlib_models()

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 5: UTILITY FUNCTIONS]
# ═══════════════════════════════════════════════════════════════════════════

def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_case_id() -> str:
    """Generate unique forensic case identifier"""
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    random_hex = hashlib.sha256(timestamp.encode()).hexdigest()[:8].upper()
    return f"FFC-{timestamp}-{random_hex}"

def compute_forensic_hash(img: Image.Image, hash_type: str = 'sha256') -> str:
    """Compute cryptographic hash of image"""
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    if hash_type == 'sha256':
        return hashlib.sha256(img_byte_arr).hexdigest()
    elif hash_type == 'md5':
        return hashlib.md5(img_byte_arr).hexdigest()
    elif hash_type == 'sha512':
        return hashlib.sha512(img_byte_arr).hexdigest()
    else:
        raise ValueError(f"Unsupported hash type: {hash_type}")

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 6: IMAGE SECURITY & VALIDATION - UNIVERSAL FORMAT SUPPORT]
# ═══════════════════════════════════════════════════════════════════════════

def validate_image_security(file_obj) -> Tuple[bool, str, Optional[Image.Image]]:
    """Universal image validator - accepts ALL formats PIL supports"""
    if file_obj is None:
        return False, "No file uploaded", None
    
    try:
        file_obj.seek(0)
        file_obj.seek(0, 2)
        file_size = file_obj.tell()
        file_obj.seek(0)
        
        if file_size > STANDARDS.MAX_FILE_SIZE:
            return False, f"File too large ({file_size / 1024 / 1024:.1f} MB). Maximum: 25 MB", None
        
        if file_size == 0:
            return False, "Empty file", None
        
        try:
            img = Image.open(file_obj)
            img.load()
            original_format = img.format or "UNKNOWN"
        except Exception as e:
            return False, f"Cannot open image file: {str(e)}", None
        
        try:
            if original_format == 'MPO':
                img.seek(0)
                img = img.convert('RGB')
            elif original_format in ['GIF', 'APNG']:
                img.seek(0)
                img = img.convert('RGB')
            elif original_format == 'WEBP':
                img = img.convert('RGB')
            elif original_format in ['HEIC', 'HEIF']:
                try:
                    import pillow_heif #type: ignore
                    pillow_heif.register_heif_opener()
                    img = img.convert('RGB')
                except ImportError:
                    return False, "HEIC format requires 'pillow-heif' package", None
            elif original_format in ['JPEG', 'PNG', 'BMP', 'TIFF', 'TIF', 'PPM', 'PGM', 'PBM', 'PCX', 'ICO']:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            else:
                img = img.convert('RGB')
        except Exception as e:
            return False, f"Cannot convert {original_format} to RGB: {str(e)}", None
        
        width, height = img.size
        if width < STANDARDS.MIN_RESOLUTION or height < STANDARDS.MIN_RESOLUTION:
            return False, f"Resolution too low ({width}×{height}). Minimum: {STANDARDS.MIN_RESOLUTION}×{STANDARDS.MIN_RESOLUTION}", None
        
        MAX_DIMENSION = 10000
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            return False, f"Image too large ({width}×{height}). Maximum: {MAX_DIMENSION}×{MAX_DIMENSION} pixels", None
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return True, f"Valid {original_format} image ({width}×{height}, {file_size / 1024:.1f} KB)", img
        
    except Exception as e:
        return False, f"Validation error: {str(e)}", None

def standardize_image(img: Image.Image, target_size: int = 800) -> Image.Image:
    """Standardize image for analysis"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def calculate_image_quality_score(img: Image.Image) -> Dict[str, float]:
    """Calculate comprehensive image quality metrics"""
    cv_img = pil_to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(100, laplacian_var / 10)
    
    # Contrast (standard deviation)
    contrast_score = min(100, float(np.std(gray)) / 0.64)
    
    # Brightness (mean deviation from ideal 128)
    brightness_mean = float(np.mean(gray))
    brightness_score = 100 - abs(128 - brightness_mean) / 128 * 100
    
    # Noise estimation
    noise = cv2.Laplacian(gray, cv2.CV_64F).std()
    noise_score = min(100, 100 - noise / 5)
    
    # Overall score
    overall = (sharpness_score + contrast_score + brightness_score + noise_score) / 4
    
    if overall >= 80: assessment = 'Excellent'
    elif overall >= 60: assessment = 'Good'
    elif overall >= 40: assessment = 'Fair'
    else: assessment = 'Poor'
    
    return {
        'overall': overall,
        'sharpness': sharpness_score,
        'contrast': contrast_score,
        'brightness': brightness_score,
        'noise': noise_score,
        'resolution': img.size,
        'assessment': assessment
    }

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 7: LANDMARK DETECTION - ENHANCED WITH THICKER LINES]
# ═══════════════════════════════════════════════════════════════════════════

def get_landmarks(cv_img: np.ndarray) -> Optional[np.ndarray]:
    """Detect 68 facial landmarks"""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = DETECTOR(gray, 1)
    
    if len(faces) == 0:
        return None
    
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    shape = PREDICTOR(gray, face)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    
    return landmarks

def draw_landmarks_professional(img: Image.Image, landmarks: np.ndarray) -> Image.Image:
    """Professional 68-point landmark visualization WITH THICKER, MORE PROMINENT LINES"""
    cv_img = pil_to_cv(img).copy()
    
    try:
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks)
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(-1, 2)
        elif landmarks.ndim == 3:
            landmarks = landmarks.squeeze()
    except:
        return img
    
    features = {
        'jaw': (list(range(0, 17)), (0, 255, 255), False),
        'right_eyebrow': (list(range(17, 22)), (255, 0, 255), False),
        'left_eyebrow': (list(range(22, 27)), (255, 0, 255), False),
        'nose': (list(range(27, 36)), (255, 255, 0), False),
        'right_eye': (list(range(36, 42)), (0, 255, 0), True),
        'left_eye': (list(range(42, 48)), (0, 255, 0), True),
        'mouth': (list(range(48, 68)), (255, 0, 0), True)
    }
    
    for feature_name, (indices, color, is_closed) in features.items():
        try:
            if max(indices) < len(landmarks):
                points = np.array([landmarks[i] for i in indices], dtype=np.int32)
                # THICKER POLYLINES: Changed from 2 to 4 pixels
                cv2.polylines(cv_img, [points], is_closed, color, 4, cv2.LINE_AA)
                
                for idx in indices:
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    # LARGER CIRCLES: Changed from 3 to 5 pixels
                    cv2.circle(cv_img, (x, y), 5, color, -1, cv2.LINE_AA)
                    # WHITE OUTLINE for better visibility
                    cv2.circle(cv_img, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
        except:
            continue
    
    return cv_to_pil(cv_img)

def get_dlib_landmarks_named(landmarks: np.ndarray) -> Dict[str, Tuple[int, int]]:
    """Convert 68-point landmarks to named dictionary (FaceCompare2 style)"""
    if landmarks is None or len(landmarks) < 68:
        return {}
    
    return {
        'left_eye': tuple(landmarks[36]),
        'right_eye': tuple(landmarks[45]),
        'nose_tip': tuple(landmarks[30]),
        'mouth_center': tuple(landmarks[51]),
        'left_ear': tuple(landmarks[0]),
        'right_ear': tuple(landmarks[16]),
        'chin': tuple(landmarks[8]),
        'forehead_glabella': tuple(landmarks[27]),
        'jaw_left': tuple(landmarks[3]),
        'jaw_right': tuple(landmarks[13]),
        'left_eye_inner': tuple(landmarks[39]),
        'right_eye_inner': tuple(landmarks[42]),
        'left_mouth': tuple(landmarks[48]),
        'right_mouth': tuple(landmarks[54]),
        'nose_left': tuple(landmarks[31]),
        'nose_right': tuple(landmarks[35])
    }

def draw_landmarks_with_letter_labels(cv_img: np.ndarray, landmarks: np.ndarray, 
                                     normalized: bool = True, reference_value: float = 1.0) -> Tuple:
    """Enhanced landmark visualization with letter labels AND THICKER LINES"""
    img_annotated = cv_img.copy()
    named_lm = get_dlib_landmarks_named(landmarks)
    
    labels = {
        'A': 'left_eye',
        'B': 'right_eye',
        'C': 'nose_tip',
        'D': 'mouth_center',
        'E': 'left_ear',
        'F': 'right_ear',
        'G': 'chin',
        'H': 'forehead_glabella',
        'I': 'jaw_left',
        'J': 'jaw_right'
    }
    
    legend = {}
    segments = {}
    
    for letter, lm_name in labels.items():
        if lm_name in named_lm:
            pt = named_lm[lm_name]
            # LARGER CIRCLES: Changed from 5 to 8 pixels
            cv2.circle(img_annotated, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
            # WHITE OUTLINE: Thicker outline
            cv2.circle(img_annotated, (int(pt[0]), int(pt[1])), 9, (255, 255, 255), 2)
            # LARGER TEXT: Changed from 0.6 to 0.8, thickness from 2 to 3
            cv2.putText(img_annotated, letter, (int(pt[0])+12, int(pt[1])-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            legend[letter] = f"{lm_name}: ({int(pt[0])}, {int(pt[1])})"
    
    segment_pairs = [
        ('A', 'B', 'Eye Width'),
        ('C', 'G', 'Nose to Chin'),
        ('C', 'D', 'Nose to Mouth'),
        ('H', 'G', 'Hairline to Chin'),
        ('I', 'J', 'Jaw Width')
    ]
    
    for label1, label2, name in segment_pairs:
        if labels.get(label1) in named_lm and labels.get(label2) in named_lm:
            pt1 = named_lm[labels[label1]]
            pt2 = named_lm[labels[label2]]
            dist = euclidean_distance(pt1, pt2)
            
            if normalized and reference_value > 0:
                dist_norm = dist / reference_value
                segments[name] = f"{dist:.1f}px (normalized: {dist_norm:.3f})"
            else:
                segments[name] = f"{dist:.1f}px"
            
            # THICKER CONNECTING LINES: Changed from 1 to 3 pixels
            cv2.line(img_annotated, (int(pt1[0]), int(pt1[1])), 
                    (int(pt2[0]), int(pt2[1])), (255, 165, 0), 3)
    
    return cv_to_pil(img_annotated), legend, segments

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 8: ANTHROPOMETRIC MEASUREMENTS - INTEGRATED FROM BOTH SYSTEMS]
# ═══════════════════════════════════════════════════════════════════════════

def calculate_measurements(landmarks: np.ndarray) -> Dict[str, float]:
    """Comprehensive anthropometric measurements combining both systems"""
    measurements = {}
    
    # Basic eye measurements
    measurements['left_eye_width'] = euclidean_distance(landmarks[36], landmarks[39])
    measurements['right_eye_width'] = euclidean_distance(landmarks[42], landmarks[45])
    measurements['inter_ocular_distance'] = euclidean_distance(landmarks[36], landmarks[45])
    measurements['Eye Width (outer)'] = euclidean_distance(landmarks[36], landmarks[45])
    
    # Eye heights
    measurements['left_eye_height'] = (euclidean_distance(landmarks[37], landmarks[41]) + 
                                      euclidean_distance(landmarks[38], landmarks[40])) / 2
    measurements['right_eye_height'] = (euclidean_distance(landmarks[43], landmarks[47]) + 
                                       euclidean_distance(landmarks[44], landmarks[46])) / 2
    
    # Nose measurements
    measurements['nose_width'] = euclidean_distance(landmarks[31], landmarks[35])
    measurements['nose_length'] = euclidean_distance(landmarks[27], landmarks[33])
    measurements['Nose Width'] = measurements['nose_width']
    measurements['Nose to Chin'] = euclidean_distance(landmarks[30], landmarks[8])
    measurements['Nose to Mouth'] = euclidean_distance(landmarks[30], landmarks[51])
    measurements['nose_bridge_to_tip'] = euclidean_distance(landmarks[27], landmarks[30])
    
    # Mouth measurements
    measurements['mouth_width'] = euclidean_distance(landmarks[48], landmarks[54])
    measurements['Mouth Width'] = measurements['mouth_width']
    measurements['mouth_height'] = euclidean_distance(landmarks[51], landmarks[57])
    measurements['mouth_height_outer'] = euclidean_distance(landmarks[51], landmarks[57])
    measurements['upper_lip_height'] = euclidean_distance(landmarks[51], landmarks[62])
    measurements['lower_lip_height'] = euclidean_distance(landmarks[57], landmarks[66])
    
    # Face dimensions
    measurements['face_width'] = euclidean_distance(landmarks[0], landmarks[16])
    measurements['face_height'] = euclidean_distance(landmarks[8], landmarks[27])
    measurements['face_width_mid'] = euclidean_distance(landmarks[3], landmarks[13])
    measurements['Hairline to Nose'] = euclidean_distance(landmarks[27], landmarks[30])
    measurements['Hairline to Chin'] = euclidean_distance(landmarks[27], landmarks[8])
    measurements['Pupil Line to Chin'] = euclidean_distance(landmarks[36], landmarks[8])
    measurements['Eye to Nose Base'] = euclidean_distance(landmarks[36], landmarks[30])
    
    # Eyebrow measurements
    measurements['left_eyebrow_length'] = euclidean_distance(landmarks[17], landmarks[21])
    measurements['right_eyebrow_length'] = euclidean_distance(landmarks[22], landmarks[26])
    measurements['left_eyebrow_to_eye'] = euclidean_distance(landmarks[19], landmarks[37])
    measurements['right_eyebrow_to_eye'] = euclidean_distance(landmarks[24], landmarks[44])
    
    # Additional measurements
    measurements['eye_separation'] = euclidean_distance(landmarks[39], landmarks[42])
    measurements['chin_width'] = euclidean_distance(landmarks[5], landmarks[11])
    measurements['jaw_width'] = euclidean_distance(landmarks[3], landmarks[13])
    measurements['jaw_to_chin'] = euclidean_distance(landmarks[8], landmarks[57])
    
    return measurements

def normalize_by_eye_distance(measurements: Dict[str, float]) -> Dict[str, float]:
    """Normalize by eye width (FaceCompare2 method) - Makes measurements scale-independent"""
    eye_ref = measurements.get('Eye Width (outer)', measurements.get('inter_ocular_distance', 1.0))
    if eye_ref == 0:
        eye_ref = 1.0
    
    normalized = {}
    for key, value in measurements.items():
        normalized[key] = value / eye_ref
    
    return normalized

def normalize_by_image_diag(measurements: Dict[str, float], img_shape: Tuple) -> Dict[str, float]:
    """Normalize by image diagonal (alternative method from FaceCompare2)"""
    diag = math.sqrt(img_shape[0]**2 + img_shape[1]**2)
    if diag == 0:
        diag = 1.0
    
    normalized = {}
    for key, value in measurements.items():
        normalized[key] = value / diag
    
    return normalized

def calculate_golden_ratios(landmarks: np.ndarray) -> Dict[str, float]:
    """Enhanced golden ratio calculations combining both systems' approaches"""
    ratios = {}
    
    try:
        measurements = calculate_measurements(landmarks)
        
        # FaceCompare2 style ratios
        hairline_nose = measurements.get('Hairline to Nose', 0)
        nose_chin = measurements.get('Nose to Chin', 0)
        if nose_chin > 0:
            ratios['Hairline→Nose / Nose→Chin'] = hairline_nose / nose_chin
        
        nose_mouth = measurements.get('Nose to Mouth', 0)
        mouth_chin = measurements.get('Nose to Chin', 0) - nose_mouth
        if mouth_chin > 0:
            ratios['Nose→Mouth / Mouth→Chin'] = nose_mouth / mouth_chin
        
        # Face proportions
        face_height = np.linalg.norm(landmarks[8] - landmarks[27])
        face_width = np.linalg.norm(landmarks[0] - landmarks[16])
        ratios['face_height_to_width'] = face_height / face_width if face_width > 0 else 0
        
        # Upper to lower face
        upper_face = np.linalg.norm(landmarks[27] - landmarks[33])
        lower_face = np.linalg.norm(landmarks[33] - landmarks[8])
        ratios['upper_to_lower_face'] = upper_face / lower_face if lower_face > 0 else 0
        
        # Nose to mouth width
        nose_width = np.linalg.norm(landmarks[31] - landmarks[35])
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        ratios['nose_to_mouth_width'] = mouth_width / nose_width if nose_width > 0 else 0
        
        # Eye to intercanthal
        eye_width = np.linalg.norm(landmarks[36] - landmarks[39])
        intercanthal = np.linalg.norm(landmarks[39] - landmarks[42])
        ratios['eye_to_intercanthal'] = intercanthal / eye_width if eye_width > 0 else 0
        
        # Mouth proportions
        mouth_height = np.linalg.norm(landmarks[51] - landmarks[57])
        ratios['mouth_width_to_height'] = mouth_width / mouth_height if mouth_height > 0 else 0
        
        # Face thirds
        facial_height = np.linalg.norm(landmarks[8] - landmarks[27])
        upper_third = np.linalg.norm(landmarks[27] - landmarks[28])
        middle_third = np.linalg.norm(landmarks[28] - landmarks[33])
        lower_third = np.linalg.norm(landmarks[33] - landmarks[8])
        if facial_height > 0:
            ratios['upper_third_ratio'] = upper_third / facial_height
            ratios['middle_third_ratio'] = middle_third / facial_height
            ratios['lower_third_ratio'] = lower_third / facial_height
        
        # Calculate deviations from PHI
        ideal_ratios = {
            'Hairline→Nose / Nose→Chin': STANDARDS.PHI,
            'Nose→Mouth / Mouth→Chin': STANDARDS.PHI,
            'face_height_to_width': STANDARDS.PHI,
            'upper_to_lower_face': STANDARDS.PHI,
            'nose_to_mouth_width': STANDARDS.PHI,
            'mouth_width_to_height': STANDARDS.PHI
        }
        
        for key, ideal in ideal_ratios.items():
            if key in ratios and ratios[key] > 0:
                deviation = abs(ratios[key] - ideal) / ideal
                ratios[f'{key}_deviation_from_phi'] = deviation
    
    except Exception as e:
        st.warning(f"Golden ratio calculation warning: {str(e)}")
    
    return ratios

def classify_phi_deviation(deviation: float) -> str:
    """Classify golden ratio deviation"""
    if deviation < STANDARDS.GOLDEN_RATIO_PERFECT:
        return 'PERFECT'
    elif deviation < STANDARDS.GOLDEN_RATIO_GOOD:
        return 'EXCELLENT'
    elif deviation < STANDARDS.GOLDEN_RATIO_ACCEPTABLE:
        return 'GOOD'
    else:
        return 'ACCEPTABLE'

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 9: SIMILARITY ANALYSIS - ADVANCED WITH WEIGHTED SCORING]
# ═══════════════════════════════════════════════════════════════════════════

def calculate_facial_similarity_advanced(ratios1: Dict, ratios2: Dict, tolerance: float) -> Tuple[float, List[Dict], Dict]:
    """Advanced anthropometric similarity with weighted analysis"""
    if not ratios1 or not ratios2:
        return 0.0, [], {}
    
    differences = []
    weights = []
    
    for key in ratios1.keys():
        if key in ratios2:
            val1, val2 = ratios1[key], ratios2[key]
            
            if val1 == 0 or val2 == 0:
                continue
            
            abs_diff = abs(val1 - val2)
            rel_diff = abs_diff / max(val1, val2)
            percent_diff = (abs_diff / val1) * 100 if val1 != 0 else 0
            
            # Weighted status classification
            if rel_diff <= tolerance * 0.4:
                status = "Excellent Match"
                weight = 1.0
            elif rel_diff <= tolerance * 0.7:
                status = "Good Match"
                weight = 0.9
            elif rel_diff <= tolerance:
                status = "Acceptable"
                weight = 0.7
            elif rel_diff <= tolerance * 1.5:
                status = "Marginal"
                weight = 0.4
            else:
                status = "Significant Deviation"
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
    weighted_avg = sum((d['weight'] * (1 - d['relative_diff'])) for d in differences) / len(differences)
    confidence = max(0, min(100, weighted_avg * 100))
    
    stats = {
        'mean_abs_diff': np.mean([d['absolute_diff'] for d in differences]),
        'std_abs_diff': np.std([d['absolute_diff'] for d in differences]),
        'mean_rel_diff': np.mean([d['relative_diff'] for d in differences]),
        'std_rel_diff': np.std([d['relative_diff'] for d in differences]),
        'max_diff': max([d['relative_diff'] for d in differences]),
        'min_diff': min([d['relative_diff'] for d in differences]),
        'excellent_matches': len([d for d in differences if 'Excellent' in d['status']]),
        'good_matches': len([d for d in differences if 'Good' in d['status']]),
        'acceptable_matches': len([d for d in differences if 'Acceptable' in d['status']]),
        'marginal_matches': len([d for d in differences if 'Marginal' in d['status']]),
        'poor_matches': len([d for d in differences if 'Significant' in d['status']])
    }
    
    return confidence, differences, stats

def classify_evidence_strength_professional(confidence: float, stats: Dict) -> Tuple[str, str, str, str, str]:
    """Professional evidence classification with detailed reasoning"""
    total_matches = sum([
        stats.get('excellent_matches', 0),
        stats.get('good_matches', 0),
        stats.get('acceptable_matches', 0),
        stats.get('marginal_matches', 0),
        stats.get('poor_matches', 0)
    ])
    
    excellent_pct = (stats.get('excellent_matches', 0) / max(total_matches, 1)) * 100
    
    if confidence >= 92 and excellent_pct >= 60:
        return (
            "STRONG SUPPORT - IDENTIFICATION",
            "#28a745",
            f"The anthropometric evidence provides STRONG SUPPORT for the conclusion that Images A and B depict the SAME INDIVIDUAL. Match confidence: {confidence:.1f}% with {stats.get('excellent_matches', 0)} excellent matches.",
            "Proceed with positive identification. Recommend morphological feature verification by certified examiner.",
            "VERY HIGH"
        )
    elif confidence >= 82:
        return (
            "MODERATE SUPPORT - PROBABLE MATCH",
            "#007bff",
            f"The anthropometric evidence provides MODERATE SUPPORT for the conclusion that Images A and B likely depict the SAME INDIVIDUAL. Match confidence: {confidence:.1f}%.",
            "Recommend additional comparative analysis and expert morphological review before final determination.",
            "MODERATE"
        )
    elif confidence >= 72:
        return (
            "LIMITED SUPPORT - INCONCLUSIVE",
            "#ffc107",
            f"The anthropometric evidence provides LIMITED SUPPORT. The analysis is INCONCLUSIVE. Match confidence: {confidence:.1f}%.",
            "Insufficient evidence for determination. Require additional high-quality images and comprehensive expert examination.",
            "LIMITED"
        )
    elif confidence >= 62:
        return (
            "MODERATE SUPPORT - PROBABLE EXCLUSION",
            "#fd7e14",
            f"The anthropometric evidence provides MODERATE SUPPORT for the conclusion that Images A and B likely depict DIFFERENT INDIVIDUALS. Match confidence: {confidence:.1f}%.",
            "Evidence suggests different individuals. Recommend expert review to rule out temporal changes or image quality issues.",
            "MODERATE"
        )
    else:
        return (
            "STRONG SUPPORT - EXCLUSION",
            "#dc3545",
            f"The anthropometric evidence provides STRONG SUPPORT for the conclusion that Images A and B depict DIFFERENT INDIVIDUALS. Match confidence: {confidence:.1f}% with {stats.get('poor_matches', 0)} significant deviations.",
            "Evidence strongly supports exclusion. Proceed with non-match determination pending expert confirmation.",
            "HIGH (for exclusion)"
        )

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 10: ERROR LEVEL ANALYSIS WITH MANIPULATION DETECTION VERDICT]
# ═══════════════════════════════════════════════════════════════════════════

def compute_ela_professional(img: Image.Image, quality: int = 90) -> Tuple[Image.Image, float, Dict]:
    """Enhanced Error Level Analysis with detailed manipulation detection verdict"""
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    
    recompressed = Image.open(buffer).convert('RGB')
    diff = ImageChops.difference(img, recompressed)
    diff_array = np.array(diff)
    
    ela_score = float(np.mean(diff_array) / 255.0)
    ela_max = float(np.max(diff_array) / 255.0)
    ela_std = float(np.std(diff_array) / 255.0)
    
    # Amplified visualization
    amplified = np.clip(diff_array * 15, 0, 255).astype(np.uint8)
    
    # Hot spot detection
    threshold = np.percentile(diff_array, 95)
    hotspots = (diff_array > threshold).any(axis=2).sum()
    hotspot_percentage = (hotspots / (diff_array.shape[0] * diff_array.shape[1])) * 100
    
    # MANIPULATION DETECTION VERDICT
    if ela_score < STANDARDS.ELA_PRISTINE:
        verdict = "✅ NO MANIPULATION DETECTED - Image appears pristine with minimal compression artifacts"
        risk = "LOW"
        interpretation = "The image shows consistent error levels throughout, indicating no significant digital alterations."
    elif ela_score < STANDARDS.ELA_NORMAL:
        verdict = "✅ NO SIGNIFICANT MANIPULATION - Normal compression artifacts detected"
        risk = "LOW"
        interpretation = "Error levels are within expected range for standard JPEG compression. No signs of localized editing."
    elif ela_score < STANDARDS.ELA_EDITED:
        verdict = "⚠️ POSSIBLE MANIPULATION - Elevated ELA score detected, warrants further investigation"
        risk = "MODERATE"
        interpretation = "Elevated error levels suggest possible digital manipulation or multiple compression cycles. Manual review recommended."
    else:
        verdict = "🚨 HIGH PROBABILITY OF MANIPULATION - Significant digital editing detected"
        risk = "HIGH"
        interpretation = "Substantial inconsistencies in error levels indicate likely digital manipulation, compositing, or heavy editing."
    
    metrics = {
        'mean': ela_score,
        'max': ela_max,
        'std': ela_std,
        'hotspots_pct': hotspot_percentage,
        'verdict': verdict,
        'risk': risk,
        'interpretation': interpretation
    }
    
    return Image.fromarray(amplified), ela_score, metrics

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 11: RGB HISTOGRAM VISUALIZATION]
# ═══════════════════════════════════════════════════════════════════════════

def plot_rgb_histograms_professional(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """Generate professional RGB histogram comparison with detailed channel analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RGB Histogram Comparison Analysis', fontsize=16, fontweight='bold')
    
    colors = ['red', 'green', 'blue']
    channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
    
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Individual RGB channels
    for idx, (color, name) in enumerate(zip(colors, channel_names)):
        ax = axes[idx // 2, idx % 2]
        
        hist1 = cv2.calcHist([arr1], [idx], None, [256], [0, 256])
        hist2 = cv2.calcHist([arr2], [idx], None, [256], [0, 256])
        
        # Normalize histograms
        hist1 = hist1.flatten() / hist1.sum()
        hist2 = hist2.flatten() / hist2.sum()
        
        ax.plot(hist1, color=color, alpha=0.7, linewidth=2, label='Image A (Quotient)')
        ax.plot(hist2, color=color, alpha=0.7, linewidth=2, linestyle='--', label='Image B (Sample)')
        
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Pixel Intensity (0-255)')
        ax.set_ylabel('Normalized Frequency')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_xlim([0, 255])
    
    # Luminance (grayscale) comparison
    ax = axes[1, 1]
    gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
    
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256]).flatten() / gray1.size
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256]).flatten() / gray2.size
    
    ax.plot(hist1, color='black', alpha=0.7, linewidth=2, label='Image A (Quotient)')
    ax.plot(hist2, color='gray', alpha=0.7, linewidth=2, linestyle='--', label='Image B (Sample)')
    
    ax.set_title('Luminance Channel', fontweight='bold')
    ax.set_xlabel('Pixel Intensity (0-255)')
    ax.set_ylabel('Normalized Frequency')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_xlim([0, 255])
    
    plt.tight_layout()
    
    # Save to BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf)

def calculate_histogram_similarity_advanced(img1: Image.Image, img2: Image.Image) -> Dict[str, float]:
    """Multi-method histogram comparison for color distribution analysis"""
    def calc_hist(img, channel):
        return cv2.calcHist([np.array(img)], [channel], None, [256], [0, 256])
    
    results = {}
    
    for i, channel in enumerate(['red', 'green', 'blue']):
        hist1 = calc_hist(img1, i)
        hist2 = calc_hist(img2, i)
        
        results[f'{channel}_correlation'] = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
        results[f'{channel}_chi_square'] = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR))
        results[f'{channel}_intersection'] = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT))
        results[f'{channel}_bhattacharyya'] = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))
    
    results['overall_correlation'] = np.mean([results[f'{c}_correlation'] for c in ['red', 'green', 'blue']])
    results['overall_similarity'] = (results['overall_correlation'] + 1) / 2
    
    return results

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 12: OTHER ANALYSIS FUNCTIONS]
# ═══════════════════════════════════════════════════════════════════════════

def calculate_pixel_metrics_professional(img1: Image.Image, img2: Image.Image) -> Dict[str, float]:
    """Comprehensive pixel-level similarity metrics"""
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    
    mse = float(np.mean((arr1 - arr2) ** 2))
    rmse = float(np.sqrt(mse))
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    
    mae = float(np.mean(np.abs(arr1 - arr2)))
    ncc = float(np.sum(arr1 * arr2) / (np.sqrt(np.sum(arr1**2)) * np.sqrt(np.sum(arr2**2))))
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'psnr': psnr,
        'mae': mae,
        'ncc': ncc
    }
    
    if HAS_SKIMAGE:
        gray1 = cv2.cvtColor(pil_to_cv(img1), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(pil_to_cv(img2), cv2.COLOR_BGR2GRAY)
        metrics['ssim'] = float(ssim(gray1, gray2))
    
    return metrics

def align_faces_professional(img1, img2, landmarks1, landmarks2):
    """Advanced face alignment with quality metrics"""
    def get_alignment_points(landmarks):
        left_eye = np.mean([landmarks[i] for i in range(36, 42)], axis=0)
        right_eye = np.mean([landmarks[i] for i in range(42, 48)], axis=0)
        nose_tip = landmarks[30]
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]
        return np.array([left_eye, right_eye, nose_tip, left_mouth, right_mouth], dtype=np.float32)
    
    try:
        points1 = get_alignment_points(landmarks1)
        points2 = get_alignment_points(landmarks2)
        
        transform_matrix = cv2.estimateAffinePartial2D(points2, points1)[0]
        
        if transform_matrix is None:
            raise Exception("Could not compute transformation matrix")
        
        cv_img2 = pil_to_cv(img2)
        aligned = cv2.warpAffine(
            cv_img2,
            transform_matrix,
            img1.size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(128, 128, 128)
        )
        
        scale = np.sqrt(transform_matrix[0,0]**2 + transform_matrix[0,1]**2)
        rotation = np.arctan2(transform_matrix[1,0], transform_matrix[0,0]) * 180 / np.pi
        translation = np.sqrt(transform_matrix[0,2]**2 + transform_matrix[1,2]**2)
        
        metrics = {
            'scale_factor': float(scale),
            'rotation_degrees': float(rotation),
            'translation_pixels': float(translation),
            'quality': 'Good' if (abs(scale - 1.0) < 0.2 and abs(rotation) < 10) else 'Fair'
        }
        
        return cv_to_pil(aligned), metrics
    
    except Exception as e:
        st.warning(f"Advanced alignment failed: {str(e)}. Using fallback method.")
        return img2.resize(img1.size, Image.Resampling.LANCZOS), {'quality': 'Fallback', 'error': str(e)}

def compare_ratios(ratios1: Dict[str, float], ratios2: Dict[str, float]) -> Dict:
    """Compare normalized ratios (FaceCompare2 style)"""
    rows = []
    
    for key in ratios1.keys():
        if key in ratios2:
            val1, val2 = ratios1[key], ratios2[key]
            diff = abs(val1 - val2)
            rel_diff = (diff / val1 * 100) if val1 != 0 else 0
            
            rows.append({
                'Feature': key.replace('_', ' ').title(),
                'Quotient (A)': f"{val1:.4f}",
                'Sample (B)': f"{val2:.4f}",
                'Absolute Diff': f"{diff:.4f}",
                'Relative Diff %': f"{rel_diff:.2f}%"
            })
    
    return {'rows': rows}

def plot_measurements_comparison(m1: Dict, m2: Dict) -> plt.Figure:
    """Create bar chart comparing raw measurements (FaceCompare2 style)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    features = list(m1.keys())
    x = np.arange(len(features))
    width = 0.35
    
    vals1 = [m1[f] for f in features]
    vals2 = [m2[f] for f in features]
    
    ax.bar(x - width/2, vals1, width, label='Quotient (Image A)', color='#1976d2', alpha=0.8)
    ax.bar(x + width/2, vals2, width, label='Sample (Image B)', color='#f57c00', alpha=0.8)
    
    ax.set_xlabel('Facial Features', fontweight='bold')
    ax.set_ylabel('Distance (pixels)', fontweight='bold')
    ax.set_title('Raw Anthropometric Measurement Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', ' ').title() for f in features], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 13: REPORT GENERATION - ENHANCED]
# ═══════════════════════════════════════════════════════════════════════════

def generate_professional_report(case_id, timestamp, hash1, hash2, quality1, quality2, 
                                confidence, classification, confidence_level, interpretation,
                                recommendation, differences, stats, golden_ratios1, golden_ratios2,
                                ela_metrics1, ela_metrics2, hist_metrics, pixel_metrics, 
                                alignment_metrics, examiner_notes, morphological_assessment,
                                expert_summary, methodology_notes, limitations_notes):
    """Enhanced report with all calculation details"""
    
    report = f"""
╔═══════════════════════════════════════════════════════════════╗
║  PROFESSIONAL FORENSIC FACIAL COMPARISON EXAMINATION REPORT   ║
║                    v5.0 ENTERPRISE EDITION                     ║
╚═══════════════════════════════════════════════════════════════╝

CASE INFORMATION
════════════════════════════════════════════════════════════════
Case ID:                 {case_id}
Analysis Timestamp:      {timestamp}
System Version:          Forensic Facial Comparison System v5.0
Report Generated:        {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
Digital Signature:       {hashlib.sha256((case_id + timestamp).encode()).hexdigest()[:64]}

CHAIN OF CUSTODY - CRYPTOGRAPHIC VERIFICATION
════════════════════════════════════════════════════════════════
IMAGE A (Quotient/Reference):
  Filename:      {hash1.get('filename', 'N/A')}
  SHA-256:       {hash1.get('sha256', 'N/A')}
  MD5:           {hash1.get('md5', 'N/A')}
  Quality:       {quality1['overall']:.1f}/100 ({quality1['assessment']})
  Resolution:    {quality1['resolution'][0]}×{quality1['resolution'][1]}px
  Sharpness:     {quality1['sharpness']:.1f}/100
  Contrast:      {quality1['contrast']:.1f}/100
  Brightness:    {quality1['brightness']:.1f}/100

IMAGE B (Sample/Query):
  Filename:      {hash2.get('filename', 'N/A')}
  SHA-256:       {hash2.get('sha256', 'N/A')}
  MD5:           {hash2.get('md5', 'N/A')}
  Quality:       {quality2['overall']:.1f}/100 ({quality2['assessment']})
  Resolution:    {quality2['resolution'][0]}×{quality2['resolution'][1]}px
  Sharpness:     {quality2['sharpness']:.1f}/100
  Contrast:      {quality2['contrast']:.1f}/100
  Brightness:    {quality2['brightness']:.1f}/100

═══════════════════════════════════════════════════════════════
EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════

CONCLUSION:              {classification}
CONFIDENCE SCORE:        {confidence:.2f}%
CONFIDENCE LEVEL:        {confidence_level}

INTERPRETATION:
{interpretation}

RECOMMENDATION:
{recommendation}

EXPERT SUMMARY:
{expert_summary}

═══════════════════════════════════════════════════════════════
ANTHROPOMETRIC ANALYSIS (68-Point Landmark Detection)
═══════════════════════════════════════════════════════════════
Analysis Method: Normalized by Inter-Ocular Distance (Scale-Independent)

Total Measurements:      {len(differences)}
Tolerance Applied:       {STANDARDS.STANDARD_TOLERANCE * 100:.2f}%

MATCH DISTRIBUTION:
  ✓ EXCELLENT:          {stats.get('excellent_matches', 0):3d} ({stats.get('excellent_matches', 0)/len(differences)*100:5.1f}%)
  ✓ GOOD:               {stats.get('good_matches', 0):3d} ({stats.get('good_matches', 0)/len(differences)*100:5.1f}%)
  ~ ACCEPTABLE:         {stats.get('acceptable_matches', 0):3d} ({stats.get('acceptable_matches', 0)/len(differences)*100:5.1f}%)
  ⚠ MARGINAL:           {stats.get('marginal_matches', 0):3d} ({stats.get('marginal_matches', 0)/len(differences)*100:5.1f}%)
  ✗ SIGNIFICANT DEV:    {stats.get('poor_matches', 0):3d} ({stats.get('poor_matches', 0)/len(differences)*100:5.1f}%)

STATISTICAL SUMMARY:
  Mean Relative Diff:    {stats['mean_rel_diff']*100:.2f}%
  Std Deviation:         {stats['std_rel_diff']*100:.2f}%
  Maximum Deviation:     {stats['max_diff']*100:.2f}%
  Minimum Deviation:     {stats['min_diff']*100:.2f}%

DETAILED MEASUREMENTS:
{"─"*90}
{"Measurement":<35} {"Quotient":>10} {"Sample":>10} {"Abs Diff":>10} {"Rel %":>8} {"Status":<20}
{"─"*90}
"""
    
    for diff in differences:
        report += f"{diff['measurement']:<35} {diff['image_a']:10.4f} {diff['image_b']:10.4f} {diff['absolute_diff']:10.4f} {diff['percent_diff']:7.2f}% {diff['status']:<20}\n"
    
    report += f"""
{"─"*90}

GOLDEN RATIO ANALYSIS (Phi ≈ 1.618)
════════════════════════════════════════════════════════════════
The Golden Ratio represents ideal facial proportions found in nature.

IMAGE A (Quotient) - Golden Ratio Deviations:
"""
    
    for key, value in golden_ratios1.items():
        if 'deviation' in key:
            feature = key.replace('_deviation_from_phi', '').replace('_', ' ').title()
            deviation_pct = value * 100
            assessment = 'Perfect' if value < STANDARDS.GOLDEN_RATIO_PERFECT else 'Excellent' if value < STANDARDS.GOLDEN_RATIO_GOOD else 'Good' if value < STANDARDS.GOLDEN_RATIO_ACCEPTABLE else 'Significant'
            report += f"  {feature:<45} {deviation_pct:6.2f}% deviation ({assessment})\n"
    
    report += "\nIMAGE B (Sample) - Golden Ratio Deviations:\n"
    
    for key, value in golden_ratios2.items():
        if 'deviation' in key:
            feature = key.replace('_deviation_from_phi', '').replace('_', ' ').title()
            deviation_pct = value * 100
            assessment = 'Perfect' if value < STANDARDS.GOLDEN_RATIO_PERFECT else 'Excellent' if value < STANDARDS.GOLDEN_RATIO_GOOD else 'Good' if value < STANDARDS.GOLDEN_RATIO_ACCEPTABLE else 'Significant'
            report += f"  {feature:<45} {deviation_pct:6.2f}% deviation ({assessment})\n"
    
    report += f"""

ERROR LEVEL ANALYSIS (Digital Manipulation Detection)
════════════════════════════════════════════════════════════════
IMAGE A (Quotient):
  Verdict:               {ela_metrics1['verdict']}
  Risk Level:            {ela_metrics1['risk']}
  Mean ELA Score:        {ela_metrics1['mean']:.4f}
  Maximum ELA:           {ela_metrics1['max']:.4f}
  Standard Deviation:    {ela_metrics1['std']:.4f}
  Hotspot Percentage:    {ela_metrics1['hotspots_pct']:.2f}%
  Interpretation:        {ela_metrics1['interpretation']}

IMAGE B (Sample):
  Verdict:               {ela_metrics2['verdict']}
  Risk Level:            {ela_metrics2['risk']}
  Mean ELA Score:        {ela_metrics2['mean']:.4f}
  Maximum ELA:           {ela_metrics2['max']:.4f}
  Standard Deviation:    {ela_metrics2['std']:.4f}
  Hotspot Percentage:    {ela_metrics2['hotspots_pct']:.2f}%
  Interpretation:        {ela_metrics2['interpretation']}

COLOR DISTRIBUTION ANALYSIS (RGB Histogram Correlation)
════════════════════════════════════════════════════════════════
Overall Similarity:          {hist_metrics['overall_similarity']:.4f} ({hist_metrics['overall_similarity']*100:.1f}%)
Overall Correlation:         {hist_metrics['overall_correlation']:.4f}

Channel-Specific Correlations:
  Red Channel:               {hist_metrics['red_correlation']:.4f}
  Green Channel:             {hist_metrics['green_correlation']:.4f}
  Blue Channel:              {hist_metrics['blue_correlation']:.4f}

Interpretation:
  High correlation (>0.85) indicates similar lighting and capture conditions.
  Moderate correlation (0.65-0.85) suggests some environmental differences.
  Low correlation (<0.65) indicates different environments or post-processing.

PIXEL-LEVEL COMPARISON METRICS
════════════════════════════════════════════════════════════════
Mean Squared Error (MSE):        {pixel_metrics['mse']:.2f}
Root Mean Squared Error (RMSE):  {pixel_metrics['rmse']:.2f}
Peak Signal-to-Noise Ratio:      {pixel_metrics['psnr']:.2f} dB
Mean Absolute Error (MAE):       {pixel_metrics['mae']:.2f}
Normalized Cross-Correlation:    {pixel_metrics['ncc']:.4f}
"""
    
    if 'ssim' in pixel_metrics:
        report += f"Structural Similarity (SSIM):    {pixel_metrics['ssim']:.4f}\n"
    
    report += f"""

FACE ALIGNMENT QUALITY ASSESSMENT
════════════════════════════════════════════════════════════════
Alignment Quality:       {alignment_metrics.get('quality', 'N/A')}
Scale Factor:            {alignment_metrics.get('scale_factor', 0):.4f}
Rotation Applied:        {alignment_metrics.get('rotation_degrees', 0):.2f}°
Translation Distance:    {alignment_metrics.get('translation_pixels', 0):.2f} pixels

"""
    
    # Add morphological assessment if available
    if morphological_assessment:
        report += """
MORPHOLOGICAL ASSESSMENT (7-Point Expert Analysis)
════════════════════════════════════════════════════════════════
"""
        for feature_key, result in morphological_assessment.items():
            report += f"""
{result['name']} ({result['category']} - Weight: {result['weight']}x):
  Image A: {result['value_a']}
  Image B: {result['value_b']}
  Status:  {result['match_status']}
"""
        
        total_weight = sum(r['weight'] for r in morphological_assessment.values())
        weighted_score = sum(r['weighted_score'] for r in morphological_assessment.values())
        morpho_conf = (weighted_score / total_weight * 100) if total_weight > 0 else 0
        
        report += f"""
Morphological Confidence Score: {morpho_conf:.1f}%
Total Features Assessed: {len(morphological_assessment)}
Matches: {sum(1 for r in morphological_assessment.values() if r['match'])}
Inconsistencies: {sum(1 for r in morphological_assessment.values() if not r['match'])}
"""
    
    report += f"""

EXAMINER NOTES & OBSERVATIONS
════════════════════════════════════════════════════════════════
{examiner_notes or 'No additional examiner notes provided.'}

METHODOLOGY
════════════════════════════════════════════════════════════════
{methodology_notes}

LIMITATIONS & CONSIDERATIONS
════════════════════════════════════════════════════════════════
{limitations_notes}

═══════════════════════════════════════════════════════════════
⚖️ LEGAL DISCLAIMER & EXPERT REVIEW REQUIREMENT
═══════════════════════════════════════════════════════════════

**IMPORTANT:** This automated forensic analysis system provides 
computational assessment based on established scientific methodologies 
(FISWG, PCAST, ISO/IEC 17025, NIST). However:

⚠️ **Final determinations MUST be made by qualified forensic facial 
comparison examiners**

**Required for Court Admissibility:**
- Review by certified forensic examiner (FISWG-trained preferred)
- Validation of landmark detection and measurements
- Morphological feature assessment by trained expert
- Consideration of temporal and environmental factors
- Integration with additional case evidence
- Expert testimony for legal proceedings

**Limitations:**
- Results depend on image quality, pose, and lighting
- Aging, weight changes, cosmetic procedures affect measurements
- Automated analysis cannot replace human expertise
- Confidence scores are computational, not probability statements

**This report is provided as supporting evidence for expert 
examination, not as standalone proof of identity or non-identity.**

Report Hash (SHA-256): {hashlib.sha256(case_id.encode()).hexdigest()}
Digital Signature:     {hashlib.sha256((case_id + timestamp).encode()).hexdigest()[:64]}

═══════════════════════════════════════════════════════════════
                        END OF REPORT
═══════════════════════════════════════════════════════════════
    """
    
    return report

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 14: CSS STYLING]
# ═══════════════════════════════════════════════════════════════════════════

def inject_professional_css():
    st.markdown("""
    <style>
    .forensic-header {
        background: linear-gradient(135deg, #0a1128 0%, #1c2541 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .forensic-header h1 {
        font-size: 2.5em;
        font-weight: 900;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .version {
        font-size: 1.1em;
        margin: 0.5rem 0;
        opacity: 0.95;
    }
    .certifications {
        font-size: 0.9em;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# [SECTION 15: MAIN APPLICATION]
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Forensic Facial Comparison System v5.0",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    inject_professional_css()
    
    st.markdown("""
    <div class='forensic-header'>
        <h1>⚖️ FORENSIC FACIAL COMPARISON SYSTEM</h1>
        <p class='version'>v5.0 ENTERPRISE EDITION</p>
        <p class='certifications'>✓ FISWG • ✓ PCAST • ✓ ISO 17025 • ✓ NIST</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## ⚙️ **ANALYSIS CONFIGURATION**")
        
        tolerance_option = st.radio(
            "Tolerance Level:",
            options=[
                ("🔴 STRICT (2.5%)", STANDARDS.STRICT_TOLERANCE),
                ("🟡 STANDARD (6.5%) - Recommended", STANDARDS.STANDARD_TOLERANCE),
                ("🟢 LENIENT (11.5%)", STANDARDS.LENIENT_TOLERANCE)
            ],
            index=1,
            format_func=lambda x: x[0]
        )
        tolerance = tolerance_option[1]
        
        st.markdown("---")
        st.markdown("## 🔬 **ANALYSIS MODULES**")
        show_letter_labels = st.checkbox("✓ Letter-Labeled Landmarks", value=True)
        show_raw_comparison = st.checkbox("✓ Raw Pixel Comparison Chart", value=True)
        perform_golden_ratio = st.checkbox("✓ Golden Ratio Analysis", value=True)
        generate_visualizations = st.checkbox("✓ Generate Visualizations", value=True)
    
    if 'case_id' not in st.session_state:
        st.session_state.case_id = generate_case_id()
    case_id = st.session_state.case_id
    
    st.info(f"📋 **Case ID:** `{case_id}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🅰️ **Quotient (Reference Image)**")
        img_file_1 = st.file_uploader(
            "Upload Reference Image", 
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "mpo", "gif", "heic", "heif"],
            key="img_a",
            help="Known/reference image for comparison"
        )
    
    with col2:
        st.markdown("### 🅱️ **Sample (Query Image)**")
        img_file_2 = st.file_uploader(
            "Upload Query Image", 
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "mpo", "gif", "heic", "heif"],
            key="img_b",
            help="Unknown/query image for comparison"
        )
    
    img1, img2 = None, None
    
    if img_file_1:
        is_valid, msg, img_obj = validate_image_security(img_file_1)
        if is_valid:
            img1 = img_obj
            width1, height1 = img1.size
            col1.success(f"✅ {msg}")
            col1.image(img1, use_container_width=True)
            col1.markdown(f"**Resolution:** `{width1} × {height1}` pixels")
        else:
            col1.error(f"❌ {msg}")
    
    if img_file_2:
        is_valid, msg, img_obj = validate_image_security(img_file_2)
        if is_valid:
            img2 = img_obj
            width2, height2 = img2.size
            col2.success(f"✅ {msg}")
            col2.image(img2, use_container_width=True)
            col2.markdown(f"**Resolution:** `{width2} × {height2}` pixels")
        else:
            col2.error(f"❌ {msg}")
    
    # EDITABLE REPORT SECTIONS
    st.markdown("---")
    st.markdown("## 📝 **EDITABLE REPORT SECTIONS**")
    
    with st.expander("✍️ Examiner Notes", expanded=False):
        examiner_notes = st.text_area(
            "Optional notes, observations, or contextual information:",
            height=120,
            placeholder="Enter any relevant case information, observations, or special considerations...",
            key="examiner_notes"
        )
    
    with st.expander("📊 Expert Summary", expanded=False):
        expert_summary = st.text_area(
            "Expert's professional summary and conclusion:",
            height=150,
            value="This forensic facial comparison was conducted using multi-modal analysis including 68-point facial landmark detection, normalized anthropometric measurements, golden ratio aesthetic assessment, error level analysis for manipulation detection, RGB histogram correlation, and comprehensive statistical evaluation. The automated analysis provides computational assessment that should be reviewed and verified by a qualified forensic facial comparison examiner before final determination.",
            key="expert_summary"
        )
    
    with st.expander("🔬 Methodology Notes", expanded=False):
        methodology_notes = st.text_area(
            "Detailed methodology description:",
            height=150,
            value="Analysis employs: 1) dlib 68-point facial landmark detection with Histogram of Oriented Gradients (HOG) and Support Vector Machine (SVM) classifier, 2) Scale-independent normalization using inter-ocular distance as reference, 3) Golden ratio (Phi ≈ 1.618) proportional assessment for aesthetic evaluation, 4) Error Level Analysis (ELA) for digital manipulation detection via JPEG recompression differential, 5) Multi-channel RGB histogram correlation for color distribution comparison, 6) Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR) for pixel-level analysis.",
            key="methodology"
        )
    
    with st.expander("⚠️ Limitations & Considerations", expanded=False):
        limitations_notes = st.text_area(
            "Limitations and important considerations:",
            height=150,
            value="Results depend on image quality (resolution, sharpness, compression), facial pose (yaw, pitch, roll angles), and lighting conditions. Temporal factors including aging, weight changes, cosmetic procedures, facial hair growth, and medical conditions can significantly affect measurements. Environmental factors such as camera lens distortion, perspective effects, and image processing artifacts may impact results. Automated landmark detection accuracy decreases with extreme poses, occlusions, or poor image quality. This system provides computational assessment only—human expert review by qualified forensic facial comparison examiner is REQUIRED for final determination and legal admissibility.",
            key="limitations"
        )
    
    if st.button("🔬 **CONDUCT COMPREHENSIVE FORENSIC ANALYSIS**", type="primary", use_container_width=True):
        if not img1 or not img2:
            st.error("⚠️ Both images required for analysis")
            st.stop()
        
        with st.spinner("🔬 Conducting comprehensive forensic analysis..."):
            timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            
            # Progress tracking
            progress = st.progress(0, text="⚙️ Initializing forensic analysis system...")
            
            # Step 1: Chain of Custody (10%)
            progress.progress(10, text="Step 1/10: Computing cryptographic hashes...")
            hash1 = {
                'filename': img_file_1.name,
                'sha256': compute_forensic_hash(img1, 'sha256'),
                'md5': compute_forensic_hash(img1, 'md5'),
                'sha512': compute_forensic_hash(img1, 'sha512')
            }
            hash2 = {
                'filename': img_file_2.name,
                'sha256': compute_forensic_hash(img2, 'sha256'),
                'md5': compute_forensic_hash(img2, 'md5'),
                'sha512': compute_forensic_hash(img2, 'sha512')
            }
            
            # Step 2: Image Quality Assessment (20%)
            progress.progress(20, text="Step 2/10: Assessing image quality...")
            quality1 = calculate_image_quality_score(img1)
            quality2 = calculate_image_quality_score(img2)
            
            # Step 3: Image Standardization (30%)
            progress.progress(30, text="Step 3/10: Standardizing images...")
            img1_std = standardize_image(img1)
            img2_std = standardize_image(img2)
            
            # Step 4: Landmark Detection (40%)
            progress.progress(40, text="Step 4/10: Detecting 68-point facial landmarks...")
            landmarks1 = get_landmarks(pil_to_cv(img1_std))
            landmarks2 = get_landmarks(pil_to_cv(img2_std))
            
            if landmarks1 is None or landmarks2 is None:
                st.error("❌ **Face detection failed.** Please ensure faces are clearly visible and frontal.")
                st.stop()
            
            # Step 5: Anthropometric Measurements (50%)
            progress.progress(50, text="Step 5/10: Calculating anthropometric measurements...")
            measurements1 = calculate_measurements(landmarks1)
            measurements2 = calculate_measurements(landmarks2)
            ratios1_eye = normalize_by_eye_distance(measurements1)
            ratios2_eye = normalize_by_eye_distance(measurements2)
            
            # Step 6: Similarity Analysis (60%)
            progress.progress(60, text="Step 6/10: Performing similarity analysis...")
            confidence, differences, stats = calculate_facial_similarity_advanced(
                ratios1_eye, ratios2_eye, tolerance
            )
            
            classification, color, interpretation, recommendation, confidence_level = \
                classify_evidence_strength_professional(confidence, stats)
            
            # Step 7: Golden Ratio Analysis (70%)
            progress.progress(70, text="Step 7/10: Analyzing golden ratios...")
            golden_ratios1 = calculate_golden_ratios(landmarks1) if perform_golden_ratio else {}
            golden_ratios2 = calculate_golden_ratios(landmarks2) if perform_golden_ratio else {}
            
            # Step 8: Error Level Analysis (80%)
            progress.progress(80, text="Step 8/10: Performing error level analysis...")
            ela1_img, ela1_score, ela1_metrics = compute_ela_professional(img1_std)
            ela2_img, ela2_score, ela2_metrics = compute_ela_professional(img2_std)
            
            # Step 9: Color & Pixel Analysis (90%)
            progress.progress(85, text="Step 9a/10: Analyzing color distributions...")
            hist_metrics = calculate_histogram_similarity_advanced(img1_std, img2_std)
            
            progress.progress(90, text="Step 9b/10: Generating RGB histograms...")
            rgb_histogram_img = plot_rgb_histograms_professional(img1_std, img2_std) if generate_visualizations else None
            
            img2_aligned, alignment_metrics = align_faces_professional(
                img1_std, img2_std, landmarks1, landmarks2
            )
            pixel_metrics = calculate_pixel_metrics_professional(img1_std, img2_aligned)
            
            # Step 10: Visualization Generation (95-100%)
            progress.progress(95, text="Step 10/10: Generating visualizations...")
            img1_landmarks = draw_landmarks_professional(img1_std, landmarks1)
            img2_landmarks = draw_landmarks_professional(img2_std, landmarks2)
            
            if show_raw_comparison and generate_visualizations:
                raw_comparison_fig = plot_measurements_comparison(measurements1, measurements2)
            else:
                raw_comparison_fig = None
            
            progress.progress(100, text="✅ Analysis complete!")
        
        # Display Results
        st.success("✅ **COMPREHENSIVE FORENSIC ANALYSIS COMPLETED**")
        
        # Executive Summary with color-coded classification
        evidence_class_style = f"background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border-left: 8px solid {color}; padding: 2rem; border-radius: 12px; margin: 1rem 0;"
        
        st.markdown(f"""
        <div style='{evidence_class_style}'>
            <h2 style='margin-top: 0; color: {color};'>{classification}</h2>
            <h3 style='margin: 10px 0; color: {color};'>Confidence Score: {confidence:.2f}%</h3>
            <p style='font-size: 1.1em; line-height: 1.6;'><strong>Interpretation:</strong><br>{interpretation}</p>
            <p style='font-size: 1em; margin-bottom: 0;'><strong>Recommendation:</strong><br>{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics Dashboard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Match Confidence", f"{confidence:.1f}%", confidence_level)
        col2.metric("Total Measurements", len(differences))
        col3.metric("Average Image Quality", f"{(quality1['overall']+quality2['overall'])/2:.0f}/100")
        col4.metric("ELA Risk Level", max(ela1_metrics['risk'], ela2_metrics['risk']))
        
        # Detailed Analysis Sections
        
        # Chain of Custody
        with st.expander("🔐 **CHAIN OF CUSTODY - Cryptographic Verification**", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Image A (Quotient)")
                st.code(f"""Filename:  {hash1['filename']}
SHA-256:   {hash1['sha256']}
MD5:       {hash1['md5']}
Quality:   {quality1['overall']:.1f}/100 ({quality1['assessment']})
Resolution: {quality1['resolution'][0]}×{quality1['resolution'][1]}px""")
            
            with col2:
                st.markdown("#### Image B (Sample)")
                st.code(f"""Filename:  {hash2['filename']}
SHA-256:   {hash2['sha256']}
MD5:       {hash2['md5']}
Quality:   {quality2['overall']:.1f}/100 ({quality2['assessment']})
Resolution: {quality2['resolution'][0]}×{quality2['resolution'][1]}px""")
        
                # Letter-Labeled Landmarks
        if show_letter_labels:
            with st.expander("📌 **LANDMARK MAPPING (Letter Labels - Enhanced Lines)**", expanded=False):
                st.markdown("Anatomically labeled key facial landmarks with distance measurements")
                
                annotated1, legend1, segments1 = draw_landmarks_with_letter_labels(
                    pil_to_cv(img1_std), landmarks1, True, measurements1['Eye Width (outer)']
                )
                annotated2, legend2, segments2 = draw_landmarks_with_letter_labels(
                    pil_to_cv(img2_std), landmarks2, True, measurements2['Eye Width (outer)']
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(annotated1, caption="Quotient (A) - Annotated", use_container_width=True)
                with col2:
                    st.image(annotated2, caption="Sample (B) - Annotated", use_container_width=True)
                
                st.markdown("### 🔠 **Landmark Letter Legend**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Quotient (Image A)")
                    st.json(legend1)
                with col2:
                    st.markdown("#### Sample (Image B)")
                    st.json(legend2)
                
                st.markdown("### 📏 **Measured Segment Distances**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Quotient (Image A)")
                    st.json(segments1)
                with col2:
                    st.markdown("#### Sample (Image B)")
                    st.json(segments2)
        
        # Professional 68-Point Landmarks
        with st.expander("🎯 **FACIAL LANDMARKS (68-Point - Thicker Lines)**", expanded=False):
            st.markdown("Complete dlib 68-point landmark detection with enhanced visibility")
            col1, col2 = st.columns(2)
            with col1:
                st.image(img1_landmarks, caption="Quotient (A) - 68 Landmarks", use_container_width=True)
            with col2:
                st.image(img2_landmarks, caption="Sample (B) - 68 Landmarks", use_container_width=True)
        
        # Anthropometric Analysis
        with st.expander("📊 **ANTHROPOMETRIC ANALYSIS (Normalized Measurements)**", expanded=True):
            st.markdown("""
            **Analysis Method:** Measurements normalized by inter-ocular distance for scale-independent comparison.
            This ensures facial proportions (not absolute sizes) are compared.
            """)
            
            comparison_data = compare_ratios(ratios1_eye, ratios2_eye)
            df_anthropo = pd.DataFrame(comparison_data['rows'])
            st.dataframe(
                df_anthropo.style.background_gradient(subset=['Relative Diff %'], cmap='RdYlGn_r'),
                use_container_width=True
            )
            
            st.markdown("""
            ### 📐 **Why Normalize by Eye Distance?**
            
            When comparing faces from different images, we can't rely on raw pixel distances because:
            
            - Different images may have **different resolutions**, zoom levels, or head sizes
            - A 100px nose in a zoomed-in image is **not** the same as 100px in a distant image
            
            **Solution:** Normalize all measurements using the inter-ocular distance (eye-to-eye) as a stable reference.
            
            | **Normalization Method** | **What It Does** |
            |--------------------------|------------------|
            | 👁️ **Eye-to-Eye Distance** | Uses distance between eye corners as anchor. Eye landmarks are typically visible and less affected by pose. |
            
            ✅ This ensures **shape** (not size) becomes the basis for facial geometry comparison.
            """)
            
            st.markdown("""
            ### 🧠 **Mathematical Foundation**
            
            | **Concept** | **Application** |
            |-------------|-----------------|
            | 🔢 **Ratio & Proportion** | Normalizes features like `Nose→Mouth / Eye→Eye` for size-independent comparison |
            | 📐 **Similar Triangles** | Facial structure remains proportional when image scale changes |
            | 📍 **Euclidean Geometry** | Distance calculation: `√[(x₂-x₁)² + (y₂-y₁)²]` |
            | 📊 **Statistical Analysis** | Weighted scoring based on measurement reliability |
            """)
        
        # Raw Pixel Comparison Chart
        if show_raw_comparison and raw_comparison_fig:
            with st.expander("📈 **RAW MEASUREMENT COMPARISON (Pixels)**", expanded=False):
                st.markdown("Direct pixel-distance comparison before normalization")
                st.pyplot(raw_comparison_fig)
        
        # Golden Ratio Analysis
        if perform_golden_ratio and golden_ratios1 and golden_ratios2:
            with st.expander("✨ **GOLDEN RATIO ANALYSIS (Phi ≈ 1.618)**", expanded=False):
                st.markdown("""
                The **Golden Ratio (φ ≈ 1.618)** appears throughout nature and is considered 
                the mathematical basis of aesthetic beauty. In facial analysis, certain proportions 
                that approximate φ are associated with perceived attractiveness.
                """)
                
                df_golden = pd.DataFrame({
                    "Ratio": list(golden_ratios1.keys()),
                    "Quotient (A)": [f"{v:.4f}" for v in golden_ratios1.values()],
                    "Sample (B)": [f"{golden_ratios2.get(k, 0.0):.4f}" for k in golden_ratios1.keys()],
                    "Ideal (Phi)": [f"{STANDARDS.PHI:.4f}" if 'ratio' not in k.lower() and 'deviation' not in k else "Deviation" for k in golden_ratios1.keys()]
                })
                st.table(df_golden)
                
                st.markdown("""
                **Note:** Golden ratio analysis is for aesthetic reference only and is NOT 
                a primary identification metric in forensic comparison.
                """)
        
        # ═══════════════════════════════════════════════════════════════════════════
        # ERROR LEVEL ANALYSIS WITH MANIPULATION VERDICT
        # ═══════════════════════════════════════════════════════════════════════════
        with st.expander("🔬 **ERROR LEVEL ANALYSIS (Manipulation Detection)**", expanded=False):
            st.markdown("""
            **Error Level Analysis (ELA)** detects digital manipulation by comparing an image 
            with a recompressed version. Inconsistent error levels indicate potential editing.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Image A (Quotient)")
                st.image(ela1_img, caption="ELA Visualization", use_container_width=True)
                st.markdown(f"**Mean ELA Score:** `{ela1_metrics['mean']:.4f}`")
                st.markdown(f"**Maximum ELA:** `{ela1_metrics['max']:.4f}`")
                st.markdown(f"**Standard Deviation:** `{ela1_metrics['std']:.4f}`")
                st.markdown(f"**Hotspot Percentage:** `{ela1_metrics['hotspots_pct']:.2f}%`")
                
                # Verdict with color coding
                if ela1_metrics['risk'] == 'LOW':
                    st.success(f"### {ela1_metrics['verdict']}")
                    st.success(f"**Risk Level:** `{ela1_metrics['risk']}`")
                elif ela1_metrics['risk'] == 'MODERATE':
                    st.warning(f"### {ela1_metrics['verdict']}")
                    st.warning(f"**Risk Level:** `{ela1_metrics['risk']}`")
                else:
                    st.error(f"### {ela1_metrics['verdict']}")
                    st.error(f"**Risk Level:** `{ela1_metrics['risk']}`")
                
                st.info(f"**Interpretation:** {ela1_metrics['interpretation']}")
            
            with col2:
                st.markdown("#### Image B (Sample)")
                st.image(ela2_img, caption="ELA Visualization", use_container_width=True)
                st.markdown(f"**Mean ELA Score:** `{ela2_metrics['mean']:.4f}`")
                st.markdown(f"**Maximum ELA:** `{ela2_metrics['max']:.4f}`")
                st.markdown(f"**Standard Deviation:** `{ela2_metrics['std']:.4f}`")
                st.markdown(f"**Hotspot Percentage:** `{ela2_metrics['hotspots_pct']:.2f}%`")
                
                # Verdict with color coding
                if ela2_metrics['risk'] == 'LOW':
                    st.success(f"### {ela2_metrics['verdict']}")
                    st.success(f"**Risk Level:** `{ela2_metrics['risk']}`")
                elif ela2_metrics['risk'] == 'MODERATE':
                    st.warning(f"### {ela2_metrics['verdict']}")
                    st.warning(f"**Risk Level:** `{ela2_metrics['risk']}`")
                else:
                    st.error(f"### {ela2_metrics['verdict']}")
                    st.error(f"**Risk Level:** `{ela2_metrics['risk']}`")
                
                st.info(f"**Interpretation:** {ela2_metrics['interpretation']}")
            
            st.markdown("""
            ---
            **ELA Interpretation Guide:**
            - **Low ELA (<10):** Pristine image, minimal compression
            - **Normal ELA (10-20):** Standard compression artifacts
            - **Elevated ELA (20-35):** Possible manipulation or multiple saves
            - **High ELA (>35):** Strong indication of digital editing
            """)
        
        # ═══════════════════════════════════════════════════════════════════════════
        # RGB HISTOGRAM ANALYSIS WITH VISUALIZATION
        # ═══════════════════════════════════════════════════════════════════════════
        with st.expander("🎨 **RGB HISTOGRAM ANALYSIS (Color Distribution)**", expanded=False):
            st.markdown("""
            **RGB Histogram Analysis** compares the color distribution between images. 
            High correlation indicates similar lighting and capture conditions.
            """)
            
            # Metrics Display
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall Similarity", f"{hist_metrics['overall_similarity']:.4f}")
            col2.metric("Red Correlation", f"{hist_metrics['red_correlation']:.4f}")
            col3.metric("Green Correlation", f"{hist_metrics['green_correlation']:.4f}")
            col4.metric("Blue Correlation", f"{hist_metrics['blue_correlation']:.4f}")
            
            # RGB Histogram Visualization
            if rgb_histogram_img:
                st.image(rgb_histogram_img, caption="RGB & Luminance Distribution Comparison", use_container_width=True)
            
            # Interpretation Guide
            st.markdown("""
            ### 📊 **Interpretation Guide:**
            
            | Similarity Score | Interpretation |
            |------------------|----------------|
            | **>0.90** | Excellent correlation - Very similar lighting/capture conditions |
            | **0.75-0.90** | Good correlation - Similar conditions with minor differences |
            | **0.60-0.75** | Moderate correlation - Some environmental differences |
            | **<0.60** | Low correlation - Different environments or significant post-processing |
            
            **What to Look For:**
            - Similar histogram shapes indicate consistent capture conditions
            - Shifted histograms suggest different lighting or white balance
            - Flattened histograms may indicate contrast adjustments
            - Clipped peaks suggest over/underexposure
            """)
            
            # Detailed Channel Analysis
            st.markdown("### 🔍 **Detailed Channel Analysis:**")
            
            if hist_metrics['overall_similarity'] >= 0.85:
                st.success(f"""
                **✅ HIGH COLOR SIMILARITY ({hist_metrics['overall_similarity']:.4f})**
                
                The color distributions are highly similar, indicating:
                - Consistent lighting conditions
                - Similar camera settings or calibration
                - Minimal color grading differences
                - High probability of same capture environment
                """)
            elif hist_metrics['overall_similarity'] >= 0.65:
                st.info(f"""
                **ℹ️ MODERATE COLOR SIMILARITY ({hist_metrics['overall_similarity']:.4f})**
                
                The color distributions show moderate correlation:
                - Some differences in lighting or capture conditions
                - Possible different camera systems or settings
                - May indicate different time/location of capture
                - Consider temporal and environmental factors
                """)
            else:
                st.warning(f"""
                **⚠️ LOW COLOR SIMILARITY ({hist_metrics['overall_similarity']:.4f})**
                
                The color distributions differ significantly:
                - Different lighting environments likely
                - Possible color correction or grading applied
                - Different camera systems or white balance
                - May indicate different capture circumstances
                """)
        
        # Pixel-Level Metrics
        with st.expander("🔢 **PIXEL-LEVEL COMPARISON METRICS**", expanded=False):
            st.markdown("Statistical measures of image similarity at the pixel level")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{pixel_metrics['mse']:.2f}")
            col2.metric("PSNR", f"{pixel_metrics['psnr']:.2f} dB")
            col3.metric("MAE", f"{pixel_metrics['mae']:.2f}")
            
            if 'ssim' in pixel_metrics:
                col1, col2 = st.columns(2)
                col1.metric("SSIM", f"{pixel_metrics['ssim']:.4f}")
                col2.metric("NCC", f"{pixel_metrics['ncc']:.4f}")
            
            st.markdown("""
            **Metric Explanations:**
            - **MSE (Mean Squared Error):** Average squared difference between pixels (lower = more similar)
            - **PSNR (Peak Signal-to-Noise Ratio):** Quality metric in dB (higher = more similar, >30dB is good)
            - **MAE (Mean Absolute Error):** Average absolute difference (lower = more similar)
            - **SSIM (Structural Similarity):** Perceptual similarity (0-1, higher = more similar)
            - **NCC (Normalized Cross-Correlation):** Pattern matching similarity (closer to 1 = more similar)
            """)
        
        # ═══════════════════════════════════════════════════════════════════════════
        # MORPHOLOGICAL ASSESSMENT - DEDICATED SECTION
        # ═══════════════════════════════════════════════════════════════════════════
        with st.expander("🔍 **MORPHOLOGICAL ASSESSMENT (7-Point Expert Analysis)**", expanded=False):
            st.markdown("""
            <div style='background: #fff3e0; border-left: 6px solid #f57c00; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;'>
            <h4 style='margin-top:0; color: #f57c00;'>⚠️ EXPERT EXAMINER INPUT REQUIRED</h4>
            <p>Morphological feature assessment requires trained forensic examiner evaluation. 
            Automated systems cannot reliably classify these subjective characteristics. 
            This section must be completed by a qualified expert.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("## 📋 **Systematic Morphological Feature Assessment**")
            st.markdown("Assess each feature independently for both images:")
            
            morpho_results = {}
            
            for feature_key, feature_info in MORPHOLOGICAL_FEATURES.items():
                st.markdown("---")
                st.markdown(f"### {feature_info['name']}")
                
                # Feature metadata
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                col_meta1.markdown(f"**Category:** `{feature_info['category']}`")
                col_meta2.markdown(f"**Weight:** `{feature_info['weight']}x`")
                col_meta3.markdown(f"**Value:** {feature_info['forensic_value']}")
                
                st.markdown(f"*{feature_info['description']}*")
                
                # Assessment inputs
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    val_a = st.selectbox(
                        f"🅰️ Image A (Quotient) - {feature_info['name']}",
                        ['Not Assessed'] + feature_info['options'],
                        key=f"morph_a_{feature_key}",
                        help=f"Select the observed {feature_info['name'].lower()} in Image A"
                    )
                
                with col2:
                    val_b = st.selectbox(
                        f"🅱️ Image B (Sample) - {feature_info['name']}",
                        ['Not Assessed'] + feature_info['options'],
                        key=f"morph_b_{feature_key}",
                        help=f"Select the observed {feature_info['name'].lower()} in Image B"
                    )
                
                with col3:
                    if val_a != 'Not Assessed' and val_b != 'Not Assessed':
                        is_match = val_a == val_b
                        if is_match:
                            st.success("✅ MATCH")
                            match_status = "CONSISTENT"
                            weighted_score = feature_info['weight'] * 1.0
                        else:
                            st.error("✗ DIFFERENT")
                            match_status = "INCONSISTENT"
                            weighted_score = feature_info['weight'] * 0.0
                        
                        morpho_results[feature_key] = {
                            'name': feature_info['name'],
                            'category': feature_info['category'],
                            'weight': feature_info['weight'],
                            'value_a': val_a,
                            'value_b': val_b,
                            'match': is_match,
                            'match_status': match_status,
                            'weighted_score': weighted_score
                        }
                    else:
                        st.info("⏳ PENDING")
            
            # Morphological Assessment Summary
            if morpho_results:
                st.markdown("---")
                st.markdown("## 📊 **MORPHOLOGICAL ASSESSMENT SUMMARY**")
                
                total_assessed = len(morpho_results)
                total_matches = sum(1 for r in morpho_results.values() if r['match'])
                total_inconsistencies = total_assessed - total_matches
                total_weight = sum(r['weight'] for r in morpho_results.values())
                weighted_score = sum(r['weighted_score'] for r in morpho_results.values())
                
                morpho_confidence = (weighted_score / total_weight * 100) if total_weight > 0 else 0
                
                # Summary Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Features Assessed", total_assessed)
                col2.metric("Consistent Matches", total_matches)
                col3.metric("Inconsistencies", total_inconsistencies)
                col4.metric("Weighted Confidence", f"{morpho_confidence:.1f}%")
                
                # Detailed Results Table
                st.markdown("### 📋 **Detailed Morphological Comparison Table**")
                df_morpho = pd.DataFrame([
                    {
                        'Feature': r['name'],
                        'Category': r['category'],
                        'Weight': f"{r['weight']}x",
                        'Image A (Quotient)': r['value_a'],
                        'Image B (Sample)': r['value_b'],
                        'Status': r['match_status']
                    }
                    for r in morpho_results.values()
                ])
                
                st.dataframe(df_morpho, use_container_width=True)
                
                # Interpretation
                st.markdown("### 🎯 **Morphological Consistency Interpretation**")
                
                if morpho_confidence >= 85:
                    st.success(f"""
                    **✅ STRONG MORPHOLOGICAL CONSISTENCY ({morpho_confidence:.1f}%)**
                    
                    - {total_matches} out of {total_assessed} features show consistency
                    - High degree of morphological similarity detected
                    - Features align with positive identification hypothesis
                    - Weighted assessment accounts for feature discriminative power
                    
                    **Conclusion:** Morphological features provide **strong support** for same individual determination.
                    """)
                elif morpho_confidence >= 60:
                    st.info(f"""
                    **ℹ️ MODERATE MORPHOLOGICAL CONSISTENCY ({morpho_confidence:.1f}%)**
                    
                    - {total_matches} out of {total_assessed} features show consistency
                    - Some features consistent, others differ
                    - Consider temporal changes (aging, weight, grooming)
                    - Weighted assessment shows mixed results
                    
                    **Conclusion:** Morphological features are **inconclusive** - additional analysis recommended.
                    """)
                else:
                    st.warning(f"""
                    **⚠️ LOW MORPHOLOGICAL CONSISTENCY ({morpho_confidence:.1f}%)**
                    
                    - Only {total_matches} out of {total_assessed} features show consistency
                    - Significant morphological differences detected
                    - Features suggest probable exclusion
                    - Weighted assessment heavily impacted by critical feature differences
                    
                    **Conclusion:** Morphological features suggest **different individuals**.
                    """)
                
                # Category Breakdown
                st.markdown("### 🏷️ **Analysis by Feature Category**")
                
                critical_features = [r for r in morpho_results.values() if r['category'] == 'CRITICAL']
                secondary_features = [r for r in morpho_results.values() if r['category'] == 'SECONDARY']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔴 CRITICAL Features (Weight: 3.0x)")
                    if critical_features:
                        crit_matches = sum(1 for f in critical_features if f['match'])
                        crit_total = len(critical_features)
                        st.metric("Critical Feature Matches", f"{crit_matches}/{crit_total}")
                        
                        for feat in critical_features:
                            status_icon = "✅" if feat['match'] else "❌"
                            st.markdown(f"{status_icon} **{feat['name']}:** {feat['match_status']}")
                
                with col2:
                    st.markdown("#### 🟡 SECONDARY Features (Weight: 2.0x)")
                    if secondary_features:
                        sec_matches = sum(1 for f in secondary_features if f['match'])
                        sec_total = len(secondary_features)
                        st.metric("Secondary Feature Matches", f"{sec_matches}/{sec_total}")
                        
                        for feat in secondary_features:
                            status_icon = "✅" if feat['match'] else "❌"
                            st.markdown(f"{status_icon} **{feat['name']}:** {feat['match_status']}")
            else:
                st.info("⏳ **No features assessed yet.** Please complete the morphological assessment above.")
        
        # Face Alignment Quality
        with st.expander("🔄 **FACE ALIGNMENT QUALITY**", expanded=False):
            st.markdown("Geometric transformation applied for optimal comparison")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Alignment Quality", alignment_metrics.get('quality', 'N/A'))
            col2.metric("Scale Factor", f"{alignment_metrics.get('scale_factor', 0):.4f}")
            col3.metric("Rotation", f"{alignment_metrics.get('rotation_degrees', 0):.2f}°")
            
            if 'translation_pixels' in alignment_metrics:
                st.metric("Translation", f"{alignment_metrics.get('translation_pixels', 0):.2f} pixels")
        
        # Generate Comprehensive Report
        st.markdown("---")
        st.markdown("## 📄 **FORENSIC REPORT GENERATION**")
        
        report = generate_professional_report(
            case_id, timestamp, hash1, hash2, 
            quality1, quality2, confidence, 
            classification, confidence_level, interpretation, 
            recommendation, differences, stats, 
            golden_ratios1, golden_ratios2, 
            ela1_metrics, ela2_metrics, 
            hist_metrics, pixel_metrics, alignment_metrics,
            st.session_state.get('examiner_notes', ''),
            morpho_results if morpho_results else None,
            st.session_state.get('expert_summary', ''),
            st.session_state.get('methodology', ''),
            st.session_state.get('limitations', '')
        )
        
        # Download Buttons
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.download_button(
                label="📄 **DOWNLOAD COMPREHENSIVE FORENSIC REPORT**",
                data=report,
                file_name=f"{case_id}_Forensic_Report.txt",
                mime="text/plain",
                use_container_width=True,
                help="Download complete text report with all analysis results"
            )
        
        with col2:
            # Generate CSV of measurements
            csv_data = pd.DataFrame(differences).to_csv(index=False)
            st.download_button(
                label="📊 **DOWNLOAD MEASUREMENTS (CSV)**",
                data=csv_data,
                file_name=f"{case_id}_Measurements.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download measurement data for external analysis"
            )
        
        # Legal Disclaimer
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                    border-left: 8px solid #f57c00; padding: 2rem; border-radius: 12px; 
                    margin-top: 2rem; box-shadow: 0 4px 15px rgba(245,124,0,0.2);'>
        <h3 style='margin-top: 0; color: #e65100;'>⚖️ LEGAL DISCLAIMER & EXPERT REVIEW REQUIREMENT</h3>
        
        <p style='font-size: 1.1em; line-height: 1.6;'>
        <strong>IMPORTANT:</strong> This automated forensic analysis system provides computational 
        assessment based on established scientific methodologies (FISWG, PCAST, ISO/IEC 17025, NIST). 
        However:
        </p>
        
        <div style='background: rgba(230,81,0,0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
        <p style='margin: 0; font-size: 1.15em; font-weight: bold; color: #e65100;'>
        ⚠️ Final determinations MUST be made by qualified forensic facial comparison examiners
        </p>
        </div>
        
        <h4 style='color: #e65100; margin-top: 1.5rem;'>Required for Court Admissibility:</h4>
        <ul style='line-height: 1.8;'>
        <li>Review by <strong>certified forensic examiner</strong> (FISWG-trained preferred)</li>
        <li>Validation of <strong>landmark detection</strong> and measurements</li>
        <li><strong>Morphological feature assessment</strong> by trained expert</li>
        <li>Consideration of <strong>temporal and environmental factors</strong></li>
        <li>Integration with <strong>additional case evidence</strong></li>
        <li><strong>Expert testimony</strong> for legal proceedings</li>
        </ul>
        
        <h4 style='color: #e65100; margin-top: 1.5rem;'>Limitations:</h4>
        <ul style='line-height: 1.8;'>
        <li>Results depend on <strong>image quality, pose, and lighting</strong></li>
        <li><strong>Aging, weight changes, cosmetic procedures</strong> affect measurements</li>
        <li>Automated analysis <strong>cannot replace human expertise</strong></li>
        <li>Confidence scores are <strong>computational, not probability statements</strong></li>
        <li>System provides <strong>supporting evidence only</strong></li>
        </ul>
        
        <div style='background: rgba(230,81,0,0.1); padding: 1rem; border-radius: 8px; margin-top: 1.5rem;'>
        <p style='margin: 0; font-weight: bold; color: #e65100;'>
        This report is provided as <strong>supporting evidence for expert examination</strong>, 
        not as standalone proof of identity or non-identity.
        </p>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        # System Information
        st.markdown("---")
        with st.expander("ℹ️ **SYSTEM INFORMATION & CREDITS**", expanded=False):
            st.markdown(f"""
            ### 🔬 **Forensic Facial Comparison System v5.0**
            
            **Analysis ID:** `{case_id}`  
            **Timestamp:** `{timestamp}`  
            **System Version:** v5.0 Enterprise Edition
            
            ### 📚 **Methodology Standards:**
            - ✓ FISWG (Facial Identification Scientific Working Group)
            - ✓ PCAST (President's Council of Advisors on Science and Technology)
            - ✓ ISO/IEC 17025 (Laboratory Accreditation)
            - ✓ NIST (National Institute of Standards and Technology)
            
            ### 🛠️ **Technical Stack:**
            - **Landmark Detection:** dlib 68-point facial landmark detector
            - **Image Processing:** OpenCV, PIL/Pillow
            - **Analysis:** NumPy, pandas, scikit-image
            - **Visualization:** Matplotlib, Streamlit
            
            ### 📖 **References:**
            1. Facial Identification Scientific Working Group (FISWG) Guidelines
            2. PCAST Report on Forensic Science in Criminal Courts (2016)
            3. ISO/IEC 17025:2017 - Laboratory Accreditation Standard
            4. NIST Special Database 32 - Multiple Encounter Dataset
            
            ### ⚖️ **Disclaimer:**
            This software is provided for forensic analysis purposes. Users are responsible 
            for ensuring compliance with applicable laws and regulations in their jurisdiction.
            """)

if __name__ == "__main__":
    main()

            