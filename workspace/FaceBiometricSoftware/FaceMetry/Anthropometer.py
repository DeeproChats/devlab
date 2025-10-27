import math
import cv2 # type: ignore
import dlib # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# === Facial Landmark Mapping (Name → Dlib index) ===
LANDMARKS = {
    "left_eye": 36, "right_eye": 45,
    "nose_tip": 30, "mouth_center": 62,
    "left_ear": 0, "right_ear": 16,
    "chin": 8, "forehead_glabella": 27,
    "jaw_left": 3, "jaw_right": 13
}

# === Measurement Pairs (Extended for Golden Ratio) ===
MEASUREMENTS = {
    "Eye Width (outer)": ("left_eye", "right_eye"),
    "Nose Width": ("jaw_left", "jaw_right"),
    "Nose to Chin": ("nose_tip", "chin"),
    "Nose to Mouth": ("nose_tip", "mouth_center"),
    "Hairline to Nose": ("forehead_glabella", "nose_tip"),
    "Hairline to Chin": ("forehead_glabella", "chin"),
    "Pupil Line to Chin": ("left_eye", "chin"),
    "Eye to Nose Base": ("left_eye", "nose_tip"),
    "Mouth Width": ("jaw_left", "jaw_right")
}

# === Dlib Model Setup ===
MODEL_PATH = "dlibModel/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# === Landmark Extraction ===
def get_dlib_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    return {k: (shape.part(i).x, shape.part(i).y) for k, i in LANDMARKS.items()}

# === Euclidean Distance ===
def euclidean_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# === Pixel-Based Feature Measurements ===
def calculate_measurements(landmarks):
    result = {}
    for name, (p1, p2) in MEASUREMENTS.items():
        if p1 in landmarks and p2 in landmarks:
            result[name] = euclidean_distance(landmarks[p1], landmarks[p2])
    return result

# === Golden Ratio Metrics ===
def calculate_golden_ratios(measurements):
    ratios = {}
    try:
        ratios['(Hairline to Nose) / (Nose to Chin)'] = round(
            measurements['Hairline to Nose'] / measurements['Nose to Chin'], 4)
        ratios['(Hairline to Chin) / (Eye Width (outer))'] = round(
            measurements['Hairline to Chin'] / measurements['Eye Width (outer)'], 4)
        ratios['(Nose to Mouth) / (Nose to Chin)'] = round(
            measurements['Nose to Mouth'] / measurements['Nose to Chin'], 4)
    except ZeroDivisionError:
        pass
    return ratios

# === Normalization ===
def normalize_by_eye_distance(measurements):
    base = measurements.get("Eye Width (outer)", 1.0)
    return {k: round(v / base, 4) for k, v in measurements.items()}

def normalize_by_image_diag(measurements, image_shape):
    h, w = image_shape[:2]
    diag = math.hypot(w, h)
    return {k: round(v / diag, 4) for k, v in measurements.items()}

# === Comparison Table ===
def compare_ratios(r1, r2):
    comparison = []
    for key in MEASUREMENTS.keys():
        val1 = r1.get(key, 0)
        val2 = r2.get(key, 0)
        abs_diff = abs(val1 - val2)
        pct_diff = (abs_diff / val1 * 100) if val1 else 0
        comparison.append({
            "Feature": key,
            "Quotient Ratio": round(val1, 4),
            "Sample Ratio": round(val2, 4),
            "Abs Diff": round(abs_diff, 4),
            "% Diff": round(pct_diff, 2)
        })
    return {"rows": comparison}

# === Bar Chart Comparison ===
def plot_measurements_comparison(meas1, meas2, title="Measurement Comparison"):
    labels = list(meas1.keys())
    values1 = [meas1[k] for k in labels]
    values2 = [meas2[k] for k in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, values1, width, label='Quotient', color='steelblue')
    ax.bar([p + width for p in x], values2, width, label='Sample', color='darkorange')

    ax.set_xlabel('Facial Feature')
    ax.set_ylabel('Distance (pixels)')
    ax.set_title(title)
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

# === Visual Overlay Drawing ===
def draw_landmarks_with_letter_labels(image, landmarks, normalized=False, reference_value=None):
    # Convert to grayscale and back to BGR for visual consistency
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = annotated.copy()
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(1.2, h / 450))
    font_scale = scale * 1.2
    font_thickness = max(1, int(2 * scale))
    point_radius = max(3, int(5 * scale))
    line_thickness = max(2, int(2.5 * scale))
    label_offset = (8, -8)

    label_map = {}
    sorted_keys = sorted(landmarks.keys())
    for idx, key in enumerate(sorted_keys):
        label_map[chr(65 + idx)] = key
    reverse_map = {v: k for k, v in label_map.items()}

    for key, (x, y) in landmarks.items():
        letter = reverse_map[key]
        cv2.circle(overlay, (x, y), point_radius, (0, 255, 0), -1)
        cv2.putText(overlay, letter, (x + label_offset[0], y + label_offset[1]),
                    font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)

    mesh_lines = [
        ("left_eye", "right_eye"),
        ("left_eye", "nose_tip"), ("right_eye", "nose_tip"),
        ("nose_tip", "mouth_center"), ("mouth_center", "chin"),
        ("chin", "jaw_left"), ("chin", "jaw_right"),
        ("jaw_left", "left_ear"), ("jaw_right", "right_ear"),
        ("forehead_glabella", "left_eye"), ("forehead_glabella", "right_eye")
    ]
    for p1, p2 in mesh_lines:
        if p1 in landmarks and p2 in landmarks:
            cv2.line(overlay, landmarks[p1], landmarks[p2], (255, 0, 255), line_thickness)

    measurements_legend = {}
    for _, (p1, p2) in MEASUREMENTS.items():
        if p1 in landmarks and p2 in landmarks:
            pt1, pt2 = landmarks[p1], landmarks[p2]
            dist = euclidean_distance(pt1, pt2)
            label = f"{dist / reference_value:.2f}x" if (normalized and reference_value) else f"{int(dist)}px"
            a, b = reverse_map[p1], reverse_map[p2]
            measurements_legend[f"{a} → {b}"] = label
            cv2.line(overlay, pt1, pt2, (0, 255, 255), line_thickness)

    golden_lines = [
        ("forehead_glabella", 1.0, (0, 255, 0)),       # Green
        ("nose_tip", 1.618, (255, 165, 0)),            # Orange
        ("mouth_center", 0.6, (0, 255, 255)),          # Yellow
        ("chin", 2.0, (255, 0, 0)),                    # Blue
    ]
    for base_key, label_value, color in golden_lines:
        if base_key in landmarks:
            y = landmarks[base_key][1]
            cv2.line(overlay, (0, y), (w, y), color, 1)
            cv2.putText(overlay, f"{label_value}", (10, y - 4), font,
                        font_scale * 0.9, color, 1, cv2.LINE_AA)

    if "left_eye" in landmarks and "right_eye" in landmarks and "chin" in landmarks:
        cv2.line(overlay, landmarks["left_eye"], landmarks["chin"], (0, 100, 255), line_thickness)
        cv2.line(overlay, landmarks["right_eye"], landmarks["chin"], (0, 100, 255), line_thickness)
        cv2.line(overlay, landmarks["left_eye"], landmarks["right_eye"], (0, 100, 255), line_thickness)

    annotated = cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), label_map, measurements_legend