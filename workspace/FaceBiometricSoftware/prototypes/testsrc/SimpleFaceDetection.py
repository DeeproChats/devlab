import streamlit as st # type: ignore
import dlib # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore

# --- Load dlib face detector and landmark predictor ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/dlib/shape_predictor_68_face_landmarks.dat")

st.title("Minimal Dlib + Streamlit Test")

# --- Upload image ---
img_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if img_file:
    img = Image.open(img_file)
    st.image(img, caption="Uploaded Image", width=300)
    
    # Convert to numpy array (OpenCV format)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Detect faces
    faces = detector(img_cv)
    st.write(f"Detected {len(faces)} face(s)")

    for face in faces:
        shape = predictor(img_cv, face)
        # Draw landmarks
        for i in range(68):
            cv2.circle(img_cv, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1)

    # Show annotated image
    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Landmarks Detected", width=300)
