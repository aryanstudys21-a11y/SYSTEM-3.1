import os
import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="MORA Vu: Clinical Posture AI", layout="wide")

@st.cache_resource
def load_onnx_model():
    return ort.InferenceSession("posture_cnn_griffin.onnx", providers=["CPUExecutionProvider"])

session = load_onnx_model()

# Built-in OpenCV Haar Cascade for automatic head/ear region localization (Zero external dependencies)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_cervical_roi(img_bgr):
    """Automatically locates the cervical-thoracic junction ROI or falls back to a centered clinical framing box."""
    h, w, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        faces = frontal_cascade.detectMultiScale(gray, 1.1, 4)
        
    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        # Expand box from ear/head level down to upper thoracic (C7) region
        x_min = max(0, int(fx - fw * 0.6))
        x_max = min(w, int(fx + fw * 1.6))
        y_min = max(0, int(fy - fh * 0.2))
        y_max = min(h, int(fy + fh * 2.2))
    else:
        # Clinical sagittal framing box fallback
        x_min, x_max = int(w * 0.2), int(w * 0.8)
        y_min, y_max = int(h * 0.1), int(h * 0.85)
        
    return x_min, y_min, x_max, y_max

def calculate_cva(c7_pt, tragus_pt):
    dx = max(abs(tragus_pt[0] - c7_pt[0]), 1e-5)
    dy = c7_pt[1] - tragus_pt[1]
    return round(float(np.degrees(np.arctan2(dy, dx))), 2)

def analyze_posture_frame(img_bgr):
    x_min, y_min, x_max, y_max = get_cervical_roi(img_bgr)
    roi = img_bgr[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return img_bgr, 0.0, "No ROI"

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (224, 224))
    
    # Normalize for MobileNetV3 ONNX input
    img_array = roi_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_array - mean) / std
    input_tensor = np.expand_dims(np.transpose(img_norm, (2, 0, 1)), axis=0).astype(np.float32)
    
    pred_coords, _ = session.run(None, {"input_frame": input_tensor})
    pts = (pred_coords[0] * 224).astype(int)
    c7_roi, tragus_roi = (pts[0], pts[1]), (pts[2], pts[3])
    cva = calculate_cva(c7_roi, tragus_roi)
    
    # Map ROI coordinates back to original frame resolution
    scale_x = (x_max - x_min) / 224.0
    scale_y = (y_max - y_min) / 224.0
    c7_full = (int(c7_roi[0] * scale_x) + x_min, int(c7_roi[1] * scale_y) + y_min)
    tragus_full = (int(tragus_roi[0] * scale_x) + x_min, int(tragus_roi[1] * scale_y) + y_min)
    
    status = "Normal" if cva >= 50 else "Mild FHP" if cva >= 40 else "Severe FHP"
    color = (0, 220, 0) if status == "Normal" else (0, 165, 255) if status == "Mild FHP" else (0, 0, 255)
    
    # Draw ROI bounding box, horizontal C7 reference line, and CVA vector
    cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (200, 200, 200), 2)
    cv2.line(img_bgr, c7_full, (c7_full[0] + 100, c7_full[1]), (255, 255, 255), 2)
    cv2.line(img_bgr, c7_full, tragus_full, color, 3)
    cv2.circle(img_bgr, c7_full, 7, (255, 50, 50), -1)
    cv2.circle(img_bgr, tragus_full, 7, (0, 200, 255), -1)
    
    cv2.putText(img_bgr, f"CVA: {cva} deg ({status})", (x_min, max(30, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
    return img_bgr, cva, status

class PostureVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        import av
        img = frame.to_ndarray(format="bgr24")
        annotated_img, _, _ = analyze_posture_frame(img)
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

st.title("MORA Vu: Clinical Posture & CVA Analyzer")
st.markdown("Markerless Craniovertebral Angle (CVA) & Forward Head Posture tracking powered by ONNX[cite: 3].")

tab1, tab2 = st.tabs(["Snapshot / Image Upload (Recommended for Cloud)", "Live WebRTC Stream"])

with tab1:
    col1, col2 = st.columns([1.5, 1])
    with col1:
        mode = st.radio("Input Source:", ["Camera Snapshot", "Upload Clinical Photo"], horizontal=True)
        img_input = st.camera_input("Capture Sagittal Posture") if mode == "Camera Snapshot" else st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        
    with col2:
        st.subheader("Biomechanical Assessment")
        if img_input:
            pil_img = Image.open(img_input).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            annotated_bgr, cva, status = analyze_posture_frame(img_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            
            st.image(annotated_rgb, caption="Cervical-Thoracic ROI & Landmark Vector", use_container_width=True)
            st.metric("Craniovertebral Angle (CVA)", f"{cva}°")
            if status == "Normal":
                st.success(f"Posture Classification: {status} (≥ 50°)")
            elif status == "Mild FHP":
                st.warning(f"Posture Classification: {status} (40° - 49.9°)")
            else:
                st.error(f"Posture Classification: {status} (< 40°)")
        else:
            st.info("Capture or upload a side-profile photo to compute CVA.")

with tab2:
    st.markdown("Real-time webcam stream with automated ROI cropping[cite: 3].")
    webrtc_streamer(key="posture-live", video_processor_factory=PostureVideoProcessor)
