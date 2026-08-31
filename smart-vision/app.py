import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from detector import ObjectDetector

st.set_page_config(page_title="Smart Vision Pro", layout="wide", page_icon="👁️")

# Cache the model loading, using model_name as part of the key so it reloads if changed
@st.cache_resource
def load_detector(model_name):
    return ObjectDetector(model_name=model_name)

st.title("Smart Vision Pro 👁️")
st.write("### Advanced Object Detection using YOLO")

# Sidebar Configuration
st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose YOLO Model",
    options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0,
    help="Nano (fastest), Small (balanced), Medium (most accurate)"
)

# Load the AI model dynamically
detector = load_detector(model_choice)

st.sidebar.header("Inference Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.50, 
    step=0.05
)

iou_threshold = st.sidebar.slider(
    "IoU (NMS) Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.45, 
    step=0.05,
    help="Intersection over Union threshold for overlapping boxes."
)

# Convert class dictionary to a list for the multiselect
all_classes = list(detector.class_names.values())
selected_class_names = st.sidebar.multiselect(
    "Filter Classes (Leave empty to detect all)",
    options=all_classes,
    default=[]
)

# Map selected class names back to class IDs for YOLO
selected_class_ids = [k for k, v in detector.class_names.items() if v in selected_class_names]
if len(selected_class_ids) == 0:
    selected_class_ids = None  # None means detect all classes

st.sidebar.header("Visualization")
blur_objects = st.sidebar.checkbox("Blur Detected Objects (Privacy Mode)", value=False)

# Main UI
tab1, tab2 = st.tabs(["🖼️ Image Detection", "ℹ️ About Advanced Mode"])

with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            st.write("---")
            if st.button("Detect Objects", type="primary"):
                with st.spinner(f"Analyzing image with {model_choice}..."):
                    
                    # Call our backend logic with new parameters
                    annotated_img, detections, total_objects, class_counts = detector.detect_objects(
                        image_array, 
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        classes=selected_class_ids,
                        blur=blur_objects
                    )
                    
                    st.subheader("Detection Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Image**")
                        st.image(image, use_container_width=True)
                        
                    with col2:
                        st.write("**Detected Image**")
                        st.image(annotated_img, use_container_width=True)
                    
                    st.write("---")
                    st.write(f"### Total Objects: {total_objects}")
                    if class_counts:
                        summary_str = " | ".join([f"{k.capitalize()}: {v}" for k, v in class_counts.items()])
                        st.write(f"**Summary:** {summary_str}")
                    
                    if total_objects > 0:
                        df = pd.DataFrame(detections)
                        df_ui = df.copy()
                        df_ui['confidence_pct'] = (df_ui['confidence'] * 100).round(1).astype(str) + "%"
                        df_ui['bounding_box'] = df_ui.apply(lambda row: f"({row['x1']}, {row['y1']}, {row['x2']}, {row['y2']})", axis=1)
                        display_df = df_ui[['class', 'confidence_pct', 'bounding_box']].rename(
                            columns={'class': 'Object', 'confidence_pct': 'Confidence', 'bounding_box': 'Bounding Box'}
                        )
                        st.table(display_df)
                        
                        csv_data = df[['class', 'confidence', 'x1', 'y1', 'x2', 'y2']].to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_data,
                            file_name="detections.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("No objects detected with the current settings.")
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

with tab2:
    st.write("### Advanced Features Added")
    st.markdown("""
    - **Model Selection**: Switch between Nano, Small, and Medium YOLOv8 models dynamically.
    - **IoU Threshold (NMS)**: Fine-tune how the AI handles overlapping bounding boxes.
    - **Class Filtering**: Restrict the AI to only detect specific objects (e.g., only 'person' and 'car').
    - **Privacy Mode (Blurring)**: Automatically blur detected objects instead of drawing bounding boxes.
    """)
