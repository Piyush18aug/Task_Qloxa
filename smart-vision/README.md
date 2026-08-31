# Smart Vision 👁️

## Overview
Smart Vision is a web-based artificial intelligence application that performs real-time object detection on user-uploaded images. It identifies objects, draws bounding boxes around them, and provides confidence scores and aggregated counts.

## Features
- **Image Upload:** Supports standard image formats (JPG, PNG).
- **Object Detection:** Identifies 80 different classes of objects (people, cars, animals, etc.).
- **Confidence Filtering:** Interactive slider to filter out uncertain AI predictions.
- **Visual Feedback:** Side-by-side comparison of the original and AI-annotated image.
- **Analytics Table:** Detailed breakdown of every detected object's location and confidence.
- **CSV Export:** Download detection data for further analysis.

## Technologies Used
- **Python 3.10+**: Core programming language.
- **Ultralytics YOLO (yolov8n)**: State-of-the-art, lightweight Deep Learning model for object detection.
- **OpenCV**: Computer vision library used for image matrix and color format manipulation.
- **Streamlit**: Python framework used to build the interactive web UI seamlessly.
- **NumPy & Pillow**: For image reading and array data manipulation.

## How It Works
1. The user uploads an image via the web interface.
2. The application converts the image into a numerical matrix (NumPy array).
3. The YOLO neural network processes the image and predicts object locations.
4. The system filters out predictions that fall below the user-defined confidence threshold.
5. Bounding boxes and labels are drawn onto the image.
6. The annotated image, summary counts, and tabular data are rendered on the screen.

## Installation

Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

Execute the following command to start the web app:
```bash
streamlit run app.py
```
