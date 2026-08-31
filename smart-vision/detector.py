from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """
        Initializes the YOLO object detector. 
        """
        self.model_name = model_name
        self.model = YOLO(model_name)
        # Expose class names for the UI to populate the class filter
        self.class_names = self.model.names

    def detect_objects(self, image_array, conf_threshold=0.5, iou_threshold=0.45, classes=None, blur=False):
        """
        Runs object detection on the provided image array.
        """
        # Run inference using the YOLO model
        results = self.model(image_array, conf=conf_threshold, iou=iou_threshold, classes=classes)
        
        result = results[0]
        
        if blur:
            # image_array is RGB from Streamlit
            annotated_img_rgb = image_array.copy()
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Ensure coordinates are within image bounds
                y1, y2 = max(0, y1), min(annotated_img_rgb.shape[0], y2)
                x1, x2 = max(0, x1), min(annotated_img_rgb.shape[1], x2)
                
                # Extract ROI
                roi = annotated_img_rgb[y1:y2, x1:x2]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    h, w = roi.shape[0], roi.shape[1]
                    # Dynamic odd kernel size bounded by ROI dimensions (max 99)
                    kw = min(99, w if w % 2 == 1 else w - 1)
                    kh = min(99, h if h % 2 == 1 else h - 1)
                    if kw >= 3 and kh >= 3:
                        blurred_roi = cv2.GaussianBlur(roi, (kw, kh), 30)
                    else:
                        blurred_roi = cv2.blur(roi, (max(1, w), max(1, h)))
                    annotated_img_rgb[y1:y2, x1:x2] = blurred_roi
        else:
            # result.plot() returns BGR
            annotated_img_bgr = result.plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
            
        detections = []
        class_counts = {}
        
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            detections.append({
                "class": class_name,
                "confidence": round(conf, 3),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        total_objects = len(detections)
        
        return annotated_img_rgb, detections, total_objects, class_counts

    def process_video_frame(self, frame_bgr, conf_threshold=0.5, iou_threshold=0.45, classes=None, blur=False):
        """
        Processes a single video frame (OpenCV BGR format).
        Returns annotated_frame_bgr, detections, total_objects, class_counts.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated_rgb, detections, total_objects, class_counts = self.detect_objects(
            frame_rgb, conf_threshold=conf_threshold, iou_threshold=iou_threshold, classes=classes, blur=blur
        )
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        return annotated_bgr, detections, total_objects, class_counts

