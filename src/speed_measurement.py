import cv2
import os
import yaml
import torch
import pytesseract
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from tqdm import tqdm
import supervision as sv

# Load paths from YAML
with open("Documents\config.yaml", "r") as file:
    paths = yaml.safe_load(file)

# Assign paths
VEHICLE_MODEL = paths["vehicle_model"]
SPEED_MODEL = paths["speed_model"]
TESSERACT_PATH = paths["tesseract_path"]
INPUT_VIDEO = paths["input_video"]
OUTPUT_VIDEO = paths["output_video"]

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

# Load models
vehicle_model = YOLO(VEHICLE_MODEL)
speed_model = YOLO(SPEED_MODEL)

# Set up Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Load video
cap = cv2.VideoCapture(INPUT_VIDEO)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video Writer Setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

# Tracking & Speed Calculation Setup
byte_track = sv.ByteTrack()
coordinates = defaultdict(lambda: deque(maxlen=fps))
thickness = max(1, int(min(frame_width, frame_height) / 400))
text_scale = max(0.5, min(frame_width, frame_height) / 1000)

# Annotators
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=fps * 2)

# Process video
print("Processing Video...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Vehicle & Number Plate Detection
    results = vehicle_model(frame)
    detections = results[0].boxes.data.cpu().numpy() if results else []
    labels = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{vehicle_model.names[cls]} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Process Number Plate with OCR
        if cls == 0:  # Assuming 0 is the class ID for number plates
            plate_img = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, config='--psm 7')
            text = ''.join(filter(str.isalnum, text)).upper()
            if text:
                cv2.putText(frame, text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print("Detected Plate:", text)

    # Speed Detection
    speed_results = speed_model(frame, imgsz=640, verbose=False)[0]
    speed_detections = sv.Detections.from_ultralytics(speed_results)
    speed_detections = speed_detections[speed_detections.confidence > 0.3]
    speed_detections = byte_track.update_with_detections(speed_detections)

    for tracker_id in speed_detections.tracker_id:
        coordinates[tracker_id].append(tracker_id)
        if len(coordinates[tracker_id]) > fps / 2:
            distance = abs(coordinates[tracker_id][-1] - coordinates[tracker_id][0])
            speed = (distance / (len(coordinates[tracker_id]) / fps)) * 3.6  # Convert to km/h
            labels.append(f"#{tracker_id} {int(speed)} km/h")

    # Annotate & Write Frame
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=speed_detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=speed_detections, labels=labels)
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=speed_detections)
    out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processing complete! Output saved at {OUTPUT_VIDEO}")
