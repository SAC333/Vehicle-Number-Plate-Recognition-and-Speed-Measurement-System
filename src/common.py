import cv2
import os
import torch
import pytesseract
import yaml
import numpy as np
import supervision as sv
from collections import defaultdict, deque
from ultralytics import YOLO
from tqdm import tqdm

# Load paths from path.yaml
with open("Documents\config.yaml", "r") as file:
    paths = yaml.safe_load(file)

# Load YOLO models
vehicle_model = YOLO(paths["vehicle_model"])
speed_model = YOLO(paths["speed_model"])

# Load Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = paths["tesseract_path"]

# Define input/output video paths
input_video_path = paths["input_video"]
output_video_path = paths["output_video"]

# Video properties
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize ByteTrack tracker
byte_track = sv.ByteTrack()
coordinates = defaultdict(lambda: deque(maxlen=fps))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    results = vehicle_model(frame)
    detections = results[0].boxes.data.cpu().numpy()
    
    for det in detections:
        x1, y1, x2, y2, conf, class_id = map(int, det[:6])
        label = f"{vehicle_model.names[class_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if class_id == 0:  # Assuming 0 is the class for number plates
            plate_img = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, config='--psm 7')
            text = ''.join(filter(str.isalnum, text)).upper()
            if text:
                cv2.putText(frame, text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Speed detection
    speed_results = speed_model(frame)
    detections = sv.Detections.from_ultralytics(speed_results[0])
    detections = detections[detections.confidence > 0.3]
    detections = byte_track.update_with_detections(detections)

    for tracker_id in detections.tracker_id:
        coordinates[tracker_id].append(detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0, 1])
        if len(coordinates[tracker_id]) > 10:
            dist = abs(coordinates[tracker_id][-1] - coordinates[tracker_id][0])
            speed = (dist / (len(coordinates[tracker_id]) / fps)) * 3.6  # Convert to km/h
            cv2.putText(frame, f"ID {tracker_id}: {int(speed)} km/h", (50, 50 + tracker_id * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved at: {output_video_path}")
