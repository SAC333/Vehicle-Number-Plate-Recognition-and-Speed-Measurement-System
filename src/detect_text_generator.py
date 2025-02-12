
import cv2
import pytesseract
import torch
from ultralytics import YOLO
import os
import re

# Load the YOLO model (Number plate detection model)
model = YOLO("D:/IT/IASCD/vnprsm/models/vehicle_and _number_plate/best.pt")  # Replace with your trained model path

# Load Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Update the path if needed

# Input video path
video_path = "img_and_videos/carLicence4.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Output video writer
output_video_path = os.path.join(output_dir, "output_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using YOLO
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, class_id = map(int, det[:6])

        # Process only number plates (class_id == 0)
        if class_id == 0:
            # Crop detected number plate region
            plate_img = frame[y1:y2, x1:x2]

            # Preprocess the image for better OCR accuracy
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply OCR
            text = pytesseract.image_to_string(thresh, config='--psm 7')  # Use mode 7 for single-line text

            # Clean extracted text (remove unwanted characters)
            text = re.sub(r'[^A-Za-z0-9]', '', text).upper()

            # Only save if text is detected
            if text:
                # Save image with number plate text as filename
                plate_img_path = os.path.join(output_dir, f"{text}.jpg")
                cv2.imwrite(plate_img_path, plate_img)

                # Display the detected text
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write processed frame to output video
    out.write(frame)

    # Show frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at: {output_video_path}")
print(f"Detected number plate images saved in: {output_dir}")
