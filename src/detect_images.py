import cv2
import os
import torch
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("models/vehicle_and _number_plate/best.pt")  # Update with your local path

# Define input and output folders
input_images_folder = "D:/IT/IASCD/vnprsm/img_and_videos"  # Folder containing images
input_video_path = "img_and_videos\\vehicle.mp4"  # Input video
output_folder = "D:\\IT\\IASCD\\vnprsm\\output"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

### **Process Images**
def process_images():
    print("\n Processing Images...")
    for img_name in os.listdir(input_images_folder):
        img_path = os.path.join(input_images_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            print(f" Could not read image: {img_path}. Skipping...")
            continue
        
        # Perform object detection
        results = model(image)

        # Draw bounding boxes and labels
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = map(int, box[:6])
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw rectangle and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Save the processed image
        cv2.imwrite(output_path, image)
        print(f" Processed: {img_name} -> Saved to {output_path}")

### 🎥 **Process Video**
def process_video():
    print("\n🎥 Processing Video...")

    # Open video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Output video path
    output_video_path = os.path.join(output_folder, "vehicle_output.mp4")

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Break if video ends

        # Perform object detection
        results = model(frame)

        # Draw bounding boxes and labels
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = map(int, box[:6])
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Write frame to output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Processed Video Saved to: {output_video_path}")

### 🔄 **Run Both Image & Video Processing**
process_images()
process_video()
