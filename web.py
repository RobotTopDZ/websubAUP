
# RUN IT USING THE COMMAND "streamlit run (the path)"____________________________________________________

import streamlit as st
import torch
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load trained YOLOv8 model
model = YOLO("modelultime.pt")  # Ensure best.pt is in "model/" directory

# Streamlit UI
st.title("📊 Ramy Shelf Share Detection")
st.write("Upload an image of a store shelf, or use the camera to take a picture, and the model will detect products and calculate Ramy's shelf share.")

# Option to upload an image
uploaded_file = st.file_uploader("📸 Upload a shelf image", type=["jpg", "png", "jpeg"])

# Option to capture image from webcam
camera_input = st.camera_input("📷 Capture image from webcam")

if uploaded_file:
    # If user uploads an image, use that
    image = Image.open(uploaded_file)
    image = np.array(image)
elif camera_input:
    # If user uses the webcam, use the captured image
    image = Image.open(camera_input)
    image = np.array(image)

# Run YOLOv8 inference only if an image is available
if uploaded_file or camera_input:
    # Run YOLOv8 inference
    results = model.predict(image, conf=0.4)  # Adjust confidence threshold if needed

    # Display detected image
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs

        # Load class names from dataset
        class_names = model.names

        # Count Ramy vs. Competitor products
        total_products = len(class_ids)

        # Count Ramy products
        ramy_count = sum(1 for cls in class_ids if int(cls) in class_names and "ramy" in class_names[int(cls)].lower())


        # ramy_count = sum(1 for cls in class_ids if "Ramy" in class_names[int(cls)].lower())
        competitor_count = total_products - ramy_count

        # Calculate percentage
        ramy_percentage = (ramy_count / total_products) * 100 if total_products > 0 else 0

        # Draw bounding boxes
        for box, cls in zip(boxes, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = class_names[int(cls)]
            color = (0, 255, 0) if "Ramy" in label.lower() else (255, 0, 0)  # Green for Ramy, Red for Competitors
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert back to PIL for display
        st.image(image, caption="Detected Products", use_container_width=True)

        # Display results
        st.subheader("📊 Results")
        st.write(f"🧃 **Total Products Detected:** {total_products}")
        st.write(f"✅ **Ramy Products:** {ramy_count}")
        st.write(f"🚨 **Competitor Products:** {competitor_count}")
        st.write(f"📊 **Ramy's Shelf Share:** {ramy_percentage:.2f}%")

        # Progress bar for visualization
        st.progress(int(ramy_percentage))
