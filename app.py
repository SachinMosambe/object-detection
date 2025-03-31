import streamlit as st

#  THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="AI Object Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import asyncio
import sys

#  Fix "no running event loop" error for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#  Load YOLO Model (Only Once for Performance)
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")  # Load from local file
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_model()

#  Streamlit UI
st.title("AI Object Detector")
st.write("Upload an image to detect objects using YOLOv8.")

# Sidebar for Additional Information
with st.sidebar:
    st.header("About this App")
    st.markdown("""
    This AI-powered application uses the **YOLOv8 object detection model** to identify objects in images. 
    Simply upload an image, and the model will highlight detected objects with bounding boxes.
    """)
    
    st.header("How it Works")
    st.markdown("""
    - Uses the **YOLOv8n** model for fast and accurate detection.
    - Draws **dark, thin bounding boxes** around detected objects.
    - Labels each object with its name and confidence score.
    - Allows you to **download the processed image**.
    """)

#  File Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model:
    try:
        #  Read & Show Uploaded Image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert image to OpenCV format
        image_np = np.array(image.convert("RGB"))
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        #  Perform Object Detection with Progress Bar
        with st.spinner("Detecting Objects... Please wait."):
            results = model(image_cv)

        #  Convert to PIL Image for Fancy Drawing
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        #  Load a Fancy Font (Built-in Pillow Font)
        try:
            font = ImageFont.truetype("arial.ttf", 20)  # Windows/Mac/Linux Compatible
        except:
            font = ImageFont.load_default()  # Use default if Arial is unavailable

        #  Draw Bounding Boxes with Improved Styling
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                confidence = float(box.conf[0]) * 100

                # 🔹 Dark Thin Bounding Box
                draw.rectangle([(x1, y1), (x2, y2)], outline="black", width=2)

                # 🔹 Add Text with Semi-Transparent Background
                text = f"{label} {confidence:.1f}%"
                text_size = draw.textbbox((0, 0), text, font=font)
                text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]

                # Text Background
                draw.rectangle([(x1, y1 - text_height - 4), (x1 + text_width + 4, y1)], fill=(0, 0, 0, 128))

                # Add White Text on Top
                draw.text((x1 + 2, y1 - text_height - 2), text, font=font, fill="white")

        #  Show Processed Image
        st.image(image_pil, caption="Detected Objects", use_container_width=True)

        #  Convert Image to Bytes for Download
        img_bytes = io.BytesIO()
        image_pil.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        #  Provide Download Option
        st.download_button(
            label="Download Processed Image",
            data=img_bytes,
            file_name="detected_objects.jpg",
            mime="image/jpeg",
        )

    except Exception as e:
        st.error(f"Error processing image: {e}")
