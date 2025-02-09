import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import glob

# ✅ Set up Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCkpiuszi32b0lu6cRNl4os7hVefFvqH8g"  # Replace with your API Key

# ✅ Initialize the Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Load the YOLOv8 model
yolo_model = YOLO("best.pt")
names = yolo_model.names

# Open the video file
cap = cv2.VideoCapture('vid4.mp4')
if not cap.isOpened():
    raise Exception("Error: Could not open video file.")

# Constants for ROI detection and tracking
cx1 = 491
offset = 8

# Get current date for folder and file naming
current_date = time.strftime("%Y-%m-%d")

# Create a folder for cropped images based on current date
crop_folder = f"crop_{current_date}"
if not os.path.exists(crop_folder):
    os.makedirs(crop_folder)

# Track last saved image
last_saved_image = None

def get_last_saved_image():
    """Retrieve the last saved image from the 'crop' folder."""
    global last_saved_image
    # Use glob to list all jpg files and sort them by modification time
    images = glob.glob(os.path.join(crop_folder, "*.jpg"))
    images.sort(key=os.path.getmtime, reverse=True)

    if images:
        last_saved_image = images[0]
        return cv2.imread(last_saved_image)
    
    return None

def encode_image_to_base64(image):
    """Convert an image to a base64 string."""
    _, img_buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(img_buffer).decode('utf-8')

def analyze_images_with_gemini(last_image, current_image):
    """Compare two images using Gemini AI."""
    if last_image is None or current_image is None:
        return "No images available for comparison."

    # Convert images to base64
    last_image_data = encode_image_to_base64(last_image)
    current_image_data = encode_image_to_base64(current_image)

    # Create the Gemini request
    message = HumanMessage(
    content=[{
            "type": "text",
            "text": """
            Compare these two images and determine if they are the same product or different. Ignore mirror image differences. Report differences in a structured format. Specifically, check:
            1. **Product Type** (Class-wise categorization)
            2. **Is it the same product?** (Yes/No)
            3. **Is the label present?** (Yes/No)
            4. **Is there any damage?** (Yes/No)

            Return the result strictly in a structured table format like below:

            | Product Type | Same Product | Label Present | Damage |
            |-------------|-------------|--------------|--------|
            | ExampleType | Yes/No      | Yes/No       | Yes/No |
            """
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{last_image_data}"},
            "description": "Last detected product"
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{current_image_data}"},
            "description": "Current detected product"
        }
    ])

    # Send the request to Gemini AI
    try:
        response = gemini_model.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error invoking Gemini model: {e}")
        return "Error processing images."

def save_response_to_file(track_id, response):
    """Save the analysis response to a text file with current date."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    response_filename = f"gemini_response_{current_date}_report.txt"
    
    try:
        with open(response_filename, "a", encoding="utf-8") as file:
            file.write(f"Track ID: {track_id} | Condition: {response} | Date: {timestamp}\n\n")
        print(f"Response saved to {response_filename}")
    except Exception as e:
        print(f"Error saving response to file: {e}")

def save_crop_image(crop, track_id):
    """Save cropped image with track ID and timestamp in current date folder."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{crop_folder}/{track_id}_{timestamp}.jpg"
    try:
        cv2.imwrite(filename, crop)
        print(f"Cropped image saved as {filename}")
    except Exception as e:
        print(f"Error saving cropped image: {e}")
    return filename

def crop_and_process(frame, box, track_id):
    """Crop detected objects and send for analysis."""
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]

    # Save the cropped image
    crop_filename = save_crop_image(crop, track_id)

    # Retrieve the last saved image
    last_image = get_last_saved_image()

    # Start thread to compare images using Gemini
    threading.Thread(target=process_crop_image, args=(last_image, crop, track_id, crop_filename)).start()

def process_crop_image(last_image, current_image, track_id, crop_filename):
    """Process the cropped image and compare it to the last one using Gemini AI."""
    response_content = analyze_images_with_gemini(last_image, current_image)
    print("Gemini Response:", response_content)

    # Save response
    save_response_to_file(track_id, response_content)

    # Save response in a file
    response_filename = crop_filename.replace(".jpg", "_response.txt")
    try:
        with open(response_filename, "w", encoding="utf-8") as f:
            f.write(f"Track ID: {track_id}\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}\nResponse: {response_content}\n")
    except Exception as e:
        print(f"Error saving response file: {e}")

def process_video_frame(frame):
    """Process video frame for object detection and analysis."""
    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLOv8 tracking
    results = yolo_model.track(frame, persist=True)
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            c = names[class_id]
            x1, y1, x2, y2 = box
            cx = int(x1 + x2) // 2

            if cx1 < (cx + offset) and cx1 > (cx - offset):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{track_id}', (x2, y2), 1, 1)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                crop_and_process(frame, box, track_id)

    return frame

def main():
    """Main function to run video processing."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_video_frame(frame)

        cv2.line(frame, (491, 1), (491, 499), (0, 0, 255), 2)

        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
