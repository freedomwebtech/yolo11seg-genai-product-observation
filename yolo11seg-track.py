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

# ✅ Set up Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCkpiuszi32b0lu6cRNl4os7hVefFvqH8g"  # Enter your API Key here

# ✅ Initialize the Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Load the YOLOv8 model
yolo_model = YOLO("best.pt")
names = yolo_model.names

# Open the video file (use video file or webcam, here using video file)
cap = cv2.VideoCapture('vid.mp4')
if not cap.isOpened():
    raise Exception("Error: Could not open video file.")

# Constants for ROI detection and tracking
cx1 = 257
offset = 6

# Create a folder for cropped images if it doesn't exist
if not os.path.exists("crop"):
    os.makedirs("crop")

# Set to track processed track_ids to avoid reprocessing
processed_track_ids = set()

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

def analyze_image_with_gemini(image):
    """Send image to Gemini for analysis."""
    if image is None:
        return "No image to analyze."

    # Convert the captured image to base64
    _, img_buffer = cv2.imencode('.jpg', image)
    image_data = base64.b64encode(img_buffer).decode('utf-8')

    # Create the message with the image
    message = HumanMessage(
        content=[ 
            {"type": "text", "text": "The agent's task is to detect packages which moving on conveyor belt. Mention package type or box type like color what kind of package it is. Check for each package individually and report if they are broken or good. Provide only that information."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}} 
        ]
    )
    
    # Send the message to Gemini and get the response
    try:
        response = gemini_model.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error invoking Gemini model: {e}")
        return "Error processing image."

def save_response_to_file(track_id, response):
    """Save the analysis response to a text file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    filename = "gemini_responses.txt"
    
    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"Track ID: {track_id} | Condition: {response} | Date: {timestamp}\n\n")
    
    print(f"Response saved to {filename}")

def save_crop_image(crop, track_id):
    """Save cropped image with track ID and timestamp."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"crop/{track_id}_{timestamp}.jpg"
    cv2.imwrite(filename, crop)
    print(f"Cropped image saved as {filename}")
    return filename

def crop_and_process(frame, box, track_id):
    """Crop detected objects and send for analysis, only if track_id hasn't been processed."""
    if track_id in processed_track_ids:
        print(f"Track ID {track_id} has already been processed. Skipping.")
        return
    
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]

    # Save the cropped image to the 'crop' folder
    crop_filename = save_crop_image(crop, track_id)

    # Start a thread to send the cropped image to Gemini for text extraction
    threading.Thread(target=process_crop_image, args=(crop, track_id, crop_filename)).start()

    # Mark track_id as processed
    processed_track_ids.add(track_id)

def process_crop_image(crop, track_id, crop_filename):
    """Process the cropped image in a separate thread."""
    response_content = analyze_image_with_gemini(crop)
    print("Gemini Response:", response_content)

    # Save Gemini's response to a file with track ID and timestamp
    save_response_to_file(track_id, response_content)

    # Save Gemini response along with the cropped image's timestamp in a separate file
    response_filename = crop_filename.replace(".jpg", "_response.txt")
    with open(response_filename, "w", encoding="utf-8") as f:
        f.write(f"Track ID: {track_id}\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}\nResponse: {response_content}\n")

def process_video_frame(frame):
    """Process video frame for object detection and analysis."""
    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLOv8 tracking on the frame
    results = yolo_model.track(frame, persist=True)
    
    # Check if boxes exist in the results
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        # Get tracking IDs if available
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)
        
        masks = results[0].masks
        if masks is not None:
            for box, track_id, class_id, mask in zip(boxes, track_ids, class_ids, masks):
                c = names[class_id]
                x1, y1, x2, y2 = box
                cx = int(x1 + x2) // 2
                cy = int(y1 + y2) // 2
                if cx1 < (cx + offset) and cx1 > (cx - offset):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x2, y2), 1, 1)
                    cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                    crop_and_process(frame, box, track_id)

    return frame

def main():
    """Main function to run video processing and capture."""
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for object detection and analysis
        frame = process_video_frame(frame)

        # Draw a reference line
        cv2.line(frame, (257, 1), (257, 499), (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
