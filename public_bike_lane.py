# Imports and Setup
import numpy as np
import cv2
import os
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import supervision as sv

# Google Sheets Setup (OAuth 2.0)
import gspread
from google.oauth2.service_account import Credentials

creds = Credentials.from_service_account_file('credentials.json', scopes=[
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
])

googlesheets = gspread.authorize(creds)
document = googlesheets.open_by_key('<your sheets key>')  # Change this to your Google Sheet key
worksheet = document.worksheet('BikeLaneCounts')

# VideoWriter Setup (OpenCV)
video_info = (352, 240, 60)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter("bike_lane_output.mp4", fourcc, video_info[2], video_info[:2])

# Signal Handling for Graceful Exit (Ctrl + C)
import signal
import sys

def signal_handler(sig, frame):
    writer.release()
    print("Keyboard Interrupt")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Roboflow API Key (Hardcoded for now)
api_key = "< your api key>" # Change this to your Roboflow API Key

# Inference Pipeline Callback Function for Predictions and Video Frames
from datetime import datetime
import pytz

# Define Bike Lane Polygon
polygon = np.array([
    [177, 237], [192, 20], [162, 21], [0, 213]
])

# Define Polygon Zone
zone = sv.PolygonZone(
    polygon=polygon,
    frame_resolution_wh=video_info[:2]
)

zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.WHITE,
    thickness=3
)

# Handle prediction processing for each frame
def on_prediction(predictions, video_frame, writer):
     # Convert predictions into Supervision detections
    detections = sv.Detections.from_inference(predictions)
    
     # Trigger the zone to count objects inside the polygon
    zone.trigger(detections=detections)

    # Access current count of objects in the bike lane
    bike_lane_count = zone.current_count
    
    # Debugging: Print count of cars in the bike lane
    print(f"Cars in bike lane: {bike_lane_count}")
    
    # Annotate the frame with bounding boxes and zone details
    annotated_frame = sv.BoxAnnotator(thickness=1).annotate(video_frame.image, detections)
    annotated_frame = zone_annotator.annotate(scene=annotated_frame)
    
    writer.write(annotated_frame)  # Write frame to timelapse video
    
    ET = pytz.timezone('America/New_York')
    time = datetime.now(ET).strftime("%H:%M")
    
    fields = [time, bike_lane_count]
    
    print(fields)  # Print timestamp and number of detections in the bike lane
    
    worksheet.append_rows([fields], "USER_ENTERED")  # Append data to Google Sheet

# Initialize and Start Inference Pipeline (Roboflow Model)
pipeline = InferencePipeline.init(
    model_id="vehicle-detection-3mmwj/1",
    max_fps=0.5,
    confidence=0.01,
    video_reference="https://webcams.nyctmc.org/api/cameras/557af346-2f9f-4306-8388-3974b7a49e4d/image/",
    on_prediction=lambda predictions, video_frame: on_prediction(predictions, video_frame, writer),
    api_key=api_key
)
print("Pipeline initialized... waiting for frames.")

pipeline.start()
pipeline.join()

# Release VideoWriter resources when done
writer.release()
print("Video writer released.")