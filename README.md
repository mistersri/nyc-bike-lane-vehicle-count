# Bike Lane Vehicle Detection

This project uses computer vision to detect and count vehicles in a designated bike lane from public video footage provided by NYC DOT. The system is designed to identify vehicles encroaching into the bike lane and log this information to a Google Sheet for analysis.

**_Warning: Test purposes only. The confidence interval had to be set very low (0.01) to detect any vehicles in the bike lane, but often reports false postives in testing. For easy confidence threshold testing, use Roboflow's [web interface for vehicle detection](https://universe.roboflow.com/detection-vehicle-g6pdp/vehicle-detection-vokgr/model/)._**

## Features

- **Real-time Detection**: Utilizes a machine learning model [(vehicle detection)](https://universe.roboflow.com/leo-ueno/vehicle-detection-3mmwj) to detect vehicles in video frames.
- **Zone-Based Counting**: Counts vehicles within a defined polygonal zone representing the bike lane.
- **Data Logging**: Records the count of vehicles in the bike lane along with timestamps to a Google Sheet.
- **Video Annotation**: Annotates video frames with detection boxes and zone overlays.
- **Fickleness**: Sometimes it works, sometimes it doesn't (see note above).

## Requirements

- Python >=3.8, <=3.11 (Inference does not support Python 3.12+ at time of publishing)
- OpenCV
- NumPy
- gspread
- Google OAuth 2.0 Credentials
- Supervision and Inference Libraries

## Usage
- Run the script to start detecting and counting vehicles.
- The annotated video will be saved as `bike_lane_output.mp4`, and vehicle counts will be logged to the specified Google Sheet.
- Ride safely.

## Troubleshooting
- Ensure that your polygon zone accurately covers the bike lane.
- Adjust confidence levels if detections are inconsistent.
- Consider using a GPU for improved performance.

## Special Thanks
- [James Gallagher](https://blog.roboflow.com/author/james/). (May 30, 2023). [How to Count Objects in a Zone](https://blog.roboflow.com/how-to-count-objects-in-a-zone/). Roboflow Blog.
- [Leo Ueno](https://blog.roboflow.com/author/leo/). (May 3, 2024). [Realtime Video Stream Analysis with Computer Vision](https://blog.roboflow.com/video-stream-analysis/). Roboflow Blog.
