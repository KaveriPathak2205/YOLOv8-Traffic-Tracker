import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time

# --- Configuration ---
st.set_page_config(
    page_title="YOLOv8 Vehicle Tracking & Speed",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global configuration constants
# COCO classes for vehicles: 2: car, 3: motorcycle, 5: bus, 7: truck
CLASSES_TO_TRACK = [2, 3, 5, 7] 
# Speed calculation constants (Meters per Second to Km/Hour)
MTS_TO_KMH = 3.6

# Virtual Counting Lines (Normalized 0 to 1000)
# Speed is calculated between line_start and line_end
LINE_START_NORM = 350 # Start line for speed calculation
LINE_END_NORM = 750   # End line for speed calculation

# --- Data Structures for Tracking and Speed ---
class VehicleData:
    """Stores persistent data for a single tracked vehicle."""
    def __init__(self, track_id):
        self.track_id = track_id
        self.start_frame = None
        self.end_frame = None
        self.speed_kmh = 0.0
        self.is_counted = False
        self.direction = "N/A"

# Global dictionary to store data for all tracked vehicles
vehicle_registry = {}


# --- Core Functions ---

@st.cache_resource
def load_model():
    """Loads the YOLOv8 nano model for fast inference."""
    return YOLO('yolov8n.pt')

model = load_model()

def calculate_speed(pixel_distance, total_frames, fps, ppm_factor):
    """
    Calculates speed in Km/H based on pixel distance, time, and PPM factor.
    
    Formula: Speed (m/s) = (Pixel Distance / PPM Factor) / Time (seconds)
    Time (seconds) = Total Frames / FPS
    """
    if total_frames <= 0 or fps <= 0 or ppm_factor <= 0:
        return 0.0

    # 1. Calculate the time elapsed in seconds
    time_seconds = total_frames / fps

    # 2. Calculate the real-world distance in meters
    distance_meters = pixel_distance / ppm_factor

    # 3. Calculate speed in meters/second and convert to km/h
    speed_ms = distance_meters / time_seconds
    speed_kmh = speed_ms * MTS_TO_KMH

    return speed_kmh

def process_video_stream(input_video_path, output_video_path, fps, width, height, ppm_factor):
    """Runs the YOLOv8 tracking and speed estimation pipeline."""
    global vehicle_registry
    vehicle_registry = {} # Reset registry for new video
    
    st.info("Starting analysis: Detection, Tracking, and Speed Estimation...")
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {input_video_path}")
        return False
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="Processing video frames...")
    
    # Calculate pixel positions of virtual lines
    line_start_y = int(height * (LINE_START_NORM / 1000))
    line_end_y = int(height * (LINE_END_NORM / 1000))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- AI CORE: Detection and Tracking ---
        results = model.track(
            frame, 
            persist=True, 
            tracker='bytetrack.yaml',
            classes=CLASSES_TO_TRACK,
            verbose=False
        )
        
        # Draw virtual lines on the frame
        cv2.line(frame, (0, line_start_y), (width, line_start_y), (0, 255, 255), 2) # Yellow
        cv2.line(frame, (0, line_end_y), (width, line_end_y), (0, 0, 255), 2)     # Red
        
        # --- METRIC CALCULATION LOGIC ---
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # Bounding box coordinates (x_center, y_center, width, height)
            boxes = results[0].boxes.xywh.cpu().numpy() 

            for box, track_id in zip(boxes, track_ids):
                x_c, y_c, w, h = box 
                
                # Check registry for new/existing vehicle
                if track_id not in vehicle_registry:
                    vehicle_registry[track_id] = VehicleData(track_id)
                
                vehicle = vehicle_registry[track_id]
                
                # 1. Start Measurement
                # Vehicle's bottom center crosses the start line
                if y_c > line_start_y and vehicle.start_frame is None:
                    vehicle.start_frame = frame_count
                    vehicle.start_y = y_c

                # 2. End Measurement and Calculate Speed
                # Vehicle's bottom center crosses the end line AND has a start time
                if y_c > line_end_y and vehicle.start_frame is not None and vehicle.end_frame is None:
                    vehicle.end_frame = frame_count
                    vehicle.end_y = y_c
                    
                    # Calculate total pixel distance travelled (y-axis only for simple vertical movement)
                    pixel_distance = abs(vehicle.end_y - vehicle.start_y)
                    
                    # Calculate frames elapsed
                    frames_elapsed = vehicle.end_frame - vehicle.start_frame
                    
                    # Calculate and store speed
                    vehicle.speed_kmh = calculate_speed(pixel_distance, frames_elapsed, fps, ppm_factor)
                    vehicle.is_counted = True # Mark as complete
                
                # Update annotation text
                speed_text = f"{vehicle.speed_kmh:.1f} km/h" if vehicle.speed_kmh > 0 else f"ID {track_id}"
                
                # Draw speed label on the frame (manual plotting needed for custom text)
                (text_w, text_h), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x_c - w/2), int(y_c - h/2 - text_h - 10)), (int(x_c - w/2) + text_w + 10, int(y_c - h/2 - 5)), (0, 0, 0), -1)
                cv2.putText(frame, speed_text, (int(x_c - w/2 + 5), int(y_c - h/2 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        # Draw detection results (Bounding boxes and IDs are drawn by the model.plot() method)
        annotated_frame = results[0].plot(img=frame)
        
        # Write the final frame with speed and lines
        out.write(annotated_frame)
        
        frame_count += 1
        if total_frames > 0:
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress, text=f"Processing frame {frame_count}/{total_frames}...")

    # Release resources
    cap.release()
    out.release()
    progress_bar.empty()
    st.success(f"Processing complete! {frame_count} frames analyzed.")
    return True

# --- Streamlit Frontend UI ---

def main():
    st.title("🚦 YOLOv8 Vehicle Tracking and Speed Analysis")
    st.markdown(
        """
        This application uses your **Multi-Object Tracking Framework** (YOLOv8 + Kalman-based tracking) to detect vehicles and estimate their speed.
        
        **Virtual Calibration is Required:** Speed calculation relies on the **Pixels Per Meter (PPM) factor**, which you must estimate based on your video's camera angle.
        """
    )

    # --- Sidebar for Configuration ---
    st.sidebar.header("1. Upload Video")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video file (MP4, MOV)", 
        type=["mp4", "mov", "avi"]
    )

    st.sidebar.header("2. Calibration Settings")
    ppm_factor = st.sidebar.number_input(
        "Pixels Per Meter (PPM) Factor:",
        min_value=1.0,
        value=50.0,
        step=5.0,
        format="%f",
        help="Estimate of how many pixels correspond to 1 meter in the video scene. Adjust this value to calibrate speed."
    )
    
    if uploaded_file is not None:
        
        # 1. Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            input_video_path = tfile.name
        
        st.sidebar.success(f"File uploaded: {uploaded_file.name}")

        # Get video properties
        cap = cv2.VideoCapture(input_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Define the output path in another temporary file
        temp_dir = tempfile.gettempdir()
        output_video_path = os.path.join(temp_dir, f"tracked_{os.path.basename(input_video_path)}")

        st.subheader("Results")
        col1, col2 = st.columns([1, 2])
        
        # 3. Process the video
        with st.spinner("Analyzing traffic video..."):
            if process_video_stream(input_video_path, output_video_path, fps, width, height, ppm_factor):
                
                # --- Metric Display (Final Summary) ---
                speeds = [v.speed_kmh for v in vehicle_registry.values() if v.speed_kmh > 0]
                
                total_vehicles = len([v for v in vehicle_registry.values() if v.is_counted])
                avg_speed = np.mean(speeds) if speeds else 0
                max_speed = np.max(speeds) if speeds else 0

                with col1:
                    st.metric("Total Vehicles Tracked & Counted", f"{total_vehicles}")
                    st.metric("Average Speed Estimated", f"{avg_speed:.1f} km/h")
                    st.metric("Max Speed Recorded", f"{max_speed:.1f} km/h")
                    st.info(f"Using a PPM factor of **{ppm_factor}**")
                    st.markdown("---")
                    st.caption("Lines: Yellow (Start), Red (End)")
                
                # 4. Read and display the processed video
                with col2:
                    st.subheader("Processed Video Output")
                    with open(output_video_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    st.video(video_bytes, format='video/mp4', start_time=0)
                    
                    st.download_button(
                        label="Download Processed Video",
                        data=video_bytes,
                        file_name=f"tracked_{uploaded_file.name}",
                        mime="video/mp4"
                    )

        # 5. Cleanup temporary files
        os.remove(input_video_path)
        if os.path.exists(output_video_path):
             os.remove(output_video_path)

    else:
        st.info("Please upload a traffic video file and set the PPM factor to begin analysis.")
        st.sidebar.markdown("---")
        st.sidebar.caption("The tracking framework runs YOLOv8 and ByteTrack to estimate speed between the two virtual lines.")

if __name__ == '__main__':
    main()
