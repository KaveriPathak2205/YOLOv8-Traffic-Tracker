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
# Overspeeding Threshold
OVERSPEED_LIMIT_KMH = 20.0

# Virtual Counting Lines (Normalized 0 to 1000)
# Speed is calculated between line_start and line_end (Y-coordinates)
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
        self.last_x = None # New: for directional tracking
        self.direction = None # New: 'Left-to-Right' or 'Right-to-Left'

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
    
    # st.info("Starting analysis: Detection, Tracking, Speed, and Directional Counting...")
    
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
    
    # NEW: Vertical Center Line for Directional Tracking
    center_x = width // 2
    
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
        cv2.line(frame, (0, line_start_y), (width, line_start_y), (0, 255, 255), 2) # Yellow (Start)
        cv2.line(frame, (0, line_end_y), (width, line_end_y), (0, 0, 255), 2)     # Red (End)
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 0), 1)      # Cyan (Center X)
        
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
                
                # 1. Determine Direction (Before tracking start/end)
                if vehicle.last_x is not None:
                    if x_c > vehicle.last_x:
                        # Vehicle is currently moving right
                        vehicle.direction = 'Left-to-Right' 
                    elif x_c < vehicle.last_x:
                        # Vehicle is currently moving left
                        vehicle.direction = 'Right-to-Left'
                
                vehicle.last_x = x_c # Store current X position

                # 2. Start Speed Measurement (Y-axis crossing)
                if y_c > line_start_y and vehicle.start_frame is None:
                    vehicle.start_frame = frame_count
                    vehicle.start_y = y_c

                # 3. End Speed Measurement and Calculate Speed
                if y_c > line_end_y and vehicle.start_frame is not None and vehicle.end_frame is None:
                    vehicle.end_frame = frame_count
                    vehicle.end_y = y_c
                    
                    # Calculate total pixel distance travelled (y-axis only for simple vertical movement)
                    pixel_distance = abs(vehicle.end_y - vehicle.start_y)
                    
                    # Calculate frames elapsed
                    frames_elapsed = vehicle.end_frame - vehicle.start_frame
                    
                    # Calculate and store speed
                    vehicle.speed_kmh = calculate_speed(pixel_distance, frames_elapsed, fps, ppm_factor)
                    vehicle.is_counted = True # Mark as complete for summary

                
                # Update annotation text
                speed_text_color = (255, 255, 255) # White default
                
                if vehicle.speed_kmh > 0:
                    # Check for overspeeding to change color
                    if vehicle.speed_kmh > OVERSPEED_LIMIT_KMH:
                        speed_text_color = (0, 0, 255) # Red for overspeeding
                    
                    # Show direction (L2R or R2L) based on the first letter of the direction stored
                    dir_abbr = vehicle.direction.split('-')[0][0] if vehicle.direction else '?'
                    speed_text = f"{vehicle.speed_kmh:.1f} km/h ({dir_abbr})"
                else:
                    speed_text = f"ID {track_id}"
                
                # Draw speed label on the frame
                (text_w, text_h), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Use bounding box position for text placement
                # (x_c - w/2) is the top-left x corner of the bounding box
                text_x = int(x_c - w/2)
                text_y_top = int(y_c - h/2 - 10)
                
                cv2.rectangle(frame, (text_x, text_y_top - text_h - 5), (text_x + text_w + 10, text_y_top), (0, 0, 0), -1)
                cv2.putText(frame, speed_text, (text_x + 5, text_y_top - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_text_color, 2)


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
        This application uses your **Multi-Object Tracking Framework** (YOLOv8 + Kalman-based tracking) to detect vehicles, estimate their speed, and count them by direction.
        
        **Note:** Speed estimation requires the **Pixels Per Meter (PPM) factor** to be set in the sidebar for accurate results.
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
        # Use tempfile.NamedTemporaryFile for input
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile_in:
            tfile_in.write(uploaded_file.read())
            input_video_path = tfile_in.name
        
        st.sidebar.success(f"File uploaded: {uploaded_file.name}")

        # Get video properties
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            st.error("Could not read video metadata.")
            os.remove(input_video_path)
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # 2. Define the output file path in a robust way
        output_video_path = None
        
        try:
            # Use a NamedTemporaryFile for the output video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile_out:
                output_video_path = tfile_out.name
            
                st.subheader("Results")
                col1, col2 = st.columns([1, 2])
                
                # 3. Process the video
                if process_video_stream(input_video_path, output_video_path, fps, width, height, ppm_factor):
                    
                    # --- Metric Calculation (Final Summary) ---
                    speeds = [v.speed_kmh for v in vehicle_registry.values() if v.speed_kmh > 0]
                    
                    total_vehicles = len([v for v in vehicle_registry.values() if v.is_counted])
                    
                    # Directional Counts
                    left_to_right_count = len([v for v in vehicle_registry.values() 
                                               if v.is_counted and v.direction == 'Left-to-Right'])
                    right_to_left_count = len([v for v in vehicle_registry.values() 
                                               if v.is_counted and v.direction == 'Right-to-Left'])
                    
                    avg_speed = np.mean(speeds) if speeds else 0
                    max_speed = np.max(speeds) if speeds else 0
                    
                    # --- New: Overspeeding Check ---
                    overspeeding_vehicles = [
                        (v.track_id, v.speed_kmh) 
                        for v in vehicle_registry.values() 
                        if v.speed_kmh > OVERSPEED_LIMIT_KMH
                    ]

                    # Display Metrics
                    with col1:
                        col1_r1, col2_r1 = st.columns(2)
                        with col1_r1:
                             st.metric("Total Vehicles Counted", f"{total_vehicles}")
                        with col2_r1:
                             st.metric("Average Speed Estimated", f"{avg_speed:.1f} km/h")
                        
                        st.markdown("---")
                        
                        # New: Overspeeding Alert Section
                        st.subheader("🚨 Overspeeding Alerts")
                        if overspeeding_vehicles:
                            st.warning(f"**{len(overspeeding_vehicles)}** vehicles exceeded {OVERSPEED_LIMIT_KMH} km/h!")
                            
                            # Prepare data for display
                            overspeed_data = [
                                {"ID": id, "Speed (km/h)": f"{speed:.1f}"} 
                                for id, speed in overspeeding_vehicles
                            ]
                            
                            # Display as a table/dataframe
                            st.dataframe(overspeed_data, use_container_width=True, hide_index=True)
                        else:
                            st.success(f"No vehicles detected over {OVERSPEED_LIMIT_KMH} km/h.")
                        
                        st.markdown("---")
                        st.caption("Directional Traffic Counts")
                        col1_r2, col2_r2 = st.columns(2)
                        with col1_r2:
                             st.metric("⬅️ Right to Left", f"{right_to_left_count}")
                        with col2_r2:
                             st.metric("➡️ Left to Right", f"{left_to_right_count}")
                        
                        st.markdown("---")
                        st.metric("Max Speed Recorded", f"{max_speed:.1f} km/h")
                        st.info(f"Using a PPM factor of **{ppm_factor}**")
                        st.caption("Lines: Yellow (Start Y), Red (End Y), Cyan (Center X)")
                    
                    # 4. Read and display the processed video
                    with col2:
                        st.subheader("Processed Video Output")
                        
                        # Read the file bytes directly for display
                        with open(output_video_path, 'rb') as f:
                            video_bytes = f.read()
                        
                        # Display the video
                        st.video(video_bytes, format='video/mp4', start_time=0)
                        
                        # Download button
                        st.download_button(
                            label="Download Processed Video",
                            data=video_bytes,
                            file_name=f"tracked_{uploaded_file.name}",
                            mime="video/mp4"
                        )
                
        finally:
            # 5. Cleanup temporary files regardless of success/failure
            if os.path.exists(input_video_path):
                os.remove(input_video_path)
            if output_video_path and os.path.exists(output_video_path):
                # Ensure the file is not deleted until after st.video has completed reading
                os.remove(output_video_path)

    else:
        st.info("Please upload a traffic video file and set the PPM factor to begin analysis.")
        st.sidebar.markdown("---")
        st.sidebar.caption("The tracking framework runs YOLOv8 and ByteTrack to estimate speed and direction.")

if __name__ == '__main__':
    main()

