import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# --- Configuration ---
# Set page configuration for a wider, more professional look
st.set_page_config(
    page_title="YOLOv8 Vehicle Tracking & Speed",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the AI model once
@st.cache_resource
def load_model():
    """Loads the YOLOv8 nano model for fast inference."""
    # Using the nano model for speed, suitable for real-time applications
    # NOTE: You can change 'yolov8n.pt' to 'yolov8s.pt' for better accuracy but slower speed
    return YOLO('yolov8n.pt')

model = load_model()
CLASSES_TO_TRACK = [2, 3, 5, 7] # COCO classes: 2: car, 3: motorcycle, 5: bus, 7: truck

# --- Core Processing Function ---

def process_video_stream(input_video_path, output_video_path, fps, width, height):
    """
    Runs the YOLOv8 tracking pipeline on the input video.
    
    NOTE: This is the core logic from your Jupyter notebook, adapted for Streamlit.
    Actual Counting and Speed Logic (using virtual lines) would be integrated here.
    """
    st.info("Starting real-time vehicle detection and tracking...")
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {input_video_path}")
        return False
    
    # Define the codec and create VideoWriter object
    # Using 'mp4v' or 'avc1' is usually necessary for web compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    progress_bar = st.progress(0, text="Processing video frames...")
    
    # Calculate total frames for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
         st.warning("Could not determine total frames. Processing without progress.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- AI CORE: Detection and Tracking ---
        # The 'persist=True' flag ensures tracking history is maintained across frames.
        # The 'tracker' argument selects the tracking algorithm (e.g., 'bytetrack.yaml')
        # The 'classes' argument filters detections to focus only on vehicles.
        results = model.track(
            frame, 
            persist=True, 
            tracker='bytetrack.yaml',
            classes=CLASSES_TO_TRACK,
            verbose=False # Suppress console output during inference
        )
        
        # Draw results (bounding boxes, IDs) onto the frame
        annotated_frame = results[0].plot()

        # --- Metric Calculation Placeholder (Future Enhancement) ---
        # In a full deployment, the custom counting and speed logic would go here:
        # 1. Get tracked IDs and coordinates: results[0].boxes.id and results[0].boxes.xywh
        # 2. Use a helper function to check if the vehicle crossed the virtual line.
        # 3. Use frame_count and fps to estimate speed (km/h).
        # 4. Display the live count and speed on the 'annotated_frame'.
        
        # Save annotated frame
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
        This application implements your **Multi-Object Tracking Framework** using **YOLOv8** and a **Kalman-based tracker** to detect, track, and provide foundational data for speed estimation in traffic videos.
        """
    )

    # File Uploader in the Sidebar
    st.sidebar.header("Upload Video")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video file (MP4, MOV)", 
        type=["mp4", "mov", "avi"]
    )

    if uploaded_file is not None:
        # 1. Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            input_video_path = tfile.name

        st.sidebar.success(f"File uploaded: {uploaded_file.name}")
        
        # Get video properties to configure the output writer
        cap = cv2.VideoCapture(input_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        st.info(f"Video Properties: {width}x{height} pixels @ {fps} FPS.")
        
        # 2. Define the output path in another temporary file
        temp_dir = tempfile.gettempdir()
        output_video_path = os.path.join(temp_dir, f"tracked_{os.path.basename(input_video_path)}")

        st.subheader("Results")
        # Placeholder for the final video display
        video_placeholder = st.empty()
        
        # 3. Process the video
        if process_video_stream(input_video_path, output_video_path, fps, width, height):
            # 4. Read and display the processed video
            
            # Note: For reliable web deployment, it is often necessary to convert
            # the output video to a format that Streamlit/browsers prefer. 
            # We are relying on cv2.VideoWriter_fourcc(*'mp4v') to handle this.
            
            with open(output_video_path, 'rb') as f:
                video_bytes = f.read()
            
            video_placeholder.video(video_bytes, format='video/mp4', start_time=0)
            
            # Add a download button
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
        st.info("Please upload a traffic video file to begin analysis.")
        st.sidebar.markdown("---")
        st.sidebar.caption("The app will process the video, perform tracking, and display the result.")

if __name__ == '__main__':
    main()
