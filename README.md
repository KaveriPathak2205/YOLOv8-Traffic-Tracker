# YOLOv8-Traffic-Tracker


## Repository Description

**Repository Name:** `YOLOv8-Traffic-Tracker`

**Short Description (GitHub Summary):**

> A robust Multi-Object Tracking (MOT) framework for real-time traffic analysis using YOLOv8, Kalman filtering, and Streamlit for web visualization.


## Detailed README Content (Repository Description)

### 🚥 A Robust Multi-Object Tracking Framework for Traffic Analysis

This repository contains the complete implementation of a Computer Vision pipeline designed for **Intelligent Transportation Systems (ITS)**. The framework efficiently performs real-time vehicle detection, tracking, counting, and provides the foundational data for speed estimation from standard video surveillance footage.

### Key Technologies Used

| Technology | Role |
| :--- | :--- |
| **Detection** | **YOLOv8** (from `ultralytics`) for fast and accurate vehicle localization and classification. |
| **Tracking** | **Kalman Filter-based Re-Identification** (ByteTrack or similar) for maintaining persistent vehicle IDs across occlusions and dense traffic. |
| **Interface** | **Streamlit** for a user-friendly web interface allowing video upload, processing, and display of results. |
| **Core Libraries** | **OpenCV (`cv2`)** for video input/output and frame manipulation. |

### Core Functionality

1.  **Real-Time Detection:** Accurately identifies and classifies common vehicle types (cars, trucks, buses) in every video frame.
2.  **Persistent ID Tracking:** Assigns a unique, stable ID to each vehicle. The tracker uses predictive filtering (Kalman) and association logic to ensure the ID is maintained even when vehicles briefly overlap or are partially obscured.
3.  **Data Generation for Metrics:** Outputs annotated video frames with bounding boxes and IDs. This structured output is used to power the final analysis layer:
      * **Counting:** Enables precise, directional counting by tracking IDs across virtual lines.
      * **Speed Estimation:** Provides the necessary data (position history, frame time) for calculating real-world speed via scene calibration.

### How to Run Locally

1.  **Clone the repository.**
    ```bash
    git clone https://github.com/YourUsername/YOLOv8-Traffic-Tracker.git
    cd YOLOv8-Traffic-Tracker
    ```
2.  **Install Dependencies** (Ensure you have a suitable Python environment, preferably using a virtual environment).
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the Streamlit App.**
    ```bash
    streamlit run app.py
    ```
4.  Upload your traffic video file via the web interface to see the real-time tracking in action.

### Deployment Status

[](https://www.google.com/search?q=Your-Streamlit-Cloud-Link-Goes-Here)
*(Replace the placeholder link with the actual URL after deploying to Streamlit Cloud.)*
