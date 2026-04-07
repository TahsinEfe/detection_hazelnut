# Hazelnut Detection & Classification System

This project is a computer vision system that detects and classifies hazelnuts (e.g., determining their types or health status) in real-time using a camera feed. The system leverages the Roboflow Inference infrastructure and utilizes OpenCV for real-time video processing. It features optimized configuration options for various environments, such as high-performance Desktop PCs or constrained devices like Raspberry Pis.

## Architecture & Layers

The project has a robust and clean architecture where tasks are separated into modular, independent classes:

### 1. Configuration Layer (`config.py`)
This central configuration layer stores environment-specific operational parameters (confidence thresholds, bounding box size limits, etc.) for both `desktop` and `raspberry_pi` settings, as well as necessary API keys.

### 2. Detection Layer (`detector.py`)
- **`HazelnutDetector` Class**: Responsible for finding the positions (bounding boxes) of hazelnuts in the image and calculating their detection confidence scores. It operates as an independent module, meaning the underlying detection model can be easily swapped out.

### 3. Classification Layer (`classifier.py`)
- **`HazelnutClassifier` Class**: Analyzes the cropped hazelnut images extracted by the detector and performs classification. It applies preprocessing to adapt the image for the classification model and returns the in-class prediction confidence levels.

### 4. Decision Engine Layer (`decision.py`)
- **`DecisionEngine` Class**: Acts as the ultimate "judge" layer, auditing and validating the outputs from both the detection and classification steps.
  - **Detection Validation**: Verifies the bounding box area and aspect ratio against configured constraints.
  - **Classification Validation**: Evaluates classification confidence scores and the margin between the Top-1 and Top-2 predictions to produce the final valid/invalid decision. This prevents false classifications caused by improper crops or misleading backgrounds.

### 5. Application & Interface Layer (`main.py`)
Houses the main operational pipeline loop. It captures real-time frames from the webcam, sequentially triggers detection, cropping, classification, and the decision mechanisms, and then feeds the results back to the screen live via OpenCV (including bounding boxes, text labels, and color codes).

### 6. Utility Functions (`utils.py`)
Contains the central logic and helper functions used to prepare images for the models (e.g., cropping, scaling, padding), such as the `prepare_for_type_model` function.

##  Model Training (Roboflow)

The detection and classification models were trained using Roboflow, following a structured pipeline:

### 1. Dataset Preparation
- Images of hazelnuts were collected from real-world scenarios.
- A total dataset of 1000+ images was used.
- Data includes variations such as:
  - Different lighting conditions
  - Background noise
  - Orientation and shape differences

### 2. Annotation
- Bounding boxes were manually labeled for each hazelnut.
- Each hazelnut instance was annotated as an object.
- Annotation was performed directly in Roboflow.

### 3. Preprocessing
- Auto-orientation applied.
- Image resizing (e.g., 640x640 for detection).
- Normalization handled automatically by Roboflow.

### 4. Data Augmentation
To improve generalization:
- Rotation
- Horizontal/vertical flips
- Brightness & contrast adjustments
- Zoom and scaling transformations

### 5. Model Training
**Detection Model**
- **Model Type:** YOLOv8
- **Task:** Object Detection
- **Output:** Bounding boxes + confidence scores

**Classification Model**
- **Task:** Image Classification
- **Input:** Cropped hazelnut images from the detection stage
- **Output:** Class probabilities (e.g., Palaz, Çakıldak, Uzun Musa, Yassı Badem)

### 6. Deployment
- Models were deployed via Roboflow Inference API.
- Integrated directly into the Python pipeline.
- Real-time inference achieved through API calls.

##  Key Technologies
- **Python 3**
- **Roboflow Inference:** For running machine learning models.
- **OpenCV:** For image processing and real-time display.
- **Supervision:** For handling analytic data in the computer vision pipeline.

## 🛠 Setup and Execution

### Prerequisites
Make sure to install the required dependencies before running the application. (If necessary, add your Roboflow API Key inside `config.py`).

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd detection_hazelnut
   ```

2. Install the required libraries in your virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```
*(You can press `ESC` or `Q` to close the application.)*

##  Configuration & Integration
To alter the behavior of the system, update the `MODE` variable in `config.py`. This adjusts operational limits according to your device's processing capabilities:
- `MODE = "desktop"` (Optimized default thresholds for high-performance desktop processing)
- `MODE = "raspberry_pi"` (More lenient thresholds optimized for low-end hardware)

---
*This project serves as an example of a modular pipeline composed of independent and expandable components.*
