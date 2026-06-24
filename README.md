# Hazelnut Detection & Classification System

This project is a computer vision system that detects and classifies hazelnuts in real time using a camera feed. It leverages Roboflow Inference and OpenCV, and it is organized into modular components so the detection, classification, and decision logic can be maintained independently.

## Architecture & Core Classes

The repository is built around a clean, layered architecture with clearly separated responsibilities:

### `config.py` — Configuration Layer
- Stores API keys, model identifiers, environment flags, and operational thresholds.
- Supports both `desktop` and `raspberry_pi` modes with device-specific settings.

### `detector.py` — Detection Layer
- **`HazelnutDetector`** detects hazelnut locations in the input image.
- Uses a Roboflow detection endpoint to return the highest-confidence bounding box.
- Outputs a normalized bounding box and a confidence score.

### `classifier.py` — Classification Layer
- **`HazelnutClassifier`** classifies the cropped hazelnut image.
- Preprocesses the crop for the classification model and sends it to Roboflow.
- Returns top prediction, all class confidences, and raw model output.

### `decision.py` — Decision Engine Layer
- **`DecisionEngine`** validates both detection and classification results.
- Detection validation checks bounding box size, aspect ratio, and confidence.
- Classification validation checks prediction confidence and top-1/top-2 margin.
- Produces a final decision of either a valid hazelnut class or `invalid`.

### `main.py` — Application Layer
- Runs the main camera loop.
- Captures frames, calls the detector, crops the result, classifies the crop, and applies decision logic.
- Displays bounding boxes and live status text using OpenCV.

### `utils.py` — Utility Helpers
- Contains reusable image preprocessing utilities such as `prepare_for_type_model`.
- Keeps the model pipelines consistent by centralizing resize, padding, and normalization logic.

## Example Result Images
Below are sample outputs from the pipeline, using the images stored in `result_images/`.

### Detection and Classification Results

![Result 2](result_images/Screenshot%202026-06-24%20at%2016.52.23.png)
*Example UI overlay showing detected hazelnut type.*

![Result 3](result_images/Screenshot%202026-06-24%20at%2016.52.54.png)
*Another run showing the system's live classification feedback.*

![Result 4](result_images/Screenshot%202026-06-24%20at%2016.53.54.png)
*Final example illustrating the real-time bounding box and decision visualization.*

![Result 5](result_images/Screenshot%202026-06-24%20at%2017.04.47.png)
*Updated result showing the latest detection and classification output.*

## Model Training and Deployment

The detection and classification models were trained using Roboflow with a complete pipeline that includes dataset preparation, annotation, augmentation, and deployment.

### Training Workflow
- Dataset contains hazelnut images across lighting and orientation variations.
- Bounding boxes were annotated in Roboflow.
- Data augmentation included rotations, flips, brightness changes, and zoom.
- Detection model is trained for object localization.
- Classification model is trained on cropped hazelnut images.

### Deployment
- Models are deployed through Roboflow Inference.
- The Python pipeline sends images directly to the API for real-time inference.

## Key Technologies
- Python 3
- Roboflow Inference
- OpenCV
- Optional UI support via `ui_camera.py` and `ui_batch.py`

## Setup and Execution

### Prerequisites
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd detection_hazelnut
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

### Configuration
Set `MODE` in `config.py`:
- `MODE = "desktop"` for desktop usage.
- `MODE = "raspberry_pi"` for low-power hardware.

## Notes
- Replace the placeholder Roboflow API key in `config.py` with your own key for inference to work.
- Press `ESC` or `Q` to close the live display.
- `result_images/` contains example output screenshots and sample results.
