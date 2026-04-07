import os
import cv2
import warnings

os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from config import (
    ROBOFLOW_API_KEY,
    DETECTOR_MODEL_ID,
    TYPE_MODEL_ID,
    DET_THRESHOLD,
    TYPE_THRESHOLD,
    MIN_BBOX_AREA,
    MAX_BBOX_AREA,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
    CROP_PADDING,
    PREPARED_TARGET_SIZE,
    PREPARED_PADDING_RATIO,
    FRAME_SKIP,
    CAMERA_INDEX,
    MIN_TYPE_MARGIN
)
from detector import HazelnutDetector
from classifier import HazelnutClassifier
from decision import DecisionEngine


def main():
    os.environ["********"] = ROBOFLOW_API_KEY

    print("Loading models...")
    detector = HazelnutDetector(DETECTOR_MODEL_ID)
    classifier = HazelnutClassifier(
        TYPE_MODEL_ID,
        target_size=PREPARED_TARGET_SIZE,
        padding_ratio=PREPARED_PADDING_RATIO
    )
    decision_engine = DecisionEngine(
        det_threshold=DET_THRESHOLD,
        type_threshold=TYPE_THRESHOLD,
        min_bbox_area=MIN_BBOX_AREA,
        max_bbox_area=MAX_BBOX_AREA,
        min_aspect_ratio=MIN_ASPECT_RATIO,
        max_aspect_ratio=MAX_ASPECT_RATIO,
        crop_padding=CROP_PADDING,
        min_type_margin=MIN_TYPE_MARGIN
    )
    print("Models loaded.")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Webcam could not be opened.")
        return

    frame_count = 0
    current_label = "waiting..."
    current_box = None
    current_info = "Press Q or ESC to quit"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame could not be read.")
                break

            frame_count += 1

            if frame_count % FRAME_SKIP == 0:
                try:
                    detection = detector.detect(frame)
                    detection_eval = decision_engine.evaluate_detection(detection, frame.shape)

                    if not detection_eval["valid"]:
                        current_label = "invalid"
                        current_box = None
                        current_info = detection_eval["reason"]
                    else:
                        x1, y1, x2, y2 = detection_eval["bbox"]
                        crop = frame[y1:y2, x1:x2]

                        if crop.size == 0:
                            current_label = "invalid"
                            current_box = None
                            current_info = "Empty crop"
                        else:
                            classification_result = classifier.classify(crop)
                            final_eval = decision_engine.evaluate_classification(classification_result)

                            current_label = final_eval["final_decision"]
                            current_box = (x1, y1, x2, y2)
                            current_info = final_eval["reason"]

                except Exception as e:
                    current_label = "error"
                    current_box = None
                    current_info = str(e)

            display = frame.copy()

            if current_box is not None:
                x1, y1, x2, y2 = current_box
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)

            color = (0, 255, 0) if current_label not in ["invalid", "error"] else (0, 0, 255)

            cv2.putText(
                display,
                f"Decision: {current_label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            cv2.putText(
                display,
                current_info,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Hazelnut Classifier", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print("App is shutting down...")
                break

    except KeyboardInterrupt:
        print("\nApp is shutting down...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera and windows released.")

if __name__ == "__main__":
    main()