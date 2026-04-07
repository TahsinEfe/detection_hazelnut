import os
import sys
import cv2
import warnings
import csv

# system and model configurations
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
    MIN_TYPE_MARGIN,
    MIN_BBOX_AREA,
    MAX_BBOX_AREA,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
    CROP_PADDING,
    PREPARED_TARGET_SIZE,
    PREPARED_PADDING_RATIO,
)
from detector import HazelnutDetector
from classifier import HazelnutClassifier
from decision import DecisionEngine

def get_image_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return os.path.join("test_images", "test.jpg")

def build_output_paths(image_path):
    os.makedirs("outputs", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return {
        "crop": os.path.join("outputs", f"{base_name}_crop.jpg"),
        "prepared_crop": os.path.join("outputs", f"{base_name}_prepared_crop.jpg"),
        "result": os.path.join("outputs", f"{base_name}_result.jpg"),
        "invalid_result": os.path.join("outputs", f"{base_name}_invalid_result.jpg"),
    }

def log_to_csv(image_path, final_decision, reason, det_conf=None, type_conf=None, aspect_ratio=None):
    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", "results.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "image", "final_decision", "reason", 
                "detector_confidence", "type_confidence", "aspect_ratio"
            ])
        writer.writerow([
            os.path.basename(image_path),
            final_decision,
            reason,
            f"{det_conf:.4f}" if det_conf is not None else None,
            f"{type_conf:.4f}" if type_conf is not None else None,
            f"{aspect_ratio:.4f}" if aspect_ratio is not None else None
        ])

#  show_or_save function to handle both display and saving of results
def show_or_save(display, output_path, window_name="Test Result", wait_ms=4000):
    force_save = os.environ.get("FORCE_SAVE", "0") == "1"

    if force_save:
        cv2.imwrite(output_path, display)
        print(f"FORCE_SAVE active, saved image to: {output_path}")
        return

    has_display = False

    if os.name == "nt":
        has_display = True
    else:
        if os.environ.get("DISPLAY"):
            has_display = True

    if has_display:
        try:
            cv2.imshow(window_name, display)
            cv2.waitKey(wait_ms)
            cv2.destroyAllWindows()
            cv2.imwrite(output_path, display)
            return
        except Exception:
            pass

    cv2.imwrite(output_path, display)
    print(f"GUI not available, saved image to: {output_path}")

def main():
    os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY

    image_path = get_image_path()
    output_paths = build_output_paths(image_path)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Image loaded: {image_path}")
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

    print("Models loaded.\nRunning detector...")

    detection = detector.detect(image)
    detection_eval = decision_engine.evaluate_detection(detection, image.shape)
    display = image.copy()

    if not detection_eval["valid"]:
        print("\nFinal Decision: invalid")
        print(f"Reason: {detection_eval['reason']}")

        log_to_csv(
            image_path,
            "invalid",
            detection_eval["reason"],
            None, None, None
        )

        cv2.putText(display, "Decision: invalid", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display, detection_eval["reason"], (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        show_or_save(display, output_paths["invalid_result"], wait_ms=4000)
        return

    x1, y1, x2, y2 = detection_eval["bbox"]
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        print("\nFinal Decision: invalid (Empty crop)")
        log_to_csv(image_path, "invalid", "Empty crop")
        return

    cv2.imwrite(output_paths["crop"], crop)
    print(f"Detector confidence: {detection_eval['detector_confidence']:.4f}")

    print("\nRunning classifier...")
    classification_result = classifier.classify(crop)
    final_eval = decision_engine.evaluate_classification(classification_result)

    final_decision = final_eval["final_decision"]
    reason = final_eval["reason"]

    type_conf = None
    if classification_result.get("prediction"):
        type_conf = classification_result["prediction"]["confidence"]

    log_to_csv(
        image_path,
        final_decision,
        reason,
        detection_eval["detector_confidence"],
        type_conf,
        detection_eval["aspect_ratio"]
    )

    print(f"\nFinal Decision: {final_decision}")
    print(f"Reason: {reason}")

    prepared_crop = classification_result["prepared_crop"]
    cv2.imwrite(output_paths["prepared_crop"], prepared_crop)

    color = (0, 255, 0) if final_decision != "invalid" else (0, 0, 255)
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
    cv2.putText(display, f"Decision: {final_decision}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(display, reason, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # this will display the image if possible, otherwise it will save the result image to disk
    show_or_save(display, output_paths["result"], wait_ms=4000)

if __name__ == "__main__":
    main()