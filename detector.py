import inference
import numpy as np
import supervision as sv


class HazelnutDetector:
    def __init__(self, model_id: str):
        self.model = inference.get_model(model_id=model_id)

    def detect(self, image):
        result = self.model.infer(image)[0]
        detections = sv.Detections.from_inference(result)

        if len(detections) == 0:
            return None

        best_idx = int(np.argmax(detections.confidence))
        best_conf = float(detections.confidence[best_idx])
        x1, y1, x2, y2 = detections.xyxy[best_idx].astype(int)

        return {
            "confidence": best_conf,
            "bbox": (x1, y1, x2, y2),
            "raw_result": result
        }