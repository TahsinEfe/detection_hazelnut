import inference
import numpy as np
import supervision as sv

from utils import prepare_for_type_model


class HazelnutClassifier:
    def __init__(self, model_id: str, target_size: int, padding_ratio: float):
        self.model = inference.get_model(model_id=model_id)
        self.target_size = target_size
        self.padding_ratio = padding_ratio

    def classify(self, crop):
        prepared_crop = prepare_for_type_model(
            crop,
            target_size=self.target_size,
            padding_ratio=self.padding_ratio
        )

        result = self.model.infer(prepared_crop)[0]
        detections = sv.Detections.from_inference(result)

        if len(detections) == 0:
            return {
                "prepared_crop": prepared_crop,
                "prediction": None,
                "all_predictions": [],
                "raw_result": result
            }

        all_predictions = []
        for i in range(len(detections)):
            class_name = detections.data["class_name"][i]
            confidence = float(detections.confidence[i])
            all_predictions.append({
                "class_name": class_name,
                "confidence": confidence
            })

        best_idx = int(np.argmax(detections.confidence))
        best_class = detections.data["class_name"][best_idx]
        best_conf = float(detections.confidence[best_idx])

        return {
            "prepared_crop": prepared_crop,
            "prediction": {
                "class_name": best_class,
                "confidence": best_conf
            },
            "all_predictions": all_predictions,
            "raw_result": result,
            "detections": detections
        }