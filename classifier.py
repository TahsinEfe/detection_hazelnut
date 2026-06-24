import cv2
try:
    import requests
except ImportError:
    requests = None
from config import ROBOFLOW_API_KEY
from utils import prepare_for_type_model

class HazelnutClassifier:
    def __init__(self, model_id: str, target_size: int, padding_ratio: float):
        self.url = f"https://detect.roboflow.com/{model_id}"
        self.api_key = ROBOFLOW_API_KEY
        self.target_size = target_size
        self.padding_ratio = padding_ratio

    def classify(self, crop):
        prepared_crop = prepare_for_type_model(
            crop,
            target_size=self.target_size,
            padding_ratio=self.padding_ratio
        )

        ok, buffer = cv2.imencode(".jpg", prepared_crop)
        if not ok:
            return {
                "prepared_crop": prepared_crop,
                "prediction": None,
                "all_predictions": [],
                "raw_result": None
            }

        response = requests.post(
            self.url,
            params={"api_key": self.api_key},
            files={"file": ("crop.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        print("RAW CLASSIFIER RESULT:", result)

        preds = result.get("predictions", [])
        if not preds:
            return {
                "prepared_crop": prepared_crop,
                "prediction": None,
                "all_predictions": [],
                "raw_result": result
            }

        all_predictions = [
            {
                "class_name": p.get("class"),
                "confidence": float(p.get("confidence", 0))
            }
            for p in preds
        ]

        best = max(all_predictions, key=lambda p: p["confidence"])

        return {
            "prepared_crop": prepared_crop,
            "prediction": {
                "class_name": best["class_name"],
                "confidence": best["confidence"]
            },
            "all_predictions": all_predictions,
            "raw_result": result
        }
