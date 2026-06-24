import cv2
import requests
from config import ROBOFLOW_API_KEY

class HazelnutDetector:
    def __init__(self, model_id: str):
        self.url = f"https://detect.roboflow.com/{model_id}"
        self.api_key = ROBOFLOW_API_KEY

    def detect(self, image):
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            return None

        response = requests.post(
            self.url,
            params={"api_key": self.api_key},
            files={"file": ("image.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        preds = result.get("predictions", [])
        if not preds:
            return None

        best = max(preds, key=lambda p: p.get("confidence", 0))
        x, y, w, h = best["x"], best["y"], best["width"], best["height"]

        return {
            "confidence": float(best["confidence"]),
            "bbox": (
                int(x - w / 2),
                int(y - h / 2),
                int(x + w / 2),
                int(y + h / 2)
            ),
            "raw_result": result
        }
