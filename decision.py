class DecisionEngine:
    def __init__(
        self,
        det_threshold,
        type_threshold,
        min_bbox_area,
        max_bbox_area,
        min_aspect_ratio,
        max_aspect_ratio,
        crop_padding,
        min_type_margin=0.10
    ):
        self.det_threshold = det_threshold
        self.type_threshold = type_threshold
        self.min_bbox_area = min_bbox_area
        self.max_bbox_area = max_bbox_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.crop_padding = crop_padding
        self.min_type_margin = min_type_margin

    def evaluate_detection(self, detection, image_shape):
        if detection is None:
            return {
                "valid": False,
                "reason": "No hazelnut detected"
            }

        det_conf = detection["confidence"]
        if det_conf < self.det_threshold:
            return {
                "valid": False,
                "reason": f"Detector confidence too low: {det_conf:.4f}"
            }

        x1, y1, x2, y2 = detection["bbox"]

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_area = bbox_w * bbox_h
        aspect_ratio = bbox_w / bbox_h if bbox_h != 0 else 0

        if bbox_area < self.min_bbox_area or bbox_area > self.max_bbox_area:
            return {
                "valid": False,
                "reason": f"BBox area out of range: {bbox_area}",
                "bbox_area": bbox_area,
                "aspect_ratio": aspect_ratio
            }

        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return {
                "valid": False,
                "reason": f"Aspect ratio out of range: {aspect_ratio:.3f}",
                "bbox_area": bbox_area,
                "aspect_ratio": aspect_ratio
            }

        h, w = image_shape[:2]

        x1 = max(0, x1 - self.crop_padding)
        y1 = max(0, y1 - self.crop_padding)
        x2 = min(w, x2 + self.crop_padding)
        y2 = min(h, y2 + self.crop_padding)

        return {
            "valid": True,
            "bbox": (x1, y1, x2, y2),
            "detector_confidence": det_conf,
            "bbox_area": bbox_area,
            "aspect_ratio": aspect_ratio
        }

    def evaluate_classification(self, classification_result):
        prediction = classification_result.get("prediction")
        all_predictions = classification_result.get("all_predictions", [])

        if prediction is None:
            return {
                "final_decision": "invalid",
                "reason": "Type model returned no prediction"
            }

        best_class = prediction["class_name"]
        best_conf = prediction["confidence"]

        if best_conf < self.type_threshold:
            return {
                "final_decision": "invalid",
                "reason": f"Type confidence too low: {best_conf:.4f}",
                "type_confidence": best_conf
            }

        # Top1 - Top2 margin check
        if len(all_predictions) >= 2:
            sorted_preds = sorted(all_predictions, key=lambda x: x["confidence"], reverse=True)
            top1 = sorted_preds[0]
            top2 = sorted_preds[1]
            margin = top1["confidence"] - top2["confidence"]

            if margin < self.min_type_margin:
                return {
                    "final_decision": "invalid",
                    "reason": f"Type margin too low: {margin:.4f}",
                    "type_confidence": best_conf
                }

        return {
            "final_decision": best_class,
            "reason": "Accepted",
            "type_confidence": best_conf
        }