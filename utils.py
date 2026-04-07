import cv2
import numpy as np


def prepare_for_type_model(crop, target_size=400, padding_ratio=0.20):
    h, w = crop.shape[:2]

    canvas_size = max(h, w)
    pad = int(canvas_size * padding_ratio)
    canvas_size = canvas_size + 2 * pad

    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    y_offset = (canvas_size - h) // 2
    x_offset = (canvas_size - w) // 2
    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = crop

    prepared = cv2.resize(canvas, (target_size, target_size))
    return prepared