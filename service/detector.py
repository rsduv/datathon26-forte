"""Детектор объектов (torchvision Faster R-CNN MobileNetV3, BSD-3, веса COCO).

Веса скачиваются один раз в кэш torch (~/.cache/torch); дальше работает офлайн.
"""
import threading

import numpy as np
import torch
import torchvision
from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                          fasterrcnn_mobilenet_v3_large_fpn)

# COCO-классы, которые нам интересны
KEEP = {"person", "laptop", "cup", "bottle", "handbag", "backpack", "suitcase",
        "cell phone", "book", "chair", "dining table", "tv", "bowl", "sandwich",
        "donut", "cake", "banana", "apple"}
MIN_SCORE = 0.4

_model = None
_names = None
_lock = threading.Lock()
torch.set_num_threads(max(1, torch.get_num_threads()))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # на GPU в ~10 раз быстрее


def _load():
    global _model, _names
    with _lock:
        if _model is None:
            w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            m = fasterrcnn_mobilenet_v3_large_fpn(weights=w, box_score_thresh=MIN_SCORE)
            m.eval().to(DEVICE)
            _names = w.meta["categories"]
            _model = m
    return _model, _names


@torch.inference_mode()
def detect(img_bgr):
    """-> list of dict(label, score, box=[x1,y1,x2,y2] в долях кадра)."""
    model, names = _load()
    h, w = img_bgr.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1])).permute(2, 0, 1).float().to(DEVICE) / 255
    out = model([x])[0]
    res = []
    for b, l, s in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist()):
        name = names[l]
        if name in KEEP:
            res.append({"label": name, "score": round(s, 3),
                        "box": [round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)]})
    return res
