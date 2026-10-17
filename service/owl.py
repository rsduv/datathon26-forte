"""OWLv2 (Google, Apache-2.0): поиск предметов по текстовому описанию — то, чего нет в классах COCO (пачка денег).
Работает локально; веса скачиваются один раз (в Docker — при сборке образа). ~5 с на кадр на CPU,
поэтому вызывается только на нескольких ключевых кадрах.
"""
import os
import threading

import numpy as np
import torch
from PIL import Image

MODEL = os.environ.get("OWL_MODEL", "google/owlv2-base-patch16-ensemble")
_m = _p = None
_lock = threading.Lock()


def _load():
    global _m, _p
    with _lock:
        if _m is None:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            _p = Owlv2Processor.from_pretrained(MODEL)
            _m = Owlv2ForObjectDetection.from_pretrained(MODEL).eval()
            if torch.cuda.is_available():
                _m = _m.cuda()
    return _m, _p


@torch.inference_mode()
def score(img_bgr, queries):
    """Максимальная уверенность по каждому описанию (0..1) на изображении."""
    m, p = _load()
    inp = p(text=[queries], images=Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1])), return_tensors="pt")
    inp = {k: v.to(m.device) for k, v in inp.items()}
    return m(**inp).logits[0].sigmoid().max(0).values.cpu().numpy()


@torch.inference_mode()
def best_box(img_bgr, query):
    """Самая уверенная рамка по описанию: (уверенность, [x1, y1, x2, y2] в долях img)."""
    m, p = _load()
    h, w = img_bgr.shape[:2]
    inp = p(text=[[query]], images=Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1])), return_tensors="pt")
    inp = {k: v.to(m.device) for k, v in inp.items()}
    o = m(**inp)
    i = int(o.logits[0, :, 0].argmax())
    cx, cy, bw, bh = o.pred_boxes[0, i].tolist()
    s = max(h, w)   # OWLv2 дополняет картинку до квадрата справа/снизу
    box = [(cx - bw / 2) * s / w, (cy - bh / 2) * s / h, (cx + bw / 2) * s / w, (cy + bh / 2) * s / h]
    return float(o.logits[0, i, 0].sigmoid()), [min(1.0, max(0.0, v)) for v in box]
