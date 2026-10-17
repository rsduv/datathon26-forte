"""Имитация другого ракурса камеры: зеркало + поворот/перспектива + яркость → новые .mov.

py -3.11 service/tools/augment.py <папка .mov> <папка вывода> [flip|warp|both]
"""
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
src, dst, mode = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "both")
os.makedirs(dst, exist_ok=True)


def warp_matrix(w, h, k=0.08):
    s = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = np.float32([[w * k, h * k * 0.5], [w * (1 - k * 0.3), 0], [w, h * (1 - k * 0.4)], [w * k * 0.5, h]])
    return cv2.getPerspectiveTransform(s, d)


for name in sorted(os.listdir(src)):
    if not name.lower().endswith(".mov"):
        continue
    key = os.path.splitext(name)[0].replace("Х", "X")
    out = os.path.join(dst, f"{key}_{mode}.mp4")
    if os.path.exists(out):
        continue
    cap = cv2.VideoCapture(os.path.join(src, name))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wr, M = None, None
    while True:
        ok, im = cap.read()
        if not ok:
            break
        h, w = im.shape[:2]
        if mode in ("flip", "both"):
            im = cv2.flip(im, 1)
        if mode in ("warp", "both"):
            M = M if M is not None else warp_matrix(w, h)
            im = cv2.warpPerspective(im, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            im = cv2.convertScaleAbs(im, alpha=0.9, beta=-10)
        if wr is None:
            wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        wr.write(im)
    cap.release()
    if wr:
        wr.release()
    print(key, "ok", flush=True)
