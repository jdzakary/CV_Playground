from __future__ import annotations

import cv2
import numpy as np

from app.streaming.processing import Operation


class ChessGrid(Operation):
    name = 'Chess Grid Detection'
    description = 'Detect corners and interpolate the chess grid'

    def __init__(self):
        super().__init__()

    def execute(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        norm = np.empty(dst.shape, dtype=np.float32)
        cv2.normalize(dst, norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        scaled = cv2.convertScaleAbs(norm)
        rows, cols = np.nonzero(scaled > 200)
        for i in range(rows.shape[0]):
            r = int(rows[i])
            c = int(cols[i])
            cv2.circle(frame, (c, r), 1, (0, 0, 255), 1)
        return frame
