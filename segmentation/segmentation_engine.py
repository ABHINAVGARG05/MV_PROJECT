import cv2
import numpy as np

class SegmentationEngine:

    def __init__(
        self,
        thresh_block:   int   = 35,
        thresh_c:       int   = 10,
        canny_low:      int   = 50,
        canny_high:     int   = 150,
        morph_ksize:    int   = 7,
        min_area_ratio: float = 0.005,
    ):
        self.thresh_block    = thresh_block
        self.thresh_c        = thresh_c
        self.canny_low       = canny_low
        self.canny_high      = canny_high
        self.morph_ksize     = morph_ksize
        self.min_area_ratio  = min_area_ratio

    def segment(self, gray: np.ndarray) -> tuple[dict, list]:

        h, w = gray.shape[:2]
        min_area = h * w * self.min_area_ratio

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.thresh_block,
            self.thresh_c,
        )

        kernel  = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_ksize, self.morph_ksize),
        )
        closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened  = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel, iterations=1)

        raw_contours, _ = cv2.findContours(
            opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(
            [c for c in raw_contours if cv2.contourArea(c) >= min_area],
            key=cv2.contourArea,
            reverse=True,
        )

        binary_mask = np.zeros_like(gray)
        if contours:
            cv2.drawContours(binary_mask, contours, -1, 255, cv2.FILLED)

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        sx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))

        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay, contours, -1, (0, 255, 80), 2)

        return {
            "binary_mask" : binary_mask,
            "edges"       : edges,
            "sobel"       : sobel,
            "overlay"     : overlay,
        }, contours

    def isolate_seam(self, gray: np.ndarray, contours: list) -> np.ndarray | None:
        if not contours:
            return None
        x, y, ww, hh = cv2.boundingRect(contours[0])
        return gray[y:y+hh, x:x+ww]