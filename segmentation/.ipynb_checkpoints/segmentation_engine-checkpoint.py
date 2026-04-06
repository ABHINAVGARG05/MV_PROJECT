import cv2
import numpy as np

class SegmentationEngine:

    def __init__(self):
        pass

    def segment(self, gray: np.ndarray):

        # STEP 1: Smooth image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # STEP 2: OTSU threshold (better for welding brightness)
        _, binary = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # STEP 3: Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # STEP 4: Combine both
        combined = cv2.bitwise_or(binary, edges)

        # STEP 5: Morphological closing (connect broken parts)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        # STEP 6: Find contours
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Keep only largest contour
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        binary_mask = np.zeros_like(gray)

        if contours:
            cv2.drawContours(binary_mask, [contours[0]], -1, 255, -1)

        # Sobel gradient (for features)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))

        # Overlay visualization
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if contours:
            cv2.drawContours(overlay, [contours[0]], -1, (0, 255, 0), 2)

        return {
            "binary_mask": binary_mask,
            "edges": edges,
            "sobel": sobel,
            "overlay": overlay
        }, contours[:1]

    def isolate_seam(self, gray, contours):
        if not contours:
            return None
        x, y, w, h = cv2.boundingRect(contours[0])
        return gray[y:y+h, x:x+w]