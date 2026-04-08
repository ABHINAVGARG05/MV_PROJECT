import cv2
import numpy as np


class ImagePreprocessor:

    def __init__(
        self,
        gaussian_ksize: tuple = (5, 5),
        gaussian_sigma: float = 0,
        clahe: bool = True,
        clahe_clip: float = 2.0,
        clahe_tile: tuple = (8, 8),
    ):
        self.gaussian_ksize = gaussian_ksize
        self.gaussian_sigma = gaussian_sigma
        self.use_clahe      = clahe

        if clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit     = clahe_clip,
                tileGridSize  = clahe_tile,
            )

    def process(self, image: np.ndarray) -> dict:

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        filtered = cv2.GaussianBlur(gray, self.gaussian_ksize, self.gaussian_sigma)

        if self.use_clahe:
            equalized = self._clahe.apply(filtered)
        else:
            equalized = cv2.equalizeHist(filtered)

        return {
            "original"  : image,
            "gray"      : gray,
            "filtered"  : filtered,
            "equalized" : equalized,
        }

    def resize(self, image: np.ndarray, width: int = 512) -> np.ndarray:
        """Resize while keeping aspect ratio, for consistent feature scales."""
        h, w = image.shape[:2]
        if w == width:
            return image
        scale  = width / w
        height = int(h * scale)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)