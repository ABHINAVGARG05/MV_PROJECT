import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


class FeatureExtractor:

    def __init__(
        self,
        glcm_distances: list = None,
        glcm_angles:    list = None,
        lbp_radius:     int  = 3,
        lbp_n_points:   int  = 24,
    ):
        self.glcm_distances = glcm_distances or [1, 3]
        self.glcm_angles    = glcm_angles    or [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.lbp_radius     = lbp_radius
        self.lbp_n_points   = lbp_n_points

    def extract(
        self,
        gray_image:  np.ndarray,
        binary_mask: np.ndarray,
        edge_map:    np.ndarray,
        contours:    list,
    ) -> dict:

        features = {}
        features.update(self._glcm_features(gray_image, binary_mask))
        features.update(self._lbp_features(gray_image))
        features.update(self._shape_features(binary_mask, edge_map, contours))
        features.update(self._intensity_stats(gray_image, binary_mask))
        return features

    def to_vector(self, features: dict) -> np.ndarray:
        keys = [
            "contrast", "energy", "homogeneity", "correlation",
            "dissimilarity", "asm",
            "lbp_energy", "lbp_entropy", "lbp_mean", "lbp_std",
            "area", "perimeter", "aspect_ratio", "circularity",
            "edge_density", "extent", "solidity", "equiv_diameter",
            "mean_intensity", "std_intensity", "skewness", "kurtosis",
        ]
        return np.array([features.get(k, 0.0) for k in keys], dtype=np.float32)

    def _glcm_features(self, gray: np.ndarray, mask: np.ndarray) -> dict:

        roi = gray.copy()
        roi[mask == 0] = 0

        levels = 64
        roi_q  = (roi / (256 / levels)).astype(np.uint8)

        glcm = graycomatrix(
            roi_q,
            distances  = self.glcm_distances,
            angles     = self.glcm_angles,
            levels     = levels,
            symmetric  = True,
            normed     = True,
        )

        def _prop(name):
            return float(np.mean(graycoprops(glcm, name)))

        return {
            "contrast"      : _prop("contrast"),
            "energy"        : _prop("energy"),
            "homogeneity"   : _prop("homogeneity"),
            "correlation"   : _prop("correlation"),
            "dissimilarity" : _prop("dissimilarity"),
            "asm"           : _prop("ASM"),
        }

    def _lbp_features(self, gray: np.ndarray) -> dict:

        lbp = local_binary_pattern(
            gray,
            self.lbp_n_points,
            self.lbp_radius,
            method="uniform",
        )

        n_bins = self.lbp_n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        eps = 1e-10
        energy  = float(np.sum(hist ** 2))
        entropy = float(-np.sum(hist * np.log2(hist + eps)))
        mean    = float(np.mean(lbp))
        std     = float(np.std(lbp))

        return {
            "lbp_energy"  : energy,
            "lbp_entropy" : entropy,
            "lbp_mean"    : mean,
            "lbp_std"     : std,
            "lbp_score"   : energy,
        }

    def _shape_features(
        self,
        mask:      np.ndarray,
        edge_map:  np.ndarray,
        contours:  list,
    ) -> dict:

        area           = float(cv2.countNonZero(mask))
        edge_pixels    = float(cv2.countNonZero(edge_map))
        total_pixels   = float(mask.size)
        edge_density   = edge_pixels / total_pixels if total_pixels > 0 else 0.0

        if not contours:
            return {
                "area"           : area,
                "perimeter"      : 0.0,
                "aspect_ratio"   : 0.0,
                "circularity"    : 0.0,
                "edge_density"   : edge_density,
                "extent"         : 0.0,
                "solidity"       : 0.0,
                "equiv_diameter" : 0.0,
            }

        cnt        = contours[0]
        perimeter  = float(cv2.arcLength(cnt, closed=True))
        cnt_area   = float(cv2.contourArea(cnt))

        circularity = (
            (4 * np.pi * cnt_area / (perimeter ** 2))
            if perimeter > 0 else 0.0
        )

        _, _, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0

        bbox_area = float(bw * bh)
        extent    = cnt_area / bbox_area if bbox_area > 0 else 0.0

        hull      = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity  = cnt_area / hull_area if hull_area > 0 else 0.0

        equiv_diameter = float(np.sqrt(4 * cnt_area / np.pi)) if cnt_area > 0 else 0.0

        return {
            "area"           : int(cnt_area),
            "perimeter"      : round(perimeter, 2),
            "aspect_ratio"   : round(aspect_ratio, 4),
            "circularity"    : round(min(circularity, 1.0), 4),
            "edge_density"   : round(edge_density, 4),
            "extent"         : round(extent, 4),
            "solidity"       : round(solidity, 4),
            "equiv_diameter" : round(equiv_diameter, 2),
        }


    def _intensity_stats(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        pixels = gray[mask > 0].astype(np.float64)
        if pixels.size == 0:
            return {"mean_intensity": 0.0, "std_intensity": 0.0,
                    "skewness": 0.0, "kurtosis": 0.0}

        mean = float(np.mean(pixels))
        std  = float(np.std(pixels))

        if std > 0:
            skew = float(np.mean(((pixels - mean) / std) ** 3))
            kurt = float(np.mean(((pixels - mean) / std) ** 4) - 3)
        else:
            skew, kurt = 0.0, 0.0

        return {
            "mean_intensity" : round(mean, 3),
            "std_intensity"  : round(std, 3),
            "skewness"       : round(skew, 4),
            "kurtosis"       : round(kurt, 4),
        }