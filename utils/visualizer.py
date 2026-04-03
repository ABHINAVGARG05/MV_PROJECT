import cv2
import numpy as np
import os


class ResultVisualizer:

    PANEL_W = 320
    PANEL_H = 240
    FONT    = cv2.FONT_HERSHEY_SIMPLEX
    GREEN   = (60,  200, 60)
    RED     = (60,  60,  220)
    WHITE   = (255, 255, 255)
    BLACK   = (0,   0,   0)
    GRAY    = (160, 160, 160)

    def save_report(
        self,
        original:       np.ndarray,
        preprocessed:   dict,
        segmented:      dict,
        features:       dict,
        classification: dict,
        output_dir:     str,
        base_name:      str,
    ):
        panels = [
            self._to_bgr_panel(original,                     "1. Original"),
            self._to_bgr_panel(preprocessed["gray"],         "2. Grayscale"),
            self._to_bgr_panel(preprocessed["equalized"],    "3. Equalized (CLAHE)"),
            self._to_bgr_panel(segmented["edges"],           "4. Canny edges"),
            self._to_bgr_panel(segmented["overlay"],         "5. Segmented region"),
            self._result_panel(features, classification),
        ]

        row1 = np.hstack(panels[:3])
        row2 = np.hstack(panels[3:])
        report = np.vstack([row1, row2])

        out_path = os.path.join(output_dir, f"{base_name}_report.jpg")
        cv2.imwrite(out_path, report, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  Saved report: {out_path}")

    def _to_bgr_panel(self, img: np.ndarray, title: str) -> np.ndarray:
        if img.ndim == 2:
            panel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            panel = img.copy()

        panel = cv2.resize(panel, (self.PANEL_W, self.PANEL_H))

        cv2.rectangle(panel, (0, 0), (self.PANEL_W, 22), (30, 30, 30), -1)
        cv2.putText(panel, title, (6, 15), self.FONT, 0.42, self.WHITE, 1, cv2.LINE_AA)

        cv2.rectangle(panel, (0, 0), (self.PANEL_W - 1, self.PANEL_H - 1), (60, 60, 60), 1)
        return panel

    def _result_panel(self, features: dict, classification: dict) -> np.ndarray:
        """Create the result/feature summary panel."""
        panel = np.full((self.PANEL_H, self.PANEL_W, 3), 25, dtype=np.uint8)

        status  = classification["status"]
        defect  = classification["defect_type"]
        conf    = classification["confidence"]
        color   = self.GREEN if status == "Good" else self.RED

        cv2.rectangle(panel, (0, 0), (self.PANEL_W, 22), (30, 30, 30), -1)
        cv2.putText(panel, "6. Classification result",
                    (6, 15), self.FONT, 0.42, self.WHITE, 1, cv2.LINE_AA)

        cv2.putText(panel, f"Status: {status}",
                    (8, 48), self.FONT, 0.60, color, 2, cv2.LINE_AA)
        if status == "Bad":
            cv2.putText(panel, f"Defect: {defect}",
                        (8, 72), self.FONT, 0.50, color, 1, cv2.LINE_AA)
        cv2.putText(panel, f"Confidence: {conf:.1f}%",
                    (8, 92), self.FONT, 0.45, self.GRAY, 1, cv2.LINE_AA)

        cv2.line(panel, (8, 100), (self.PANEL_W - 8, 100), (60, 60, 60), 1)

        feat_lines = [
            f"Contrast    : {features.get('contrast', 0):.4f}",
            f"Energy      : {features.get('energy', 0):.4f}",
            f"Homogeneity : {features.get('homogeneity', 0):.4f}",
            f"LBP score   : {features.get('lbp_score', 0):.4f}",
            f"Circularity : {features.get('circularity', 0):.4f}",
            f"Edge density: {features.get('edge_density', 0):.4f}",
            f"Area (px²)  : {features.get('area', 0)}",
        ]

        y = 116
        for line in feat_lines:
            cv2.putText(panel, line, (8, y), self.FONT, 0.36, self.GRAY, 1, cv2.LINE_AA)
            y += 16

        cv2.rectangle(panel, (0, 0), (self.PANEL_W - 1, self.PANEL_H - 1), (60, 60, 60), 1)
        return panel