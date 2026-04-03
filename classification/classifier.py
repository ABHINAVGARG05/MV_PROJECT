import os
import pickle
import numpy as np
from features.feature_extractor import FeatureExtractor

class WeldClassifier:

    DEFECT_CLASSES = ["Crack", "Porosity", "Undercut", "Lack of Fusion"]

    def __init__(self, model_dir: str = "models", model_type: str = "svm"):
        self.model_dir   = model_dir
        self.model_type  = model_type
        self.extractor   = FeatureExtractor()

        self.binary_clf  = None
        self.defect_clf  = None
        self.scaler      = None

        self._load_models()

    def _load_models(self):
        paths = {
            "binary": os.path.join(self.model_dir, "binary_classifier.pkl"),
            "defect": os.path.join(self.model_dir, "defect_classifier.pkl"),
            "scaler": os.path.join(self.model_dir, "scaler.pkl"),
        }
        try:
            if all(os.path.exists(p) for p in paths.values()):
                with open(paths["binary"], "rb") as f:
                    self.binary_clf = pickle.load(f)
                with open(paths["defect"], "rb") as f:
                    self.defect_clf = pickle.load(f)
                with open(paths["scaler"], "rb") as f:
                    self.scaler = pickle.load(f)
                print("  [Classifier] Loaded trained models from", self.model_dir)
            else:
                print("  [Classifier] No saved models found – using rule-based fallback.")
        except Exception as e:
            print(f"  [Classifier] Could not load models: {e}. Using fallback.")

    # ------------------------------------------------------------------
    def predict(self, features: dict) -> dict:
        vec = self.extractor.to_vector(features).reshape(1, -1)

        if self.binary_clf is not None and self.scaler is not None:
            return self._ml_predict(vec)
        else:
            return self._rule_predict(features)

    def _ml_predict(self, vec: np.ndarray) -> dict:
        vec_scaled = self.scaler.transform(vec)

        binary_proba = self._get_proba(self.binary_clf, vec_scaled)
        good_idx = list(self.binary_clf.classes_).index("Good")
        good_prob = binary_proba[0][good_idx]
        is_good   = good_prob >= 0.5

        if is_good:
            conf   = round(good_prob * 100, 1)
            scores = [("Good weld", conf)] + [
                (c, round((100 - conf) / len(self.DEFECT_CLASSES), 1))
                for c in self.DEFECT_CLASSES
            ]
            return {"status": "Good", "defect_type": "None",
                    "confidence": conf, "scores": scores}

        defect_proba = self._get_proba(self.defect_clf, vec_scaled)[0]
        best_idx     = int(np.argmax(defect_proba))
        defect_name  = self.defect_clf.classes_[best_idx]
        conf         = round(float(defect_proba[best_idx]) * 100, 1)

        scores = [("Good weld", round((1 - good_prob) * 100 * 0.05, 1))] + [
            (self.defect_clf.classes_[i], round(float(p) * 100, 1))
            for i, p in enumerate(defect_proba)
        ]
        return {"status": "Bad", "defect_type": defect_name,
                "confidence": conf, "scores": scores}

    @staticmethod
    def _get_proba(clf, vec):
        """Return probability array; fall back to decision function if needed."""
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(vec)
        scores = clf.decision_function(vec)
        if scores.ndim == 1:
            p = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - p, p])
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def _rule_predict(self, features: dict) -> dict:

        contrast    = features.get("contrast", 0)
        energy      = features.get("energy", 0)
        homogeneity = features.get("homogeneity", 0)
        lbp_energy  = features.get("lbp_score", 0)
        circularity = features.get("circularity", 0)
        edge_density= features.get("edge_density", 0)
        solidity    = features.get("solidity", 1)

        defect_score = (
            (contrast    > 0.35) * 1
          + (energy      < 0.25) * 1
          + (homogeneity < 0.50) * 1
          + (lbp_energy  > 0.45) * 1
          + (edge_density> 0.40) * 1
        )

        if defect_score <= 1:
            conf   = min(95, 60 + (1 - defect_score) * 20)
            scores = [("Good weld", round(conf, 1))] + [
                (c, round((100 - conf) / 4, 1)) for c in self.DEFECT_CLASSES
            ]
            return {"status": "Good", "defect_type": "None",
                    "confidence": round(conf, 1), "scores": scores}

        if circularity < 0.35 and edge_density > 0.45:
            defect, conf = "Crack", 82.0
        elif circularity > 0.65 and solidity > 0.75:
            defect, conf = "Porosity", 78.0
        elif edge_density > 0.55 and solidity < 0.70:
            defect, conf = "Undercut", 75.0
        else:
            defect, conf = "Lack of Fusion", 70.0

        others = [c for c in self.DEFECT_CLASSES if c != defect]
        rem    = 100 - conf
        scores = [("Good weld", 2.0)] + [(defect, conf)] + [
            (c, round(rem / len(others), 1)) for c in others
        ]
        return {"status": "Bad", "defect_type": defect,
                "confidence": conf, "scores": scores}