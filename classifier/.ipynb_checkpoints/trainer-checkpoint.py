import os
import pickle
import numpy as np
import cv2

from sklearn.svm          import SVC
from sklearn.ensemble     import RandomForestClassifier
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics      import classification_report, confusion_matrix

from preprocessing.image_preprocessor import ImagePreprocessor
from segmentation.segmentation_engine import SegmentationEngine


LABEL_MAP = {
    "good"           : ("Good", None),
    "crack"          : ("Bad",  "Crack"),
    "porosity"       : ("Bad",  "Porosity"),
    "undercut"       : ("Bad",  "Undercut"),
    "lack_of_fusion" : ("Bad",  "Lack of Fusion"),
}


class ModelTrainer:

    def __init__(self, extractor, model_type: str = "svm"):
        self.extractor    = extractor
        self.model_type   = model_type
        self.preprocessor = ImagePreprocessor()
        self.segmenter    = SegmentationEngine()

    def train(self, data_dir: str, model_dir: str = "models", test_size: float = 0.2):
        """Full train–evaluate–save cycle."""
        print("\n[Trainer] Loading dataset from:", data_dir)
        X, y_binary, y_defect = self._load_dataset(data_dir)

        if len(X) == 0:
            print("[Trainer] No data found. Check dataset structure.")
            return

        print(f"[Trainer] {len(X)} samples loaded.")
        X = np.array(X, dtype=np.float32)

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print("\n[Trainer] Training binary classifier...")
        binary_clf = self._build_model(n_classes=2)
        X_tr, X_te, yb_tr, yb_te = train_test_split(
            X_scaled, y_binary, test_size=test_size, random_state=42, stratify=y_binary
        )
        binary_clf.fit(X_tr, yb_tr)
        yb_pred = binary_clf.predict(X_te)
        print(classification_report(yb_te, yb_pred))
        cv = cross_val_score(binary_clf, X_scaled, y_binary, cv=5, scoring="f1_weighted")
        print(f"  Cross-val F1 (binary): {cv.mean():.3f} ± {cv.std():.3f}")

        bad_mask = np.array(y_binary) == "Bad"
        X_bad    = X_scaled[bad_mask]
        y_bad    = np.array(y_defect)[bad_mask]

        defect_clf = None
        if len(np.unique(y_bad)) > 1:
            print("\n[Trainer] Training defect-type classifier...")
            defect_clf = self._build_model(n_classes=len(np.unique(y_bad)))
            Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
                X_bad, y_bad, test_size=test_size, random_state=42, stratify=y_bad
            )
            defect_clf.fit(Xd_tr, yd_tr)
            yd_pred = defect_clf.predict(Xd_te)
            print(classification_report(yd_te, yd_pred))
        else:
            print("[Trainer] Not enough defect classes to train defect classifier.")

        os.makedirs(model_dir, exist_ok=True)
        self._save(scaler,     os.path.join(model_dir, "scaler.pkl"))
        self._save(binary_clf, os.path.join(model_dir, "binary_classifier.pkl"))
        if defect_clf:
            self._save(defect_clf, os.path.join(model_dir, "defect_classifier.pkl"))
        print(f"\n[Trainer] Models saved to '{model_dir}/'")

    def _load_dataset(self, data_dir: str):
        X, y_binary, y_defect = [], [], []
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

        for folder, (status, defect) in LABEL_MAP.items():
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            files = [
                f for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in IMG_EXTS
            ]
            print(f"  {folder}: {len(files)} images")
            for fname in files:
                fpath = os.path.join(folder_path, fname)
                try:
                    feat = self._extract_from_path(fpath)
                    if feat is not None:
                        X.append(feat)
                        y_binary.append(status)
                        y_defect.append(defect if defect else "Good")
                except Exception as e:
                    print(f"    SKIP {fname}: {e}")

        return X, y_binary, y_defect

    def _extract_from_path(self, path: str):
        img = cv2.imread(path)
        if img is None:
            return None
        pre = self.preprocessor.process(img)
        seg, contours = self.segmenter.segment(pre["equalized"])
        feat = self.extractor.extract(
            gray_image  = pre["gray"],
            binary_mask = seg["binary_mask"],
            edge_map    = seg["edges"],
            contours    = contours,
        )
        return self.extractor.to_vector(feat)

    def _build_model(self, n_classes: int):
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=200, max_depth=None,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
        elif self.model_type == "knn":
            k = max(3, int(np.sqrt(n_classes * 10)))
            return KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
        else: 
            return SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, class_weight="balanced", random_state=42,
            )

    @staticmethod
    def _save(obj, path: str):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved: {path}")