import cv2
import numpy as np
import os
import json
from datetime import datetime

from preprocessing.image_preprocessor import ImagePreprocessor
from segmentation.segmentation_engine import SegmentationEngine
from features.feature_extractor import FeatureExtractor
from classification.classifier import WeldClassifier
from backend.database import WeldDatabase
from utils.visualizer import ResultVisualizer


class WeldingDefectPipeline:
    """
    Full machine vision pipeline for welding defect detection.

    Stages:
        1. Preprocessing  – grayscale, Gaussian filter, histogram equalization
        2. Segmentation   – adaptive threshold, Canny edge, morphological closing
        3. Feature Extraction – GLCM, LBP, shape/geometric features
        4. Classification – binary (Good/Bad) → multi-class defect type
        5. Storage & Analytics
    """

    def __init__(self, model_dir: str = "models", db_path: str = "weld_results.db"):
        self.preprocessor = ImagePreprocessor()
        self.segmenter    = SegmentationEngine()
        self.extractor    = FeatureExtractor()
        self.classifier   = WeldClassifier(model_dir=model_dir)
        self.database     = WeldDatabase(db_path=db_path)
        self.visualizer   = ResultVisualizer()

    # ------------------------------------------------------------------
    def run(self, image_path: str, save_visuals: bool = True, verbose: bool = True) -> dict:
        """
        Execute the full pipeline on one welding image.

        Returns:
            result dict with keys: status, defect_type, confidence, features, image_path
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not read image: {image_path}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Welding Defect Detection Pipeline")
            print(f"  Image : {os.path.basename(image_path)}")
            print(f"{'='*60}")

        # --- Stage 1: Preprocessing ---
        if verbose:
            print("\n[1/5] Preprocessing...")
        preprocessed = self.preprocessor.process(original)

        # --- Stage 2: Segmentation ---
        if verbose:
            print("[2/5] Segmentation...")
        segmented, contours = self.segmenter.segment(preprocessed["equalized"])

        # --- Stage 3: Feature Extraction ---
        if verbose:
            print("[3/5] Extracting features...")
        features = self.extractor.extract(
            gray_image  = preprocessed["gray"],
            binary_mask = segmented["binary_mask"],
            edge_map    = segmented["edges"],
            contours    = contours,
        )

        # --- Stage 4: Classification ---
        if verbose:
            print("[4/5] Classifying...")
        classification = self.classifier.predict(features)

        # --- Stage 5: Storage ---
        if verbose:
            print("[5/5] Saving to database...")
        record_id = self.database.save(
            image_path  = image_path,
            status      = classification["status"],
            defect_type = classification["defect_type"],
            features    = features,
        )

        result = {
            "record_id"   : record_id,
            "image_path"  : image_path,
            "status"      : classification["status"],
            "defect_type" : classification["defect_type"],
            "confidence"  : classification["confidence"],
            "scores"      : classification["scores"],
            "features"    : features,
            "timestamp"   : datetime.now().isoformat(),
        }

        if save_visuals:
            out_dir = "results"
            os.makedirs(out_dir, exist_ok=True)
            self.visualizer.save_report(
                original      = original,
                preprocessed  = preprocessed,
                segmented     = segmented,
                features      = features,
                classification = classification,
                output_dir    = out_dir,
                base_name     = os.path.splitext(os.path.basename(image_path))[0],
            )

        if verbose:
            self._print_result(result)

        return result

    # ------------------------------------------------------------------
    def run_batch(self, image_dir: str, extensions=(".jpg", ".jpeg", ".png", ".bmp")) -> list:
        """Run the pipeline on every image in a directory."""
        paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(extensions)
        ]
        if not paths:
            print(f"No images found in {image_dir}")
            return []

        results = []
        for i, path in enumerate(paths, 1):
            print(f"\nProcessing [{i}/{len(paths)}]: {os.path.basename(path)}")
            try:
                results.append(self.run(path, save_visuals=True, verbose=False))
            except Exception as e:
                print(f"  ERROR: {e}")
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _print_result(result: dict):
        status  = result["status"]
        defect  = result["defect_type"]
        conf    = result["confidence"]
        feat    = result["features"]

        print(f"\n{'─'*60}")
        print(f"  RESULT")
        print(f"{'─'*60}")
        print(f"  Status      : {'✓ GOOD' if status == 'Good' else '✗ BAD'}")
        if status == "Bad":
            print(f"  Defect type : {defect}")
        print(f"  Confidence  : {conf:.1f}%")
        print(f"\n  Feature Summary:")
        print(f"    Contrast      = {feat['contrast']:.4f}")
        print(f"    Energy        = {feat['energy']:.4f}")
        print(f"    Homogeneity   = {feat['homogeneity']:.4f}")
        print(f"    LBP score     = {feat['lbp_score']:.4f}")
        print(f"    Circularity   = {feat['circularity']:.4f}")
        print(f"    Edge density  = {feat['edge_density']:.4f}")
        print(f"    Area (px²)   = {feat['area']}")
        print(f"{'─'*60}\n")


# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Welding Defect Detection Pipeline")
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("--no-visuals", action="store_true", help="Skip saving result images")
    parser.add_argument("--train",      action="store_true", help="Train classifiers on a dataset")
    parser.add_argument("--data-dir",   default="dataset",  help="Dataset directory (for training)")
    args = parser.parse_args()

    pipeline = WeldingDefectPipeline()

    if args.train:
        from classification.trainer import ModelTrainer
        trainer = ModelTrainer(pipeline.extractor)
        trainer.train(args.data_dir, model_dir="models")
    elif os.path.isdir(args.input):
        results = pipeline.run_batch(args.input)
        good = sum(1 for r in results if r["status"] == "Good")
        bad  = len(results) - good
        print(f"\nBatch complete: {len(results)} images | Good: {good} | Bad: {bad}")
    else:
        pipeline.run(args.input, save_visuals=not args.no_visuals)