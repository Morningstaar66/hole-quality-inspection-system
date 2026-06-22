#!/usr/bin/env python3
"""
================================================================================
INDUSTRIAL HOLE QUALITY INSPECTION SYSTEM
================================================================================
Complete pipeline for:
1. Detecting holes in factory product images
2. Classifying each hole for burrs (good/bad)
3. Making automated shipping/reject decision

Author: Manufacturing QC Pipeline
Version: 1.0
================================================================================
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# ================================================================================
# CONFIGURATION - CUSTOMIZE THESE VALUES
# ================================================================================

class Config:
    """All configurable parameters in one place."""

    # MODEL PATHS
    DETECTION_MODEL = "/Users/oluwasanmiafolabi/dataset1/runs/segment/hole_seg_robust_final/weights/best.pt"
    CLASSIFIER_MODEL = "/Users/oluwasanmiafolabi/runs/classify/hole_burr_classifier_v3/weights/best.pt"

    # IMAGE PATH
    IMAGE_PATH = "/Users/oluwasanmiafolabi/Wholeproduct/IMG_3955.jpeg"

    # DETECTION PARAMETERS
    DETECTION_CONFIDENCE = 0.70  # Min confidence to detect a hole (0.0-1.0)
    HOLE_PADDING = 15  # Pixels to add around hole for classification context

    # CLASSIFICATION PARAMETERS
    CLASSIFIER_CONFIDENCE = 0.0  # Min confidence threshold (0.0 = use all predictions)
    GOOD_HOLE_KEYWORDS = ["good", "no_burr", "clean", "pass"]  # Keywords for "good" classification
    BAD_HOLE_KEYWORDS = ["bad", "burr", "defect", "fail"]  # Keywords for "bad" classification

    # QUALITY THRESHOLDS FOR SHIPPING DECISION
    MIN_HOLES_REQUIRED = 159  # Minimum number of holes must be detected
    GOOD_HOLES_PERCENTAGE = 0.95  # 95% of holes must be classified as good

    # OUTPUT
    SAVE_RESULTS = True  # Save annotated image and report
    OUTPUT_DIR = "/Users/oluwasanmiafolabi/Wholeproduct/Results"


# ================================================================================
# UTILITY FUNCTIONS
#https://github.com/Morningstaar66/hole-quality-inspection-system
#0638840753
# ================================================================================

def setup_logging():
    """Create output directory and setup logging."""
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_step(step_num, description):
    """Print formatted step indicator."""
    print(f"\n[STEP {step_num}] {description}")
    print("-" * 80)


def validate_paths():
    """Verify all required files exist."""
    print_step(1, "Validating File Paths")

    paths_to_check = {
        "Detection Model": Config.DETECTION_MODEL,
        "Classifier Model": Config.CLASSIFIER_MODEL,
        "Image File": Config.IMAGE_PATH,
    }

    all_valid = True
    for name, path in paths_to_check.items():
        exists = Path(path).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_valid = False

    if not all_valid:
        raise FileNotFoundError("One or more required files not found. Check paths in Config class.")

    print("\n✓ All paths validated successfully!")


def load_models():
    """Load detection and classification models."""
    print_step(2, "Loading AI Models")

    try:
        print(f"Loading detection model...")
        detector = YOLO(Config.DETECTION_MODEL)
        print(f"  ✓ Detection model loaded")

        print(f"Loading classifier model...")
        classifier = YOLO(Config.CLASSIFIER_MODEL)
        print(f"  ✓ Classifier model loaded")

        return detector, classifier

    except Exception as e:
        print(f"✗ Error loading models: {e}")
        raise


def run_detection(detector, image_path):
    """Run hole detection on image."""
    print_step(3, "Detecting Holes in Product")

    try:
        results = detector.predict(
            source=image_path,
            conf=Config.DETECTION_CONFIDENCE,
            verbose=False
        )

        detections = results[0].boxes
        hole_count = len(detections)

        print(f"Detection complete!")
        print(f"  • Holes detected: {hole_count}")
        print(f"  • Confidence threshold: {Config.DETECTION_CONFIDENCE:.0%}")

        return results, detections

    except Exception as e:
        print(f"✗ Error during detection: {e}")
        raise


def extract_hole_regions(image_path, detections):
    """Extract cropped image regions for each detected hole."""
    print_step(4, "Extracting Hole Regions")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image_height, image_width = image.shape[:2]
    hole_regions = []

    for hole_id, box in enumerate(detections, start=1):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Add padding for better classification context
        x1_padded = max(0, x1 - Config.HOLE_PADDING)
        y1_padded = max(0, y1 - Config.HOLE_PADDING)
        x2_padded = min(image_width, x2 + Config.HOLE_PADDING)
        y2_padded = min(image_height, y2 + Config.HOLE_PADDING)

        crop = image[y1_padded:y2_padded, x1_padded:x2_padded]

        if crop.size == 0:
            print(f"  ⚠ Warning: Hole #{hole_id} region is empty, skipping")
            continue

        hole_regions.append({
            'hole_id': hole_id,
            'crop': crop,
            'bbox': (x1, y1, x2, y2),
            'bbox_padded': (x1_padded, y1_padded, x2_padded, y2_padded),
            'detection_confidence': box.conf.item(),
        })

    print(f"✓ Extracted {len(hole_regions)} hole region(s)")
    return hole_regions


def classify_holes(classifier, hole_regions):
    """Classify each hole region for burrs/defects."""
    print_step(5, "Classifying Holes (Burr Detection)")

    classifications = []

    for hole_data in hole_regions:
        hole_id = hole_data['hole_id']
        crop = hole_data['crop']

        try:
            # Run classifier on cropped region
            results = classifier.predict(
                source=crop,
                conf=Config.CLASSIFIER_CONFIDENCE,
                verbose=False
            )

            # Extract prediction
            probs = results[0].probs
            top_class_id = probs.top1
            top_confidence = probs.top1conf.item()
            class_name = results[0].names[top_class_id]

            # Determine if hole is good or bad
            is_good = classify_quality(class_name)

            classifications.append({
                'hole_id': hole_id,
                'class_name': class_name,
                'classification_confidence': top_confidence,
                'detection_confidence': hole_data['detection_confidence'],
                'is_good': is_good,
                'quality_status': "GOOD" if is_good else "BAD",
            })

            print(f"  Hole #{hole_id}: {class_name.upper()} (conf: {top_confidence:.2%})")

        except Exception as e:
            print(f"  ✗ Error classifying hole #{hole_id}: {e}")
            continue

    print(f"\n✓ Classification complete for {len(classifications)} hole(s)")
    return classifications


def classify_quality(class_name):
    """Determine if classification indicates good or bad hole."""
    class_lower = class_name.lower().strip()

    # Check against good keywords
    for keyword in Config.GOOD_HOLE_KEYWORDS:
        if keyword in class_lower:
            return True

    # Check against bad keywords
    for keyword in Config.BAD_HOLE_KEYWORDS:
        if keyword in class_lower:
            return False

    # Default: if it doesn't match any keyword, consider it as good (optimistic)
    return True


def analyze_quality(classifications):
    """Analyze overall quality metrics."""
    print_step(6, "Analyzing Overall Quality")

    total_holes = len(classifications)
    good_holes = sum(1 for c in classifications if c['is_good'])
    bad_holes = total_holes - good_holes

    quality_percentage = (good_holes / total_holes * 100) if total_holes > 0 else 0
    required_percentage = Config.GOOD_HOLES_PERCENTAGE * 100

    print(f"  • Total holes detected: {total_holes}")
    print(f"  • Good holes (no burrs): {good_holes}")
    print(f"  • Bad holes (burrs detected): {bad_holes}")
    print(f"  • Quality rate: {quality_percentage:.1f}%")
    print(f"  • Required quality rate: {required_percentage:.0f}%")

    metrics = {
        'total_holes': total_holes,
        'good_holes': good_holes,
        'bad_holes': bad_holes,
        'quality_percentage': quality_percentage,
        'required_percentage': required_percentage,
    }

    return metrics


def make_shipping_decision(classifications, metrics):
    """Determine if product is ready for shipping."""
    print_step(7, "Making Shipping Decision")

    total_holes = metrics['total_holes']
    good_holes = metrics['good_holes']
    quality_percentage = metrics['quality_percentage']
    required_percentage = metrics['required_percentage']

    # Decision logic
    if total_holes < Config.MIN_HOLES_REQUIRED:
        decision = "REJECT"
        reason = f"Only {total_holes} hole(s) detected. Minimum required: {Config.MIN_HOLES_REQUIRED}"

    elif quality_percentage >= required_percentage:
        decision = "APPROVE"
        reason = f"{good_holes}/{total_holes} holes are good ({quality_percentage:.1f}%) - Meets {required_percentage:.0f}% threshold"

    else:
        decision = "REJECT"
        reason = f"Only {good_holes}/{total_holes} holes are good ({quality_percentage:.1f}%) - Below {required_percentage:.0f}% threshold"

    return decision, reason


def print_detailed_report(classifications, metrics, decision, reason):
    """Print detailed quality report."""
    print_section("DETAILED HOLE CLASSIFICATION REPORT")

    print("\nHole-by-Hole Analysis:")
    print("-" * 80)

    for clf in classifications:
        status_symbol = "✓" if clf['is_good'] else "✗"
        print(f"\n{status_symbol} HOLE #{clf['hole_id']}")
        print(f"   Classification: {clf['class_name'].upper()}")
        print(f"   Quality Status: {clf['quality_status']}")
        print(f"   Classification Confidence: {clf['classification_confidence']:.2%}")
        print(f"   Detection Confidence: {clf['detection_confidence']:.2%}")

    print("\n" + "-" * 80)
    print("\nQuality Summary:")
    print("-" * 80)
    print(f"  • Total Holes Detected: {metrics['total_holes']}")
    print(f"  • Good Holes (No Burrs): {metrics['good_holes']}")
    print(f"  • Bad Holes (Burrs Found): {metrics['bad_holes']}")
    print(f"  • Quality Rate: {metrics['quality_percentage']:.1f}%")
    print(f"  • Required Rate: {metrics['required_percentage']:.0f}%")
    print(
        f"  • Meets Requirement: {'YES ✓' if metrics['quality_percentage'] >= metrics['required_percentage'] else 'NO ✗'}")

    print("\n" + "-" * 80)
    print("Decision Rationale:")
    print("-" * 80)
    print(f"  {reason}")


def print_shipping_decision(decision, reason):
    """Print final shipping decision with emphasis."""
    print_section("FINAL SHIPPING DECISION")

    if decision == "APPROVE":
        print("\n")
        print("  ✓✓✓  PRODUCT APPROVED FOR SHIPPING  ✓✓✓")
        print("\n")
        print(f"  Reason: {reason}")
        print("\n")
    else:
        print("\n")
        print("  ✗✗✗  PRODUCT REJECTED - QC FAILED  ✗✗✗")
        print("\n")
        print(f"  Reason: {reason}")
        print("\n")


def save_report(classifications, metrics, decision, reason, output_dir):
    """Save inspection report to file."""
    if not Config.SAVE_RESULTS:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"inspection_report_{timestamp}.txt"

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HOLE QUALITY INSPECTION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Image: {Config.IMAGE_PATH}\n")
        f.write(f"Detection Model: {Config.DETECTION_MODEL}\n")
        f.write(f"Classifier Model: {Config.CLASSIFIER_MODEL}\n\n")

        f.write("-" * 80 + "\n")
        f.write("HOLE CLASSIFICATIONS\n")
        f.write("-" * 80 + "\n")

        for clf in classifications:
            f.write(f"\nHole #{clf['hole_id']}:\n")
            f.write(f"  Classification: {clf['class_name']}\n")
            f.write(f"  Quality Status: {clf['quality_status']}\n")
            f.write(f"  Classification Confidence: {clf['classification_confidence']:.2%}\n")
            f.write(f"  Detection Confidence: {clf['detection_confidence']:.2%}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("SUMMARY METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Holes: {metrics['total_holes']}\n")
        f.write(f"Good Holes: {metrics['good_holes']}\n")
        f.write(f"Bad Holes: {metrics['bad_holes']}\n")
        f.write(f"Quality Rate: {metrics['quality_percentage']:.1f}%\n")
        f.write(f"Required Rate: {metrics['required_percentage']:.0f}%\n\n")

        f.write("-" * 80 + "\n")
        f.write("SHIPPING DECISION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Decision: {decision}\n")
        f.write(f"Reason: {reason}\n")

    print(f"\n✓ Report saved to: {report_file}")


# ================================================================================
# MAIN EXECUTION
#https://github.com/Morningstaar66/hole-quality-inspection-system
#0638840753
# ================================================================================

def main():
    """Main execution pipeline."""

    try:
        print_section("INDUSTRIAL HOLE QUALITY INSPECTION SYSTEM")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Setup
        output_dir = setup_logging()

        # Step 1: Validate paths
        validate_paths()

        # Step 2: Load models
        detector, classifier = load_models()

        # Step 3: Detect holes
        detection_results, detections = run_detection(detector, Config.IMAGE_PATH)

        if len(detections) == 0:
            print_section("NO HOLES DETECTED")
            print("\n✗ Product REJECTED - No holes detected in image\n")
            return "REJECT"

        # Step 4: Extract hole regions
        hole_regions = extract_hole_regions(Config.IMAGE_PATH, detections)

        if len(hole_regions) == 0:
            print_section("HOLE EXTRACTION FAILED")
            print("\n✗ Product REJECTED - Could not extract hole regions\n")
            return "REJECT"

        # Step 5: Classify holes
        classifications = classify_holes(classifier, hole_regions)

        if len(classifications) == 0:
            print_section("CLASSIFICATION FAILED")
            print("\n✗ Product REJECTED - Could not classify holes\n")
            return "REJECT"

        # Step 6: Analyze quality
        metrics = analyze_quality(classifications)

        # Step 7: Make decision
        decision, reason = make_shipping_decision(classifications, metrics)

        # Print reports
        print_detailed_report(classifications, metrics, decision, reason)
        print_shipping_decision(decision, reason)

        # Save report
        save_report(classifications, metrics, decision, reason, output_dir)

        print_section("INSPECTION COMPLETE")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        return decision

    except Exception as e:
        print_section("CRITICAL ERROR")
        print(f"\n✗ {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        return "ERROR"


# ================================================================================
# ENTRY POINT
#https://github.com/Morningstaar66/hole-quality-inspection-system
#0638840753
# ================================================================================

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result == "APPROVE" else 1)