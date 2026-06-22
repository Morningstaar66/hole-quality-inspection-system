# Industrial Hole Quality Inspection System

A production-ready Python system for automated quality control in manufacturing. Detects holes in factory products, classifies them for defects (burrs), and makes automated shipping/rejection decisions using deep learning models.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output & Results](#output--results)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Support](#support)

---

## Overview

This system automates the quality inspection process for manufactured products with holes by:

1. **Detecting** all holes in a product image using YOLO segmentation
2. **Classifying** each hole as "good" (no burrs) or "bad" (burrs detected) using YOLO classification
3. **Analyzing** quality metrics and making a shipping decision
4. **Reporting** detailed results with confidence scores and audit trails

Perfect for factories needing rapid, consistent, automated QC decisions.

---

## Features

### Core Capabilities
- ✅ **Dual AI Model Integration** - Detection + Classification in one pipeline
- ✅ **Per-Hole Analysis** - Classifies each detected hole individually
- ✅ **Automated Decisions** - YES/NO shipping decisions based on configurable thresholds
- ✅ **Detailed Reporting** - Hole-by-hole breakdown with confidence scores
- ✅ **File Logging** - Saves inspection reports for audit trails
- ✅ **Error Handling** - Graceful failure modes with helpful error messages

### Quality Control
- ✅ Configurable detection sensitivity
- ✅ Customizable quality thresholds
- ✅ Per-hole confidence metrics
- ✅ Quality rate percentages
- ✅ Minimum hole requirement checks

### Production Ready
- ✅ 7-step structured pipeline
- ✅ Comprehensive logging
- ✅ Exit codes for integration
- ✅ Timestamp tracking
- ✅ Console + file output

---

## Requirements

### System Requirements
- macOS 10.13+ OR Linux OR Windows (with WSL)
- Python 3.8+
- 2GB RAM minimum (4GB+ recommended)
- GPU optional but recommended for speed

### Python Dependencies
```
ultralytics>=8.0.0    # YOLO models
opencv-python>=4.5.0 # Image processing
numpy>=1.21.0        # Numerical operations
```

---

## Installation

### 1. Clone or Download
```bash
# Download the script to your project
wget https://yourrepo/hole_inspection_system.py
```

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install ultralytics opencv-python numpy
```

### 4. Verify Installation
```bash
python3 -c "from ultralytics import YOLO; print('✓ YOLO installed')"
python3 -c "import cv2; print('✓ OpenCV installed')"
```

---

## Quick Start

### 1. Locate Your Models
```bash
# Find detection model
find ~ -path "*segment*" -name "best.pt" 2>/dev/null

# Find classifier model
find ~ -path "*classify*" -name "best.pt" 2>/dev/null
```

### 2. Update Configuration
Edit the `Config` class at the top of `hole_inspection_system.py`:

```python
class Config:
    # Set absolute paths to your models
    DETECTION_MODEL = "/full/path/to/detection/best.pt"
    CLASSIFIER_MODEL = "/full/path/to/classifier/best.pt"
    IMAGE_PATH = "/full/path/to/product/photo.jpg"
    
    # Adjust quality threshold (default 85%)
    GOOD_HOLES_PERCENTAGE = 0.85
```

### 3. Run Inspection
```bash
python3 hole_inspection_system.py
```

### Expected Output
```
================================================================================
  INDUSTRIAL HOLE QUALITY INSPECTION SYSTEM
================================================================================

[STEP 1] Validating File Paths
✓ Detection Model: /path/to/detection/best.pt
✓ Classifier Model: /path/to/classifier/best.pt
✓ Image File: /path/to/photo.jpg

[STEP 2] Loading AI Models
✓ Detection model loaded
✓ Classifier model loaded

[STEP 3] Detecting Holes in Product
✓ Detected 4 hole(s)

[STEP 4] Extracting Hole Regions
✓ Extracted 4 hole region(s)

[STEP 5] Classifying Holes (Burr Detection)
  Hole #1: NO_BURR (conf: 98.45%)
  Hole #2: NO_BURR (conf: 97.12%)
  Hole #3: HAS_BURRS (conf: 92.67%)
  Hole #4: NO_BURR (conf: 96.89%)

================================================================================
  FINAL SHIPPING DECISION
================================================================================

  ✗✗✗  PRODUCT REJECTED - QC FAILED  ✗✗✗

  Reason: Only 3/4 holes are good (75.0%) - Below 85% threshold
```

---

## Configuration

### Model Paths
```python
DETECTION_MODEL = "/path/to/detection/best.pt"
# YOLO segmentation model trained to detect holes

CLASSIFIER_MODEL = "/path/to/classifier/best.pt"
# YOLO classification model trained on good vs bad holes

IMAGE_PATH = "/path/to/product/photo.jpg"
# Full product image to inspect
```

### Detection Parameters
```python
DETECTION_CONFIDENCE = 0.70
# Confidence threshold (0.0-1.0)
# Lower = detect smaller holes (more false positives)
# Higher = only strong detections (may miss small holes)

HOLE_PADDING = 15
# Pixels to add around hole for classification context
# Increase if holes are being classified incorrectly
```

### Classification Parameters
```python
CLASSIFIER_CONFIDENCE = 0.0
# Min confidence threshold for classifier (0.0 = use all)

GOOD_HOLE_KEYWORDS = ["good", "no_burr", "clean", "pass"]
# Keywords that indicate a GOOD hole
# Update to match YOUR classifier class names

BAD_HOLE_KEYWORDS = ["bad", "burr", "defect", "fail"]
# Keywords that indicate a BAD hole
```

### Quality Thresholds
```python
MIN_HOLES_REQUIRED = 2
# Minimum holes that must be detected
# If fewer holes found, product rejected automatically

GOOD_HOLES_PERCENTAGE = 0.85
# Required percentage of holes that must be "good"
# 0.85 = 85% of holes must pass
# 0.90 = 90% for stricter QC
# 0.75 = 75% for looser QC
```

### Output Settings
```python
SAVE_RESULTS = True
# Save inspection report to file (always recommended)

OUTPUT_DIR = "./quality_inspection_results"
# Directory where reports are saved
# Created automatically if doesn't exist
```

---

## Usage

### Basic Usage
```bash
python3 hole_inspection_system.py
```

### With Virtual Environment
```bash
source venv/bin/activate
python3 hole_inspection_system.py
```

### On macOS with Specific Python
```bash
/Users/username/path/to/venv/bin/python3 hole_inspection_system.py
```

### Batch Processing Multiple Images
```bash
for image in /path/to/images/*.jpg; do
    echo "Processing: $image"
    cp "$image" /Users/username/photo.jpeg
    python3 hole_inspection_system.py
done
```

### Create Shell Alias
```bash
# Add to ~/.zshrc or ~/.bash_profile
alias inspect_product='/path/to/venv/bin/python3 /path/to/hole_inspection_system.py'

# Then use:
inspect_product
```

### Create Shell Script
```bash
cat > inspect_product.sh << 'EOF'
#!/bin/bash
source /path/to/venv/bin/activate
python3 hole_inspection_system.py "$@"
EOF

chmod +x inspect_product.sh
./inspect_product.sh
```

---

## Output & Results

### Console Output Structure

The system provides 7 sequential steps with clear feedback:

| Step | What Happens | Success Indicator |
|------|--------------|-------------------|
| 1 | Validate model & image paths | ✓ All paths validated |
| 2 | Load YOLO models | ✓ Both models loaded |
| 3 | Run hole detection | ✓ N holes detected |
| 4 | Extract hole regions | ✓ N regions extracted |
| 5 | Classify each hole | ✓ N classified |
| 6 | Analyze quality metrics | ✓ Quality rate: X% |
| 7 | Make shipping decision | ✓/✗ APPROVED/REJECTED |

### Hole-by-Hole Report
```
✓ HOLE #1
   Classification: NO_BURR
   Quality Status: GOOD
   Classification Confidence: 98.45%
   Detection Confidence: 92.15%

✗ HOLE #2
   Classification: HAS_BURRS
   Quality Status: BAD
   Classification Confidence: 88.23%
   Detection Confidence: 91.80%
```

### Quality Summary
```
• Total Holes Detected: 4
• Good Holes (No Burrs): 3
• Bad Holes (Burrs Found): 1
• Quality Rate: 75.0%
• Required Rate: 85%
• Meets Requirement: NO ✗
```

### Final Decision
```
✓✓✓  PRODUCT APPROVED FOR SHIPPING  ✓✓✓
Reason: 4/4 holes are good (100.0%) - Meets 85% threshold

OR

✗✗✗  PRODUCT REJECTED - QC FAILED  ✗✗✗
Reason: Only 3/4 holes are good (75.0%) - Below 85% threshold
```

### Saved Report File
```
./quality_inspection_results/inspection_report_20250102_143025.txt
```

Contains complete inspection details including timestamps, model paths, and all metrics.

---

## Troubleshooting

### Error: "File not found: /path/to/model"
**Problem:** Model path doesn't exist  
**Solution:** Use absolute paths, not relative paths
```python
# ✗ Wrong
DETECTION_MODEL = "models/best.pt"

# ✓ Correct
DETECTION_MODEL = "/Users/username/models/best.pt"
```

### Error: "ModuleNotFoundError: No module named 'ultralytics'"
**Problem:** YOLO not installed or wrong Python version  
**Solution:** Use virtual environment Python
```bash
# Find your venv Python
which python3
source venv/bin/activate
pip install ultralytics
python3 hole_inspection_system.py
```

### Error: "No holes detected"
**Causes & Solutions:**
1. Detection confidence too high → Lower `DETECTION_CONFIDENCE` to 0.50
2. Image quality poor → Ensure good lighting, no shadows
3. Wrong model → Verify detection model was trained on similar holes

```python
# Try lower confidence
DETECTION_CONFIDENCE = 0.50
```

### Error: "Cannot read image: /path/to/image"
**Problem:** Image file corrupted or wrong format  
**Solution:**
```bash
# Verify image exists and is readable
file /path/to/image.jpg
python3 -c "import cv2; img = cv2.imread('/path/to/image.jpg'); print(img.shape if img is not None else 'ERROR')"
```

### Holes classified incorrectly (all GOOD or all BAD)
**Problem:** Classifier keywords don't match your model classes  
**Solution:** Find your actual class names
```python
from ultralytics import YOLO
classifier = YOLO("/path/to/classifier/best.pt")
results = classifier.predict(source="/path/to/test_hole.jpg")
print("Classes:", results[0].names)
```

Then update keywords:
```python
# If your classifier uses: "quality_pass" / "quality_fail"
GOOD_HOLE_KEYWORDS = ["quality_pass", "acceptable"]
BAD_HOLE_KEYWORDS = ["quality_fail", "reject"]
```

### Slow performance
**Solutions:**
1. Use GPU if available (YOLO auto-detects)
2. Reduce image resolution (though less accurate)
3. Increase `DETECTION_CONFIDENCE` to skip marginal detections

---

## Advanced Usage

### Custom Quality Thresholds

**Strict Quality Control (95% good holes):**
```python
GOOD_HOLES_PERCENTAGE = 0.95
MIN_HOLES_REQUIRED = 3
```

**Loose Quality Control (70% good holes):**
```python
GOOD_HOLES_PERCENTAGE = 0.70
```

**Single-Hole Products:**
```python
MIN_HOLES_REQUIRED = 1
GOOD_HOLES_PERCENTAGE = 1.0  # 100% of (at least 1) hole must be good
```

### Batch Processing with Results Collection

```python
import json
from pathlib import Path

results = []
for image_file in Path("/path/to/images").glob("*.jpg"):
    Config.IMAGE_PATH = str(image_file)
    decision = main()
    results.append({
        'image': image_file.name,
        'decision': decision,
        'timestamp': datetime.now().isoformat()
    })

with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Integration with REST API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/inspect', methods=['POST'])
def inspect_product():
    image_path = request.json['image_path']
    Config.IMAGE_PATH = image_path
    decision = main()
    return jsonify({'decision': decision})

if __name__ == '__main__':
    app.run(port=5000)
```

### Database Logging

```python
import sqlite3

def log_to_database(decision, metrics):
    conn = sqlite3.connect('inspections.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO inspections 
        (timestamp, image, decision, quality_rate, total_holes)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        datetime.now(),
        Config.IMAGE_PATH,
        decision,
        metrics['quality_percentage'],
        metrics['total_holes']
    ))
    conn.commit()
    conn.close()
```

---

## Project Structure

```
hole_inspection_system.py
├── Config                          # Configuration class
├── Utility Functions
│   ├── setup_logging()
│   ├── print_section()
│   ├── print_step()
│   └── validate_paths()
├── Model Management
│   └── load_models()
├── Detection Pipeline
│   ├── run_detection()
│   └── extract_hole_regions()
├── Classification Pipeline
│   ├── classify_holes()
│   └── classify_quality()
├── Analysis & Decision
│   ├── analyze_quality()
│   └── make_shipping_decision()
├── Reporting
│   ├── print_detailed_report()
│   ├── print_shipping_decision()
│   └── save_report()
└── Main Execution
    └── main()
```

---

## Performance

### Speed Metrics

| Component | Typical Time | Notes |
|-----------|--------------|-------|
| Model loading | 2-5 seconds | One-time on start |
| Detection | 1-3 seconds | Depends on image size |
| Classification | 0.1-0.5 sec per hole | Linear with hole count |
| **Total (4 holes)** | **5-10 seconds** | Varies by GPU |

### Accuracy Metrics

Expected performance (with well-trained models):
- **Detection Recall:** 95-99% (finds nearly all holes)
- **Detection Precision:** 90-98% (few false positives)
- **Classification Accuracy:** 92-99% (correctly identifies burrs)

---

## Exit Codes

Used for scripting and automation:

```
0   ✓ APPROVE - Product approved for shipping
1   ✗ REJECT - Product rejected
1   ✗ ERROR - Critical error occurred
```

Usage in scripts:
```bash
python3 hole_inspection_system.py
if [ $? -eq 0 ]; then
    echo "APPROVED - Send to shipping"
else
    echo "REJECTED - Route to rework"
fi
```

---

## System Requirements Checklist

- [ ] Python 3.8+
- [ ] Detection model file (best.pt)
- [ ] Classifier model file (best.pt)
- [ ] Product image file (jpg/png)
- [ ] 2GB RAM available
- [ ] Write permission for output directory
- [ ] Virtual environment setup (recommended)

---

## Best Practices

### Image Capture
- Use consistent lighting across images
- Capture full product surface
- Avoid shadows and reflections
- Use 2-4 MP resolution (sufficient for holes)
- Store images in well-organized directories

### Model Management
- Keep detection and classifier models separate
- Version your models (v1, v2, etc.)
- Retrain monthly with new factory data
- Validate on holdout test set before deployment

### Quality Thresholds
- Start conservative (90% good holes)
- Adjust based on field feedback
- Document threshold changes
- Log all decisions for audit trail

### Monitoring
- Review reports daily
- Track rejection rate trends
- Analyze most common defect types
- Schedule model retraining if accuracy drops

---

## Limitations

- Requires well-trained detection and classification models
- Sensitive to image quality and lighting conditions
- Performance degrades if holes vary significantly from training data
- Cannot detect defects outside the classification classes
- Requires sufficient RAM to load both models simultaneously

---

## Contributing

To improve this system:
1. Collect more training data for edge cases
2. Retrain models with expanded datasets
3. Test on diverse product batches
4. Document any modifications
5. Track performance improvements

---

## License

This system is provided as-is for manufacturing quality control use.

---

## Support

### Getting Help

1. **Check model paths** - Most errors come from path issues
2. **Verify classifier classes** - Update keywords to match your classes
3. **Test models independently** - Use YOLO CLI to verify models work
4. **Check image quality** - Ensure product image is clear and well-lit

### Debugging Commands

```bash
# Test detection model
yolo detect predict model=/path/to/detection/best.pt source=/path/to/image.jpg

# Test classifier model
yolo classify predict model=/path/to/classifier/best.pt source=/path/to/hole.jpg

# Check YOLO installation
yolo --version
```

### Common Questions

**Q: Can I use this with multiple camera angles?**  
A: Yes - run the system separately for each angle, average results

**Q: How do I improve accuracy?**  
A: Retrain models with more diverse training data

**Q: Can I process videos instead of images?**  
A: Extract frames first, then process each frame

**Q: What if products have variable hole counts?**  
A: Set `MIN_HOLES_REQUIRED = 1` and adjust thresholds

---

## Version History

- **v1.0** (2025-01-02) - Initial release with detection + classification pipeline

---

## Changelog

### Future Improvements
- [ ] GPU support optimization
- [ ] Batch processing with progress bar
- [ ] Web dashboard for results visualization
- [ ] Email/Slack notifications
- [ ] Database integration for audit trails
- [ ] Video frame processing
- [ ] Multi-camera angle analysis

---

**Last Updated:** January 2, 2025  
**Status:** Production Ready ✓  
**Tested With:** YOLOv8 segmentation & classification models  
**Python Version:** 3.8+  

---

*For questions or issues, refer to the troubleshooting section or check model paths and classifier class names.*
