# HOLE QUALITY INSPECTION SYSTEM - COMPLETE GUIDE

## Overview
A production-ready Python system that:
1. **Detects holes** in factory product images using YOLO segmentation
2. **Classifies each hole** for burrs using YOLO classification
3. **Makes automated shipping decisions** based on quality thresholds

---

## Quick Start (5 minutes)

### 1. Locate Your Model Files
```bash
# Find detection model
find ~ -path "*hole_seg*" -name "best.pt"

# Find classifier model  
find ~ -path "*burr*" -name "best.pt"
find ~ -path "*classify*" -name "best.pt"
```

### 2. Update Configuration
Edit the `Config` class at the top of `hole_inspection_system.py`:

```python
class Config:
    # Set your model paths
    DETECTION_MODEL = "/path/to/detection/best.pt"
    CLASSIFIER_MODEL = "/path/to/classifier/best.pt"
    IMAGE_PATH = "/path/to/product/photo.jpg"
```

### 3. Run the System
```bash
# Activate virtual environment (if using one)
source /Users/oluwasammiaiofalabi/PycharmProjects/PythonProject1/.venv/bin/activate

# Run inspection
python3 hole_inspection_system.py
```

---

## Detailed Configuration

### Model Paths
```python
DETECTION_MODEL = "/Users/oluwasammiaiofalabi/dataset1/runs/segment/hole_seg_robust_final/weights/best.pt"
CLASSIFIER_MODEL = "/Users/oluwasammiaiofalabi/path/to/classifier/best.pt"
IMAGE_PATH = "/Users/oluwasammiaiofalabi/photo.jpeg"
```

### Detection Parameters
```python
DETECTION_CONFIDENCE = 0.70
# Range: 0.0-1.0
# Lower = catch smaller holes (may get false positives)
# Higher = only strong detections (may miss small holes)

HOLE_PADDING = 15
# Pixels to add around hole for classification context
# Higher = larger crop region for classifier
```

### Classification Parameters
```python
CLASSIFIER_CONFIDENCE = 0.0
# Min confidence for classifier output (0.0 = use all predictions)

GOOD_HOLE_KEYWORDS = ["good", "no_burr", "clean", "pass"]
BAD_HOLE_KEYWORDS = ["bad", "burr", "defect", "fail"]
# Edit these to match YOUR classifier class names
```

### Quality Thresholds (CRITICAL FOR SHIPPING DECISION)
```python
MIN_HOLES_REQUIRED = 2
# Product must have at least 2 holes to process
# Set to 1 if single-hole products

GOOD_HOLES_PERCENTAGE = 0.85
# 85% of detected holes must be classified as "good"
# Set to 0.90 for stricter quality (90%)
# Set to 0.75 for looser quality (75%)
```

---

## Finding Your Classifier Class Names

Your classifier needs to match hole quality. Find class names:

### Method 1: Quick Test
```python
from ultralytics import YOLO
classifier = YOLO("/path/to/classifier/best.pt")
results = classifier.predict(source="test_image.jpg")
print("Class names:", results[0].names)
```

### Method 2: Check YAML Config
```bash
cat "/path/to/classifier/runs/classify/train/args.yaml"
# Look for "names:" section
```

### Method 3: Check results.csv
```bash
cat "/path/to/classifier/runs/classify/train/results.csv"
# Check the column headers
```

---

## Understanding the Output

### Console Output Example
```
================================================================================
  INDUSTRIAL HOLE QUALITY INSPECTION SYSTEM
================================================================================

[STEP 1] Validating File Paths
✓ Detection Model: /Users/oluwasammiaiofalabi/dataset1/runs/.../best.pt
✓ Classifier Model: /Users/oluwasammiaiofalabi/classifier/best.pt
✓ Image File: /Users/oluwasammiaiofalabi/photo.jpeg

[STEP 2] Loading AI Models
Loading detection model...
  ✓ Detection model loaded
Loading classifier model...
  ✓ Classifier model loaded

[STEP 3] Detecting Holes in Product
Detection complete!
  • Holes detected: 4
  • Confidence threshold: 70%

[STEP 4] Extracting Hole Regions
✓ Extracted 4 hole region(s)

[STEP 5] Classifying Holes (Burr Detection)
  Hole #1: NO_BURR (conf: 98.23%)
  Hole #2: NO_BURR (conf: 97.45%)
  Hole #3: HAS_BURRS (conf: 88.12%)
  Hole #4: NO_BURR (conf: 96.78%)

✓ Classification complete for 4 hole(s)

================================================================================
  DETAILED HOLE CLASSIFICATION REPORT
================================================================================

Hole-by-Hole Analysis:
✓ HOLE #1
   Classification: NO_BURR
   Quality Status: GOOD
   Classification Confidence: 98.23%
   Detection Confidence: 92.15%

✓ HOLE #2
   Classification: NO_BURR
   Quality Status: GOOD
   Classification Confidence: 97.45%
   Detection Confidence: 91.80%

✗ HOLE #3
   Classification: HAS_BURRS
   Quality Status: BAD
   Classification Confidence: 88.12%
   Detection Confidence: 89.50%

✓ HOLE #4
   Classification: NO_BURR
   Quality Status: GOOD
   Classification Confidence: 96.78%
   Detection Confidence: 93.20%

Quality Summary:
  • Total Holes Detected: 4
  • Good Holes (No Burrs): 3
  • Bad Holes (Burrs Found): 1
  • Quality Rate: 75.0%
  • Required Rate: 85%
  • Meets Requirement: NO ✗

================================================================================
  FINAL SHIPPING DECISION
================================================================================

  ✗✗✗  PRODUCT REJECTED - QC FAILED  ✗✗✗

  Reason: Only 3/4 holes are good (75.0%) - Below 85% threshold

================================================================================
```

### Saved Report File
A text report is automatically saved:
```
./quality_inspection_results/inspection_report_20250102_143025.txt
```

---

## Troubleshooting

### Error: "File not found"
**Solution:** Update Config paths with absolute paths (full path from root)
```python
# ✗ Wrong (relative path)
DETECTION_MODEL = "models/best.pt"

# ✓ Correct (full path)
DETECTION_MODEL = "/Users/oluwasammiaiofalabi/models/best.pt"
```

### Error: "ModuleNotFoundError: No module named 'ultralytics'"
**Solution:** Use the correct Python environment
```bash
# Use venv Python, not system Python
/Users/oluwasammiaiofalabi/PycharmProjects/PythonProject1/.venv/bin/python3 hole_inspection_system.py
```

### Error: "No holes detected"
**Solutions:**
- Decrease `DETECTION_CONFIDENCE` (try 0.50)
- Ensure image is well-lit
- Check if hole colors match training data

### Error: "Classification failed"
**Solutions:**
- Check classifier class names match keywords
- Update `GOOD_HOLE_KEYWORDS` and `BAD_HOLE_KEYWORDS`
- Verify classifier model can run: `yolo predict model=classifier.pt source=test.jpg`

### Holes not being classified correctly
**Solution:** Adjust keywords to match your classifier:
```python
# If your classifier uses different names:
GOOD_HOLE_KEYWORDS = ["quality_pass", "acceptable", "within_spec"]
BAD_HOLE_KEYWORDS = ["quality_fail", "reject", "out_of_spec"]
```

---

## Terminal Command (macOS)

### One-liner with virtual environment:
```bash
/Users/oluwasammiaiofalabi/PycharmProjects/PythonProject1/.venv/bin/python3 hole_inspection_system.py
```

### Create alias for easy access:
```bash
# Add to ~/.zshrc or ~/.bash_profile
alias inspect_product='/Users/oluwasammiaiofalabi/PycharmProjects/PythonProject1/.venv/bin/python3 /path/to/hole_inspection_system.py'

# Then run:
inspect_product
```

### Create shell script:
```bash
cat > inspect_product.sh << 'EOF'
#!/bin/bash
source /Users/oluwasammiaiofalabi/PycharmProjects/PythonProject1/.venv/bin/activate
python3 hole_inspection_system.py
EOF

chmod +x inspect_product.sh
./inspect_product.sh
```

---

## Customization Examples

### Stricter Quality Control (95% good holes)
```python
GOOD_HOLES_PERCENTAGE = 0.95
MIN_HOLES_REQUIRED = 3
```

### Looser Quality Control (75% good holes)
```python
GOOD_HOLES_PERCENTAGE = 0.75
```

### Detect only perfect holes
```python
DETECTION_CONFIDENCE = 0.90  # Only high-confidence detections
CLASSIFIER_CONFIDENCE = 0.95  # Only high-confidence classifications
```

### Process multiple images in batch
```bash
# Create simple loop in terminal
for image in /path/to/images/*.jpg; do
  echo "Processing: $image"
  python3 hole_inspection_system.py "$image"
done
```

---

## What Each Step Does

| Step | Purpose | Output |
|------|---------|--------|
| 1 | Validate all files exist | File path verification |
| 2 | Load YOLO detection model | Model initialization |
| 3 | Detect holes in full image | Number of holes found |
| 4 | Extract cropped regions | Individual hole images |
| 5 | Classify each hole | Good/Bad for each hole |
| 6 | Calculate quality metrics | Overall quality percentage |
| 7 | Make shipping decision | APPROVE or REJECT |

---

## Best Practices

### Image Capture
- Ensure consistent lighting
- Capture full product surface
- Avoid shadows and glare
- Use 2-4 MP resolution (sufficient for hole detection)

### Model Quality
- Detection model should be trained on similar product photos
- Classifier should be trained on good vs bad hole examples
- Test on actual factory conditions before deployment

### Quality Thresholds
- Start conservative (90% good holes)
- Adjust based on field feedback
- Log all decisions for audit trail

### Regular Maintenance
- Retrain models monthly with new data
- Update classifier when new defect types appear
- Validate decisions against manual inspection

---

## Support & Debugging

### Enable Verbose Mode (see detailed logs)
```python
# In detection step, change:
results = detector.predict(
    source=image_path,
    conf=Config.DETECTION_CONFIDENCE,
    verbose=True  # ← Change to True
)
```

### Save annotated images
```python
# Add to hole_inspection_system.py after detection:
detection_results[0].save("detection_result.jpg")
```

### Manual model testing
```bash
# Test detection model
yolo detect predict model=/path/to/detection/best.pt source=/path/to/image.jpg

# Test classifier model
yolo classify predict model=/path/to/classifier/best.pt source=/path/to/image.jpg
```

---

## Integration with Factory Systems

### Via REST API (future enhancement)
```python
# Create Flask endpoint to use this system
from flask import Flask, request
app = Flask(__name__)

@app.route('/inspect', methods=['POST'])
def inspect():
    image_path = request.json['image_path']
    result = main()
    return {'decision': result}
```

### Via Database Logging
```python
# Add to save_report function
db.insert({
    'timestamp': datetime.now(),
    'image': Config.IMAGE_PATH,
    'decision': decision,
    'quality_rate': metrics['quality_percentage']
})
```

---

## Questions?

If models don't match expected class names or paths, post:
1. Output of `yolo info` (model details)
2. List of your model paths from `find ~ -name best.pt`
3. Example classification output from your classifier

Ready to deploy? You're set! 🚀
