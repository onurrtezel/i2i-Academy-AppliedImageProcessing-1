# i2i Academy - Applied Image Processing (i2i-Academy-AppliedImageProcessing-1)

This project implements an Automated License Plate Recognition (ALPR) module using OpenCV and EasyOCR. It detects, segments, and extracts the text from a vehicle's license plate.

---

## Theoretical Knowledge

1. **Difference between Computer Vision and Image Processing:** Computer Vision focuses on interpreting and understanding image content to make high-level decisions (e.g., identifying objects), whereas classical Image Processing concentrates on transforming and enhancing images (e.g., filtering noise) without understanding their semantics.
2. **Grayscale for Edge Detection:** Edge detection requires converting to grayscale because edges represent transitions in pixel intensity (brightness), and reducing the image to a single channel simplifies mathematical computations while eliminating irrelevant color noise.
3. **What an OCR Engine Does:** An OCR (Optical Character Recognition) engine processes image regions containing textual information, analyzes the character shapes, and translates them into editable, machine-readable text strings.

---

## How to Run

### Prerequisites
- **Python 3.8+**
- **OpenCV**
- **EasyOCR**

### Install Dependencies
```bash
pip install opencv-python easyocr numpy
```

### Run the Script
```bash
python3 main.py
```

### Expected Output
The script will:
1. Load `car.jpg`.
2. Clean the image (grayscale, blur, Canny edge detection).
3. Find and isolate the license plate contour.
4. Perform OCR on the cropped plate area.
5. Print the recognized license plate text to the terminal and display the intermediate processing steps.
