# i2i Academy - Applied Image Processing (i2i-Academy-AppliedImageProcessing-1)

This project implements an Automated License Plate Recognition (ALPR) module using OpenCV and EasyOCR. It detects, segments, and extracts the text from a vehicle's license plate in a real-world photograph.

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
1. Load the full-resolution `car.jpg` (a real photograph from Wikimedia Commons).
2. Perform image preprocessing (grayscale conversion, bilateral filtering, Canny edge detection).
3. Use EasyOCR's CRAFT-based text detector to locate text regions in the image.
4. Crop the license plate area, upscale it 2x, and apply CLAHE contrast enhancement.
5. Run OCR on the enhanced plate crop and combine detected text fragments.
6. Print the recognized license plate text to the terminal (e.g., `34EPE495`).
7. Save intermediate images: `gray.jpg`, `edged.jpg`, `cropped_plate.jpg`, `detected_plate.jpg`.
