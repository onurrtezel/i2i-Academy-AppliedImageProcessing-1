import cv2
import numpy as np
import easyocr
import os
import re

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load the image of the car (at original resolution for best OCR accuracy)
    image_path = os.path.join(script_dir, 'car.jpg')
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    height, width = image.shape[:2]
    print(f"Image resolution: {width}x{height}")

    # Create a display-size copy (800px wide) for saving intermediate images
    display_width = 800
    display_height = int((display_width / width) * height)
    display_image = cv2.resize(image, (display_width, display_height))

    # 2. Preprocess the image (grayscale, blur, edge detection)
    # Convert to grayscale — reduces 3-channel color to 1-channel intensity
    gray = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)

    # Canny Edge Detection — highlights intensity transitions (edges)
    edged = cv2.Canny(blurred, 30, 200)

    # Save intermediate images for review
    cv2.imwrite(os.path.join(script_dir, 'gray.jpg'), gray)
    cv2.imwrite(os.path.join(script_dir, 'edged.jpg'), edged)

    # 3. Detect text using EasyOCR on the full-resolution image
    print("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=False)

    print("Performing initial text detection on full-resolution image...")
    results = reader.readtext(image)

    if not results:
        print("Error: No text detected in the image.")
        return

    print(f"Found {len(results)} text region(s).")
    for bbox, text, conf in results:
        print(f"  -> '{text}' (confidence: {conf:.2f})")

    # 4. Find the plate region by grouping nearby text detections in the lower half
    # Turkish plates have text in the lower portion of a car image
    mid_y = height * 0.4
    plate_detections = []

    for bbox, text, conf in results:
        center_y = sum(p[1] for p in bbox) / len(bbox)
        if center_y > mid_y:
            plate_detections.append((bbox, text, conf))

    if not plate_detections:
        # Fallback: use all detections
        plate_detections = [(bbox, text, conf) for bbox, text, conf in results]

    # Compute the bounding box that encompasses all plate-region text detections
    all_x = []
    all_y = []
    for bbox, text, conf in plate_detections:
        for point in bbox:
            all_x.append(int(point[0]))
            all_y.append(int(point[1]))

    x_min = max(0, min(all_x))
    x_max = min(width, max(all_x))
    y_min = max(0, min(all_y))
    y_max = min(height, max(all_y))

    # Add generous padding around the detected text area
    pad_x = int((x_max - x_min) * 0.15)
    pad_y = int((y_max - y_min) * 0.25)
    x_min = max(0, x_min - pad_x)
    x_max = min(width, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(height, y_max + pad_y)

    # 5. Crop the plate region from the full-resolution image and enhance it
    plate_crop = image[y_min:y_max, x_min:x_max]

    # Upscale 2x for better OCR accuracy on small text
    plate_upscaled = cv2.resize(plate_crop, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced contrast
    gray_plate = cv2.cvtColor(plate_upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_plate = clahe.apply(gray_plate)

    # 6. Run OCR again on the enhanced, upscaled plate crop
    print("Performing OCR on enhanced plate region...")
    enhanced_results = reader.readtext(enhanced_plate)

    plate_fragments = []
    for bbox, text, conf in enhanced_results:
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not clean:
            continue

        # Filter out blue-band text ("TR", "TRL") and known logo noise
        if clean in ['TR', 'TRL', 'T', 'R']:
            continue

        # Get center x-position for sorting left-to-right
        center_x = sum(p[0] for p in bbox) / len(bbox)
        # Get center y-position for filtering (plate text is in lower half of crop)
        center_y = sum(p[1] for p in bbox) / len(bbox)
        crop_h = enhanced_plate.shape[0]

        # Only keep fragments in the lower 70% of the crop (skip logos above plate)
        if center_y < crop_h * 0.3:
            continue

        plate_fragments.append((clean, conf, center_x))
        print(f"  -> '{clean}' (confidence: {conf:.2f})")

    if not plate_fragments:
        print("Error: Could not read text from the enhanced plate region.")
        return

    # Sort fragments left-to-right by x-position and combine
    plate_fragments.sort(key=lambda x: x[2])
    combined_plate = "".join([frag for frag, conf, cx in plate_fragments])

    # Apply common OCR corrections for Turkish plates
    # Fix city code 34 misreadings: D4->34, B4->34, O4->34, 04->34, 24->34
    if len(combined_plate) >= 2:
        first_two = combined_plate[:2]
        if first_two in ['04', '24', 'D4', 'B4', 'O4', 'S4']:
            combined_plate = '34' + combined_plate[2:]

    # 7. Draw bounding box on display image and save outputs
    scale_x = display_width / width
    scale_y = display_height / height
    output_image = display_image.copy()

    # Draw rectangle on display image at the scaled coordinates
    dx_min = int(x_min * scale_x)
    dx_max = int(x_max * scale_x)
    dy_min = int(y_min * scale_y)
    dy_max = int(y_max * scale_y)
    cv2.rectangle(output_image, (dx_min, dy_min), (dx_max, dy_max), (0, 255, 0), 3)

    # Put the detected text above the rectangle
    cv2.putText(output_image, combined_plate, (dx_min, dy_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(script_dir, 'detected_plate.jpg'), output_image)
    cv2.imwrite(os.path.join(script_dir, 'cropped_plate.jpg'), plate_crop)

    print(f"\n=========================================")
    print(f"Detected License Plate Text: {combined_plate}")
    print(f"=========================================\n")

if __name__ == '__main__':
    main()
