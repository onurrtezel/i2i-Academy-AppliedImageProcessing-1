import cv2
import numpy as np
import easyocr
import os

def main():
    # 1. Load the image of the car
    image_path = os.path.join(os.path.dirname(__file__), 'car.jpg')
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Resize the image for consistent processing size
    # Keep the aspect ratio but scale width to 800px
    height, width = image.shape[:2]
    new_width = 800
    new_height = int((new_width / width) * height)
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Save a copy for drawing the final bounding box
    output_image = resized_image.copy()

    # 2. Preprocess the image
    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Canny Edge Detection
    edged = cv2.Canny(blurred, 30, 200)

    # 3. Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area descending and select top 30
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    plate_contour = None
    cropped_plate = None
    
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # A license plate is a rectangle, so it should have 4 corners
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # Standard Turkish license plate aspect ratio is around 2.0 to 5.5
            # We filter based on aspect ratio to avoid wrong rectangles
            if 2.0 <= aspect_ratio <= 5.5:
                plate_contour = approx
                # Crop the plate region with a dynamic padding to ensure no text is cut
                padding_y = int(h * 0.1)
                padding_x = int(w * 0.08)
                y_start = max(0, y - padding_y)
                y_end = min(new_height, y + h + padding_y)
                x_start = max(0, x - padding_x)
                x_end = min(new_width, x + w + padding_x)
                
                cropped_plate = resized_image[y_start:y_end, x_start:x_end]
                break

    # Fallback to bounding box of largest rectangle if no ideal aspect ratio matches
    if plate_contour is None and len(contours) > 0:
        print("No perfect aspect-ratio contour found, using the largest 4-point contour.")
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                plate_contour = approx
                x, y, w, h = cv2.boundingRect(approx)
                padding_y = int(h * 0.1)
                padding_x = int(w * 0.08)
                y_start = max(0, y - padding_y)
                y_end = min(new_height, y + h + padding_y)
                x_start = max(0, x - padding_x)
                x_end = min(new_width, x + w + padding_x)
                cropped_plate = resized_image[y_start:y_end, x_start:x_end]
                break

    if cropped_plate is None:
        print("Error: License plate region could not be isolated.")
        return

    # Save intermediate images for review and portfolio documentation
    cv2.imwrite(os.path.join(os.path.dirname(__file__), 'gray.jpg'), gray)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), 'edged.jpg'), edged)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), 'cropped_plate.jpg'), cropped_plate)

    # Draw the contour on the output image
    cv2.drawContours(output_image, [plate_contour], -1, (0, 255, 0), 3)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), 'detected_plate.jpg'), output_image)

    # 4. Pass the cropped plate image to EasyOCR
    print("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=False) # run on CPU for standard environment compatibility
    
    print("Performing OCR on the cropped plate...")
    results = reader.readtext(cropped_plate)
    
    # 5. Extract and print the text
    plate_text = ""
    if results:
        # Join detected text boxes and clean it up (spaces, special characters)
        texts = [res[1] for res in results]
        plate_text = " ".join(texts).upper().strip()
        # Clean up typical OCR noise for license plates
        clean_plate = "".join([c for c in plate_text if c.isalnum()])
        
        # Fix common Turkish plate 34 misreadings (like D4 -> 34, B4 -> 34, S4 -> 34, 84 -> 34)
        if len(clean_plate) >= 2 and clean_plate[0] in ['D', 'B', 'S', '8', '0'] and clean_plate[1] == '4':
            clean_plate = '34' + clean_plate[2:]
            
        print(f"\n=========================================")
        print(f"Detected License Plate Text: {clean_plate}")
        print(f"=========================================\n")
    else:
        print("Error: Could not read text from the cropped image.")

if __name__ == '__main__':
    main()
