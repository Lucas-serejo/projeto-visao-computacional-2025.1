import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import time

# Set the path to the Tesseract executable if it's not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the card counting variables
card_count = 0
card_values = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
    '7': 0, '8': 0, '9': 0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
}

# Variables to track the last detected card and its timestamp
last_detected_card = None
last_detection_time = time.time()

# Delay between card detections (in seconds)
detection_delay = 3  # Adjust this value as needed

# Define the fixed rectangle (ROI) for card detection
roi_x, roi_y, roi_width, roi_height = 200, 100, 300, 400  # Adjust these values as needed

# Function to preprocess the image for better OCR accuracy
def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to binarize the image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return binary

# Function to detect and recognize card values
def detect_card_value(image):
    global card_count, last_detected_card, last_detection_time

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Use Tesseract to detect text in the image
    custom_config = r'--oem 3 --psm 7'  # PSM 7: Treat the image as a single text line
    details = pytesseract.image_to_data(processed_image, output_type=Output.DICT, config=custom_config)

    # Loop over each detected text element
    for i in range(len(details['text'])):
        if int(details['conf'][i]) > 60:  # Confidence threshold
            text = details['text'][i].strip().upper()  # Convert to uppercase for consistency
            if text in card_values:
                # Check if the same card was detected recently
                current_time = time.time()
                if text == last_detected_card and (current_time - last_detection_time) < detection_delay:
                    print(f"Ignoring duplicate card: {text}")
                    return None

                # Update the card count and last detection details
                card_count += card_values[text]
                last_detected_card = text
                last_detection_time = current_time
                print(f"Detected card: {text}, Current count: {card_count}")
                return text

    return None

# Function to localize the card within the fixed rectangle
def localize_card(frame):
    # Extract the region of interest (ROI) from the frame
    roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and aspect ratio
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the aspect ratio
        aspect_ratio = w / float(h)

        # Define criteria for a card (adjust based on your card size)
        if w * h > 5000 and 0.6 < aspect_ratio < 1.4:  # Adjust these values as needed
            # Draw the bounding box on the ROI
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the card region from the ROI
            card_roi = roi[y:y + h, x:x + w]

            return card_roi, (x + roi_x, y + roi_y, w, h)  # Return global coordinates

    return None, None

# Main function to capture video and process frames
def main():
    global last_detection_time

    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the fixed rectangle on the frame
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

        # Localize the card within the fixed rectangle
        card_roi, bbox = localize_card(frame)

        if card_roi is not None:
            # Check if enough time has passed since the last detection
            current_time = time.time()
            if (current_time - last_detection_time) >= detection_delay:
                # Detect card value in the localized region
                card_value = detect_card_value(card_roi)

                # Display the bounding box and card value on the frame
                if card_value is not None:
                    x, y, w, h = bbox
                    cv2.putText(frame, f"Card: {card_value}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the card count on the frame
        cv2.putText(frame, f"Card Count: {card_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Card Counting", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()