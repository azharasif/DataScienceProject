import cv2
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = "/opt/local/bin/tesseract"

def capture_and_extract():
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return "Error: Could not access the camera."

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from the camera.")
        cap.release()
        return "Error: Could not read frame from the camera."

    image_path = "static/captured.jpg"
    if not os.path.exists("static"):
        os.makedirs("static")
    
    cv2.imwrite(image_path, frame)
    print(f"Image saved at {image_path}")
    cap.release()
    
    return extract_text(image_path)

def extract_text(image_path):
    if not os.path.exists(image_path):
        print("Error: Image file does not exist.")
        return "Error: Image file does not exist."

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image file.")
        return "Error: Could not read the image file."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    text = pytesseract.image_to_string(thresh, lang="eng")
    return text

