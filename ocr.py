import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/opt/local/bin/tesseract"  
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    return thresh

def extract_text_tesseract(image_path):
    img = preprocess_image(image_path)
    text = pytesseract.image_to_string(img, lang="eng")
    return text

