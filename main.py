import sys
from flask import Flask, render_template, request
from camera_ocr import capture_and_extract  
from ocr import extract_text_tesseract  

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/capture")
def capture():
    image_path = capture_and_extract()
    print('image_path:------ ', image_path)
    text = extract_text_tesseract(image_path)
    print('--'  , text)
    return render_template("result.html", extracted_text=text)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    file_path = "static/" + file.filename
    file.save(file_path)

    text = extract_text_tesseract(file_path)  
    return render_template("result.html", extracted_text=text)

if __name__ == "__main__":
    app.run(debug=True)
