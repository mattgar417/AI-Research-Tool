from flask import Flask, render_template, request, redirect
import cv2
import pytesseract
from ultralytics import YOLO
from pdf2image import convert_from_path

app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["CSV_FOLDER"] = "static/csv"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["CSV_FOLDER"], exist_ok=True)

GCS_BUCKET = "research_tool_bucket1"  # replace with your bucket name

# Load YOLO model
model = YOLO("yolov5s.pt")  # Change to path of your custom-trained weights if available

def extract_data_from_image(img_path):
    # Read the image
    img = cv2.imread(img_path)
    
    # Perform YOLO object detection
    results = model(img)

    # Initialize lists for detected elements
    detected_elements = []

    # Iterate over detections
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2, confidence, cls = detection.xyxy[0]  # Bounding box coordinates
            label = model.names[int(cls)]  # Class name
            
            # Crop the detected area
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
            
            # OCR to extract text if it's a text-based area
            if label in ["table", "figure", "text"]:  # Adjust labels as needed
                text = pytesseract.image_to_string(cropped_img)
                detected_elements.append((label, text))
            else:
                detected_elements.append((label, cropped_img))  # For non-text data

    return detected_elements

def save_csv(data, filename):
    path = os.path.join(app.config["CSV_FOLDER"], filename)
    with open(path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["label", "confidence", "text"])
        writer.writeheader()
        writer.writerows(data)
    return path

def upload_to_gcs(local_file_path, gcs_path):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_file_path)
    return f"https://storage.googleapis.com/{GCS_BUCKET}/{gcs_path}"

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    image_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)
            results = extract_data_from_image(image_path)

    return render_template("index.html", results=results, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
