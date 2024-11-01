import cv2
import pytesseract
from ultralytics import YOLO
from pdf2image import convert_from_path

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

def process_pdf(file_path):
    # Convert PDF pages to images
    pages = convert_from_path(file_path, 300)
    extracted_data = []

    # Process each page
    for i, page in enumerate(pages):
        img_path = f"page_{i}.jpg"
        page.save(img_path, "JPEG")
        page_data = extract_data_from_image(img_path)
        extracted_data.append(page_data)
    
    return extracted_data

# Path to research paper PDF
pdf_path = "research_paper.pdf"
data = process_pdf(pdf_path)

# Analyze or save extracted data
for page_idx, page_content in enumerate(data):
    print(f"\n--- Page {page_idx + 1} ---")
    for element in page_content:
        label, content = element
        if isinstance(content, str):
            print(f"[{label}] Text: {content[:100]}...")  # Print first 100 characters
        else:
            print(f"[{label}] Image detected")
