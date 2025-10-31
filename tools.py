import os, cv2, pytesseract, camelot, tempfile
from pdf2image import convert_from_path
from datetime import datetime
from openai import OpenAI
from ultralytics import YOLO
from pdfminer.high_level import extract_text

model = YOLO("yolov5su.pt")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_text(pdf_path):
    text = extract_text(pdf_path)

    prompt = f"Summarize this academic paper:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

def generate_caption(img_path):
    image = cv2.imread(img_path)

    extracted_text = pytesseract.image_to_string(image)
    context = extracted_text.strip() if extracted_text else "No readable text detected."

    prompt = f"Write a short descriptive caption for this research figure or table:\n\n{context}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=120
    )
    return response.choices[0].message.content.strip()

def extract_tables_from_pdf(pdf_path, output_folder):
    tables = []
    all_tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
    for i, table in enumerate(all_tables):
        csv_path = os.path.join(output_folder, f"table_{i+1}.csv")
        table.to_csv(csv_path)
        tables.append(csv_path)
    return tables

def extract_data_from_pdf(pdf_path, output_dir):
    pages = convert_from_path(pdf_path, dpi=300)
    all_text = ""
    figures = []

    for i, page in enumerate(pages):
        img_path = os.path.join(output_dir, f"page_{i+1}.jpg")
        page.save(img_path, "JPEG")
        img = cv2.imread(img_path)

        results = model(img)
        for result in results:
            for detection in result.boxes:
                x1, y1, x2, y2, conf, cls = detection.xyxy[0]
                label = model.names[int(cls)]
                cropped = img[int(y1):int(y2), int(x1):int(x2)]

                if label == "figure":
                    fig_path = os.path.join(output_dir, f"figure_{i+1}_{int(conf*100)}.jpg")
                    cv2.imwrite(fig_path, cropped)
                    text = pytesseract.image_to_string(cropped)
                    caption = generate_caption(text)
                    figures.append({"path": fig_path, "caption": caption})

                elif label == "text":
                    all_text += pytesseract.image_to_string(cropped) + "\n"

    summary = summarize_text(all_text)
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
        f.write("\n\n--- FIGURE CAPTIONS ---\n")
        for fig in figures:
            f.write(f"{fig['path']}: {fig['caption']}\n")

    return {"figures": figures, "summary": summary}