import os, cv2, pytesseract, camelot, json
import pandas as pd
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from openai import OpenAI
from ultralytics import YOLO
from models import table_qa, summarize_text, generate_caption, figure_type_classification, paper_similarity, named_entity_recognition

model = YOLO("yolov5su.pt")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def text_analysis(pdf_path, output_dir, query):
    summary = summarize_text(pdf_path, query)
   
    out_path = os.path.join(output_dir, "text_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return {"summary": summary}

def table_analysis(pdf_path, output_dir, queries):
    tables = []

    all_tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
    for i, table in enumerate(all_tables):
        csv_path = os.path.join(output_dir, f"table_{i+1}.csv")
        table.to_csv(csv_path)
        tables.append(pd.read_csv(csv_path).to_dict(orient="list"))

    results = table_qa(tables, queries)

    out_path = os.path.join(output_dir, "table_qa_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

def figure_analysis(pdf_path, output_dir, query):
    pages = convert_from_path(pdf_path, dpi=300)
    all_text = ""
    figures = []

    for i, page in enumerate(pages):
        img_path = os.path.join(output_dir, f"page_{i+1}.jpg")
        page.save(img_path, "JPEG")
        img = cv2.imread(img_path)

        classification = figure_type_classification(img_path)
        all_text += classification.get("predicted", "") + "\n"

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
                    caption = generate_caption(text, query)
                    figures.append({"path": fig_path, "caption": caption})

                elif label == "text":
                    all_text += pytesseract.image_to_string(cropped) + "\n"

    with open(os.path.join(output_dir, "figure_summary.txt"), "w", encoding="utf-8") as f:
        f.write(all_text)
        f.write("\n\n--- FIGURE CAPTIONS ---\n")
        for fig in figures:
            f.write(f"{fig['path']}: {fig['caption']}\n")

    return {"figures": figures}

def compare_papers(pdf_path1, pdf_path2, output_dir):
    text1 = extract_text(pdf_path1)
    text2 = extract_text(pdf_path2)

    p1 = {"title": os.path.basename(pdf_path1), "paper": text1}
    p2 = {"title": os.path.basename(pdf_path2), "paper": text2}

    results = paper_similarity(p1, p2)

    out_path = os.path.join(output_dir, "comparison_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results

def entity_recognition(pdf_path, output_dir):
    text = extract_text(pdf_path)
    results = named_entity_recognition(text)

    with open(os.path.join(output_dir, "entities.json"), "w", encoding="utf-8") as f:
        json.dump({"entities": results["entities"], "counts": results["counts"]}, f, indent=2)

    with open(os.path.join(output_dir, "entities.html"), "w", encoding="utf-8") as f:
        f.write(results["html"])
    
    return results
