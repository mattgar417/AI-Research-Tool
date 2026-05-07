from transformers import AutoTokenizer, TapasForQuestionAnswering, TapasTokenizer
from adapters import AutoAdapterModel
import os, cv2, pytesseract
import pandas as pd
import torch
import torch.nn.functional as F
import clip
import scispacy
import spacy
from spacy import displacy
from openai import OpenAI
from ultralytics import YOLO
from pdfminer.high_level import extract_text
from PIL import Image

model = YOLO("yolov5su.pt")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

specter_tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
specter_model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
specter_model.load_adapter("allenai/specter2", source="hf", set_active=True)

tapas_tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
tapas_model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

nlp = spacy.load("en_core_sci_sm")

def summarize_text(pdf_path, query):
    text = extract_text(pdf_path)

    prompt = f"Answer the following question '{query}' for this academic paper:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

def generate_caption(img_path, query):
    image = cv2.imread(img_path)

    extracted_text = pytesseract.image_to_string(image)
    context = extracted_text.strip() if extracted_text else "No readable text detected."

    prompt = f"Answer the following question '{query}' for this research figure:\n\n{context}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=120
    )
    return response.choices[0].message.content.strip()

def paper_embeddings(papers: list[dict]): 
    text_batch = [d.get("title", "") + specter_tokenizer.sep_token + d.get("abstract", "") for d in papers]

    inputs = specter_tokenizer(text_batch, padding=True, truncation=True, max_length=512, return_tensors="pt", return_token_type_ids=False)
    output = specter_model(**inputs)

    embeddings = output.last_hidden_state[:, 0, :]
    return embeddings

def paper_similarity(p1, p2):
    embeddings = paper_embeddings([p1, p2])
    e1, e2 = embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)

    score = F.cosine_similarity(e1, e2).item()

    if score >= 0.85:
        label = "Highly similar (same topic/method)"
    elif score >= 0.65:
        label = "Related (overlapping domain)"
    elif score >= 0.40:
        label = "Loosely related"
    else:
        label = "Distinct topics"

    return {"score": round(score, 4), "label": label}

def figure_type_classification(image_path):
    figure_labels = [
        "a bar chart",
        "a line graph",
        "a scatter plot",
        "a confusion matrix",
        "a neural network architecture diagram",
        "a flowchart or pipeline diagram",
        "a table or grid",
        "a microscopy or medical image",
        "a heatmap",
        "a pie chart",
    ]

    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(figure_labels).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity[0].cpu().numpy()

        logits_per_image = clip_model(image)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    label_probs = {label: round(float(prob), 4) for label, prob in zip(figure_labels, probs)}
    sorted_probs = dict(sorted(label_probs.items(), key=lambda x: x[1], reverse=True))

    top_label = next(iter(sorted_probs))
    return {
        "predicted": top_label,
        "confidence": sorted_probs[top_label],
        "all_probs": sorted_probs,
    }

def table_qa(tables: list[dict], queries: list[dict]):
    id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    all_results = []

    for table_idx, table in enumerate(tables):
        data = pd.DataFrame.from_dict(table)
        data.astype(str)

        inputs = tapas_tokenizer(
            table=data,
            queries=queries,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = tapas_model(**inputs)
        predicted_answer_coordinates, predicted_aggregation_indices = tapas_tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())

        aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

        answers = []
        for coordinates in predicted_answer_coordinates:
            if len(coordinates) == 1:
                # only a single cell:
                answers.append(data.iat[coordinates[0]])
            else:
                # multiple cells
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(data.iat[coordinate])
                answers.append(", ".join(cell_values))

        
        for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
            '''
            print(query)
            if predicted_agg == "NONE":
                print("Predicted answer: " + answer)
            else:
                print("Predicted answer: " + predicted_agg + " > " + answer)
            '''

            all_results.append({
                "table_index": table_idx,
                "query": query,
                "answer": answer,
                "aggregation": predicted_agg,
                "formatted": f"{predicted_agg} > {answer}" if predicted_agg != "NONE" else answer,
            })

    return all_results

def named_entity_recognition(paper):
    doc = nlp(paper)

    entities = [
        {
            "text":  ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end":   ent.end_char,
        }
        for ent in doc.ents
    ]

    counts: dict[str, int] = {}
    for ent in entities:
        counts[ent["label"]] = counts.get(ent["label"], 0) + 1

    html = displacy.render(doc, style="ent", page=False, jupyter=False)

    return {
        "entities": entities,
        "counts":   counts,
        "html":     html
    }