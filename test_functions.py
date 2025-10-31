import os
from app import (
    summarize_text,
    generate_caption,
    extract_tables_from_pdf,
    extract_data_from_pdf
)

# Paths to test files
PDF_PATH = "test_samples/paper.pdf"
IMG_PATH = "test_samples/figure1.png"
OUTPUT_DIR = "test_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nTesting summarize_text()...")
summary = summarize_text(PDF_PATH)
print("Summary:\n", summary[:500], "...\n")

print("\nTesting generate_caption()...")
caption = generate_caption(IMG_PATH)
print("Caption:\n", caption, "\n")

print("\nTesting extract_tables_from_pdf()...")
tables = extract_tables_from_pdf(PDF_PATH, OUTPUT_DIR)
print(f"Extracted {len(tables)} tables.")
for t in tables:
    print(" -", t)

print("\nTesting extract_data_from_pdf()...")
results = extract_data_from_pdf(PDF_PATH, OUTPUT_DIR)
print("Summary length:", len(results['summary']))
print("Detected figures:", len(results['figures']))
for fig in results["figures"]:
    print(" -", fig["path"], "| Caption:", fig["caption"][:80])