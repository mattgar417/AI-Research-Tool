import os
from ml.tools import text_analysis, table_analysis, figure_analysis, compare_papers, entity_recognition

# Paths to test files
PDF_PATH = "test_samples/paper.pdf"
PDF_PATH1 = "test_samples/paper1.pdf"
IMG_PATH = "test_samples/figure1.png"
OUTPUT_DIR = "test_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nTesting text analysis...")
summary = text_analysis(PDF_PATH, OUTPUT_DIR, "Summarize the text of this paper")
print("Summary:\n", summary[:500], "...\n")

print("\nTesting table analysis...")
summary = table_analysis(PDF_PATH, OUTPUT_DIR, "Summarize the tables of this paper")
print("Summary:\n", summary[:500], "\n")

print("\nTesting figure analysis...")
summary = figure_analysis(PDF_PATH, OUTPUT_DIR, "Summarize the figures of this paper")
print("Summary:\n", summary[:500], "\n")

print("\nTesting paper comparison...")
summary = compare_papers(PDF_PATH, PDF_PATH1, OUTPUT_DIR)
print("Summary:\n", summary[:500], "\n")

print("\nTesting named entity recognition...")
summary = table_analysis(PDF_PATH, OUTPUT_DIR, "Summarize the figures of this paper")
print("Summary:\n", summary[:500], "\n")
