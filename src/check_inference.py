from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np

MODEL_DIR = "models/roberta-risk"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def score_text(text):
    inputs = tok(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze().item()
    score = max(0.0, min(100.0, logits * 100.0))  # normalisasi 0–100
    return round(score, 2)

samples = [
    "Email karyawan john@corp.co dan password disebutkan di forum",
    "Dokumen umum tanpa PII",
    "Daftar pelanggan lengkap dengan nomor HP",
    "Artikel umum tentang keamanan data"
]

for s in samples:
    print(f"[{score_text(s):.2f}] {s}")
