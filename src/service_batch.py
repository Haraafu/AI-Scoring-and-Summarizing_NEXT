import os, asyncio
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.pii_masker import mask_text, detect_signals
from src.gemini_langchain import gen_summary_rationale_async

# Konfigurasi model
MODEL_DIR = os.getenv("MODEL_DIR", "models/roberta-risk")
MODEL_NAME = os.getenv("MODEL_NAME", "xlm-roberta-base")

# Alert threshold default
ALERT_HIGH = float(os.getenv("ALERT_HIGH", "80"))
ALERT_MED  = float(os.getenv("ALERT_MED",  "50"))

def alert_level(score: float) -> str:
    if score >= ALERT_HIGH: return "HIGH"
    if score >= ALERT_MED:  return "MEDIUM"
    return "LOW"

# Load model dan tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR if os.path.exists(MODEL_DIR) else MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR if os.path.exists(MODEL_DIR) else MODEL_NAME)
model.eval()
device = "cpu"
model.to(device)

app = FastAPI(title="NEXT Intelligence AI Batch Service")

# Skema input dan output
class ItemIn(BaseModel):
    text: str
    url: Optional[str] = None
    timestamp: Optional[str] = None
    lang: Optional[str] = "auto"

class BatchIn(BaseModel):
    items: List[ItemIn]
    score_threshold: float = Field(40, ge=0, le=100)
    max_parallel_gemini: int = Field(16, ge=1, le=128)
    max_summary_chars: int = Field(1200, ge=200, le=4000)

class ItemOut(BaseModel):
    index: int
    vulnerability_score: float = Field(ge=0, le=100)
    summary: str
    rationale: str
    alerts: str
    signals: List[str] = []

class BatchOut(BaseModel):
    results: List[ItemOut]

def batch_score_texts(texts: List[str], batch_size: int = 48, max_len: int = 384) -> List[float]:
    scores: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tok = tokenizer(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.no_grad():
            logits = model(**tok).logits.squeeze(-1)  # [B]
        pred01 = torch.clamp(logits, 0.0, 1.0).cpu().numpy()  # 0..1
        scores.extend((pred01 * 100.0).tolist())
    return [round(float(s), 2) for s in scores]

@app.post("/analyze_batch", response_model=BatchOut)
async def analyze_batch(payload: BatchIn):
    n = len(payload.items)
    texts = [it.text for it in payload.items]

    # 1) Scoring batch
    scores = batch_score_texts(texts, batch_size=48)

    # 2) Masking dan sinyal
    masked_list, signals_list, metas = [], [], []
    for it in payload.items:
        sig = detect_signals(it.text)
        msk = mask_text(it.text)[:payload.max_summary_chars]
        masked_list.append(msk)
        signals_list.append(sig)
        metas.append({"url": it.url, "timestamp": it.timestamp})

    # 3) Tentukan yang perlu disummarize
    indices_for_gemini = [i for i, s in enumerate(scores) if s >= payload.score_threshold]
    sem = asyncio.Semaphore(payload.max_parallel_gemini)

    async def call_one(i: int):
        async with sem:
            return i, await gen_summary_rationale_async(
                masked_excerpt=masked_list[i],
                url=metas[i].get("url"),
                timestamp=metas[i].get("timestamp"),
                signals=signals_list[i],
            )

    tasks = [call_one(i) for i in indices_for_gemini]
    gemini_results: Dict[int, Dict[str, str]] = {}
    if tasks:
        for i, res in await asyncio.gather(*tasks):
            gemini_results[i] = res

    # 4) Susun hasil
    results: List[ItemOut] = []
    for i in range(n):
        score = scores[i]
        lvl = alert_level(score)
        if i in gemini_results:
            summary = gemini_results[i].get("summary", "Ringkasan tidak tersedia")
            rationale = gemini_results[i].get("rationale", "Rationale tidak tersedia")
        else:
            # Fallback ringkas jika di bawah threshold
            summary = masked_list[i].split("\n")[0][:300]
            rationale = "Risiko rendah atau di bawah threshold sehingga tidak dikirim ke model generatif."
        results.append(ItemOut(
            index=i,
            vulnerability_score=score,
            summary=summary,
            rationale=rationale,
            alerts=lvl,
            signals=signals_list[i]
        ))

    return BatchOut(results=results)
