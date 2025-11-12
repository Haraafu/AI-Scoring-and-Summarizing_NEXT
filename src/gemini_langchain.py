import os
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableMap

class GenFields(BaseModel):
    summary: str = Field(description="Ringkasan faktual dan ringkas")
    rationale: str = Field(description="Alasan singkat risiko berdasarkan sinyal dan konteks")

def make_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
        temperature=0.2,
        max_output_tokens=512
    )

PROMPT = PromptTemplate.from_template("""
Anda analis keamanan data.
1) Buat ringkasan faktual dan ringkas.
2) Jelaskan rationale risiko berdasarkan indikator.
3) Jangan menebak data yang dimasking dan jangan tampilkan PII.
4) Gunakan bahasa sesuai teks.

Konteks:
- URL: {url}
- Timestamp: {timestamp}
- Sinyal: {signals}

Teks (sudah DIMASKING):
{masked_excerpt}

KELUARKAN PERSIS JSON sesuai schema:
{format_instructions}
""".strip())

def build_gemini_chain():
    llm = make_llm()
    parser = JsonOutputParser(pydantic_object=GenFields)
    chain = (
        RunnableMap({
            "url": lambda x: x.get("url") or "-",
            "timestamp": lambda x: x.get("timestamp") or "-",
            "signals": lambda x: ", ".join(x.get("signals", [])) or "-",
            "masked_excerpt": lambda x: x["masked_excerpt"],
            "format_instructions": lambda _: parser.get_format_instructions(),
        })
        | PROMPT
        | llm
        | parser
    )
    return chain

async def gen_summary_rationale_async(masked_excerpt: str, url: Optional[str], timestamp: Optional[str], signals: List[str]):
    chain = build_gemini_chain()
    payload = {"url": url, "timestamp": timestamp, "signals": signals, "masked_excerpt": masked_excerpt}
    try:
        result: GenFields = await chain.ainvoke(payload)
        return {"summary": result.summary, "rationale": result.rationale}
    except Exception:
        short = (masked_excerpt or "")[:300].replace("\n", " ")
        return {"summary": short or "Ringkasan tidak tersedia",
                "rationale": "Gagal parse output model. Periksa format, kuota, atau jaringan."}
