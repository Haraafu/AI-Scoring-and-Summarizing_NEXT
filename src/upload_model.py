from pathlib import Path
from huggingface_hub import create_repo, upload_folder

# ==== GANTI INI dengan username kamu yang BENAR ====
HF_USERNAME = "Harafu"  # contoh; ganti sesuai `huggingface-cli whoami`
REPO_NAME   = "roberta-risk-next"
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"

# Path absolut ke folder model (naik 1 level dari /src ke root proyek)
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models" / "roberta-risk"

print("Repo ID :", REPO_ID)
print("Model dir:", MODEL_DIR)

if not MODEL_DIR.is_dir():
    raise SystemExit(f"Folder model tidak ditemukan: {MODEL_DIR}")

# Buat repo (private). Kalau sudah ada, tetap lanjut.
try:
    create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print("create_repo warning:", e)

# Upload seluruh folder model
upload_folder(
    repo_id=REPO_ID,
    folder_path=str(MODEL_DIR),
    repo_type="model"
)

print(f"Upload selesai. Cek: https://huggingface.co/{REPO_ID}")
