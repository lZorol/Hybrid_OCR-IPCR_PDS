from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from collections import defaultdict
import tempfile
import os
import json
from pathlib import Path
import subprocess

# ================================
# 📚 Imports
# ================================
import torch
import torch.nn as nn
import joblib
import numpy as np
import cv2

from transformers import DistilBertTokenizer, DistilBertModel
from torchvision import models, transforms
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from PIL import Image
import pytesseract

# ================================
# 🚀 FastAPI App
# ================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# ⚙️ Config
# ================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = Path(__file__).resolve().parent / "model"
MODELS_DIR.mkdir(exist_ok=True)

PREDICTIONS_JSON = MODELS_DIR / "predictions.json"

# ================================
# 🛠 Helper Functions
# ================================
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        text = extract_text(pdf_path).strip()
        if len(text) > 50:
            return text
    except Exception:
        pass

    # OCR fallback
    ocr_text = ""
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1)
        for img in images:
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(
                gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            ocr_text += pytesseract.image_to_string(thresh) + " "
    except Exception:
        pass

    return ocr_text.strip()


def pdf_to_image(pdf_path: str) -> Image.Image:
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1)
        return images[0].convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color="white")

# ================================
# 🤖 Hybrid Model (YOUR VERSION)
# ================================
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.text_model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        self.vision_model = models.resnet18(pretrained=False)
        self.vision_model.fc = nn.Linear(
            self.vision_model.fc.in_features, 256
        )

        self.fc_fusion = nn.Sequential(
            nn.Linear(256 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        text_out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = text_out.last_hidden_state[:, 0, :]
        img_feat = self.vision_model(image)
        fused = torch.cat((text_feat, img_feat), dim=1)
        return self.fc_fusion(fused)

# ================================
# 📦 Load Model & Tokenizer
# ================================
label_map = joblib.load(MODELS_DIR / "label_map.pkl")
idx_to_label = {v: k for k, v in label_map.items()}

model = HybridModel(num_classes=len(label_map))
model.load_state_dict(
    torch.load(MODELS_DIR / "hybrid_pdf_ocr_model.pt", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================================
# 📄 API Endpoint
# ================================
@app.post("/predict")
async def predict_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload multiple PDFs.
    - Skips duplicates
    - Classifies new PDFs
    - Saves results to JSON
    """

    # Load existing results
    existing = {
        "file_names": [],
        "category_counts": {}
    }

    if PREDICTIONS_JSON.exists():
        with open(PREDICTIONS_JSON, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing_files_lower = {f.lower() for f in existing["file_names"]}
    counts = defaultdict(int, existing["category_counts"])

    new_files = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        if file.filename.lower() in existing_files_lower:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            pdf_path = tmp.name

        # ---- Text ----
        text = extract_text_from_pdf(pdf_path)
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        # ---- Image ----
        img = pdf_to_image(pdf_path)
        img_tensor = image_transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(
                encoding["input_ids"].to(DEVICE),
                encoding["attention_mask"].to(DEVICE),
                img_tensor.to(DEVICE)
            )
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = idx_to_label[pred_idx]

        counts[pred_label] += 1
        new_files.append(file.filename)

        os.remove(pdf_path)

    # Save updated results
    result = {
        "total_pdfs": sum(counts.values()),
        "category_counts": dict(counts),
        "file_names": existing["file_names"] + new_files
    }

    with open(PREDICTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


# export button
@app.post("/run-export")
def run_export_script():
    script_path = Path(__file__).resolve().parent / "export_to_excel.py"

    subprocess.Popen(["python", str(script_path)])

    return {"status": "started"}