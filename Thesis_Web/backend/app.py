# backend/app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch, joblib, numpy as np, cv2, tempfile, os
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision import transforms, models
from pdf2image import convert_from_path
from PIL import Image
from pdfminer.high_level import extract_text
import pytesseract
import torch.nn as nn

# --- App setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper functions ---
def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path).strip()
        if len(text) > 50:
            return text

        # OCR fallback
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(pdf_path, dpi=150, fmt="png", output_folder=temp_dir)
            text_ocr = ""
            for img in images[:1]:
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text_ocr += pytesseract.image_to_string(thresh) + " "
        return text_ocr.strip()
    except:
        return ""

def pdf_to_image(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=150)
        return images[0].convert("RGB")
    except:
        return Image.new("RGB", (224, 224), color="white")

# --- Model definition ---
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.vision_model = models.resnet18(pretrained=False)
        self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 256)
        self.fc_fusion = nn.Sequential(
            nn.Linear(256 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]
        img_feat = self.vision_model(image)
        fused = torch.cat((text_feat, img_feat), dim=1)
        return self.fc_fusion(fused)

# --- Load model ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
label_map = joblib.load("model/label_map.pkl")  # {'ipcr':0,'pds':1,...}
idx_to_label = {v: k for k, v in label_map.items()}

model = HybridModel(num_classes=len(label_map))
model.load_state_dict(torch.load("model/hybrid_pdf_ocr_model.pt", map_location=DEVICE))
model.to(DEVICE).eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])


# =========================================================
#  NEW ENDPOINT — MULTIPLE FILES + CATEGORY SUMMARY
# =========================================================
@app.post("/predict-multiple")
async def predict_multiple(files: list[UploadFile] = File(...)):
    results = []
    category_count = {label: 0 for label in idx_to_label.values()}

    for file in files:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extract text/image
        text = extract_text_from_pdf(tmp_path)
        img = pdf_to_image(tmp_path)

        os.remove(tmp_path)

        # Encode input for model
        enc = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE),
                img_tensor.to(DEVICE)
            )

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = idx_to_label[pred_idx]
        confidence = float(probs[pred_idx] * 100)

        category_count[pred_label] += 1

        results.append({
            "filename": file.filename,
            "predicted_class": pred_label,
            "confidence": confidence,
            "probabilities": {idx_to_label[i]: float(p * 100) for i, p in enumerate(probs)}
        })

    return {
        "categories": list(idx_to_label.values()),
        "summary": category_count,
        "results": results
    }
