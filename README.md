# 📄 Hybrid OCR-Based Document Classification System

A **full-stack document classification system** that uses a **hybrid deep learning model** combining **text (NLP)** and **visual (computer vision)** features to accurately classify PDF documents. The system supports **native text PDFs** and **scanned PDFs** through an OCR fallback mechanism.

---

## 🚀 Key Features

* **Hybrid AI Model**: Combines **DistilBERT (text)** and **ResNet-18 (image)** features
* **OCR Fallback**: Automatically applies OCR (Tesseract) when PDFs have no extractable text
* **Multi-PDF Upload**: Upload and classify multiple PDF files at once
* **Duplicate Handling**: Skips already-processed files
* **Persistent Results**: Saves predictions and category counts to JSON
* **Excel Export**: One-click export of classification results
* **Modern Web UI**: React-based frontend with loading states and error handling
* **REST API**: Built with FastAPI for high performance

---

## 🧠 System Architecture

```
project-root/
│
├── backend/
│   ├── app.py                  # FastAPI backend
│   ├── export_to_excel.py      # Export results to Excel
│   └── model/
│       ├── hybrid_pdf_ocr_model.pt
│       ├── label_map.pkl
│       └── predictions.json
│
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       └── components/
│           ├── FileUpload.jsx
│           └── Results.jsx
│
└── README.md
```

---

## 🤖 Hybrid Model Overview

The classification model fuses **textual** and **visual** information extracted from PDF documents:

### 1. Text Pipeline

* Extracts text using **pdfminer**
* Falls back to **OCR (Tesseract)** for scanned PDFs
* Tokenized using **DistilBERT tokenizer**
* Text embeddings generated via **DistilBERT**

### 2. Image Pipeline

* Converts the first PDF page to an image
* Preprocesses using **OpenCV & TorchVision**
* Feature extraction using **ResNet-18**

### 3. Fusion & Classification

* Concatenates text and image embeddings
* Fully connected fusion layers
* Outputs final document category

---

## 🛠️ Tech Stack

### Backend

* **FastAPI**
* **PyTorch**
* **Transformers (HuggingFace)**
* **TorchVision**
* **Tesseract OCR**
* **pdfminer & pdf2image**

### Frontend

* **React (Vite)**
* **Fetch API**
* **CSS / Bootstrap buttons**

---

## ⚙️ Setup Instructions

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Make sure **Tesseract OCR** is installed and available in your system path.

---

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will communicate with the backend at:

```
http://localhost:8000
```

---

## 📡 API Endpoints

### Classify PDFs

**POST** `/predict`

* Accepts multiple PDF files
* Returns classification results

Example response:

```json
{
  "total_pdfs": 10,
  "category_counts": {
    "Invoice": 4,
    "Report": 6
  },
  "file_names": ["doc1.pdf", "doc2.pdf"]
}
```

---

### Export Results

**POST** `/run-export`

* Executes `export_to_excel.py`
* Exports stored results to Excel

---

## 🖥️ User Interface

* Drag-and-drop or file picker PDF upload
* Loading indicator during classification
* Error handling and validation
* Results visualization by category
* **Export button** to generate Excel reports

---

## 📊 Use Cases

* Automated document sorting
* Enterprise document management
* Scanned archive classification
* Compliance and records processing

---

## 🔒 Notes & Limitations

* Classification is based on the **first page** of each PDF
* OCR accuracy depends on scan quality
* GPU acceleration supported when available

---

## 📜 License

This project is intended for **educational and research purposes**.

---

## ✨ Author

* Developed by **Carlo James G. Arat**
* Documented by **Eldi Nill L. Driz**

Hybrid OCR-powered document intelligence system using modern deep learning techniques.
