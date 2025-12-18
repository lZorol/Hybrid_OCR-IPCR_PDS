// App.jsx
import React, { useState } from "react";
import { Upload, FileText, CheckCircle, Loader2 } from "lucide-react";
import "./App.css";

export default function PdfUploader() {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFiles(e.dataTransfer.files);
    }
  };

  const handleUpload = async () => {
    if (!files || files.length === 0) {
      return alert("Please select at least one PDF file");
    }
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/predict-multiple", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      alert("❌ Error connecting to backend. Make sure FastAPI is running.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const getCategoryColor = (category) => {
    const colors = ["gradient-blue", "gradient-purple", "gradient-emerald", "gradient-amber"];
    const index = category.charCodeAt(0) % colors.length;
    return colors[index];
  };

  return (
    <div className="app-container">
      <div className="content-wrapper">
        {/* Header */}
        <div className="header">
          <div className="header-icon">
            <FileText className="icon-large" />
          </div>
          <h1 className="header-title">IPCR & PDS GENERATOR</h1>
          <p className="header-subtitle">Hybrid OCR Enabled & Text-based AI System for Intelligent Classification of IPCR and PDS LSPU Documents</p>
        </div>

        {/* Upload Section */}
        <div className="upload-section">
          <div
            className={`drop-zone ${dragActive ? "drop-zone-active" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="drop-zone-content">
              <div className="upload-icon-wrapper">
                <Upload className="upload-icon" />
              </div>
              <h3 className="drop-zone-title">
                Drop PDF files here or click to browse
              </h3>
              <p className="drop-zone-subtitle">Support for multiple PDF files</p>
              
              <input
                type="file"
                accept="application/pdf"
                multiple
                onChange={(e) => setFiles(e.target.files)}
                className="file-input-hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="select-files-btn">
                Select Files
              </label>
            </div>
          </div>

          {files && files.length > 0 && (
            <div className="files-preview">
              <p className="files-count">
                {files.length} file{files.length > 1 ? "s" : ""} selected
              </p>
              <div className="file-chips">
                {Array.from(files).map((file, i) => (
                  <div key={i} className="file-chip">
                    <FileText className="file-chip-icon" />
                    <span className="file-chip-name">{file.name}</span>
                  </div>
                ))}
              </div>
              <button
                onClick={handleUpload}
                disabled={loading}
                className="upload-button"
              >
                {loading ? (
                  <>
                    <Loader2 className="spinner" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Upload className="button-icon" />
                    Analyze PDFs
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Results Section */}
        {result && (
          <div className="results-section">
            {/* Summary Cards */}
            <div className="summary-card">
              <div className="card-header">
                <CheckCircle className="check-icon" />
                <h2 className="card-title">Classification Summary</h2>
              </div>
              <div className="summary-grid">
                {Object.entries(result.summary).map(([category, count]) => (
                  <div key={category} className="summary-item">
                    <div className={`summary-item-bg ${getCategoryColor(category)}`}></div>
                    <div className="summary-item-content">
                      <p className="summary-category">{category}</p>
                      <p className="summary-count">{count}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Detailed Results */}
            <div className="details-card">
              <h2 className="card-title">Detailed Results</h2>
              <div className="details-list">
                {result.results.map((file, idx) => (
                  <div key={idx} className="file-result">
                    <div className="file-result-header">
                      <div className="file-info">
                        <div className="file-icon-wrapper">
                          <FileText className="file-icon" />
                        </div>
                        <div>
                          <p className="file-name">{file.filename}</p>
                          <p className="file-type">PDF Document</p>
                        </div>
                      </div>
                      <div className="file-classification">
                        <div className="classification-badge">
                          <span className="classification-text">
                            {file.predicted_class}
                          </span>
                        </div>
                        <p className="confidence-text">
                          {file.confidence.toFixed(2)}% confidence
                        </p>
                      </div>
                    </div>
                    
                    <details className="probabilities-details">
                      <summary className="probabilities-summary">
                        <span>View Probabilities</span>
                        <svg className="chevron-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      </summary>
                      <div className="probabilities-content">
                        <pre className="probabilities-pre">
                          {JSON.stringify(file.probabilities, null, 2)}
                        </pre>
                      </div>
                    </details>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}