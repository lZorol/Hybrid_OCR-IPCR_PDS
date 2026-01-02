import { useState, useRef } from 'react'
import './FileUpload.css'

function FileUpload({ onUpload, disabled }) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState([])
  const fileInputRef = useRef(null)

  const handleDragEnter = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    const files = Array.from(e.dataTransfer.files)
    const pdfFiles = files.filter((file) => file.type === 'application/pdf')

    if (pdfFiles.length === 0) {
      alert('Please drop PDF files only')
      return
    }

    setSelectedFiles(pdfFiles)
  }

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files)
    const pdfFiles = files.filter((file) => file.type === 'application/pdf')

    if (pdfFiles.length === 0) {
      alert('Please select PDF files only')
      return
    }

    setSelectedFiles(pdfFiles)
  }

  const handleUpload = () => {
    if (selectedFiles.length > 0) {
      onUpload(selectedFiles)
      setSelectedFiles([])
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const handleClear = () => {
    setSelectedFiles([])
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="file-upload-section">
      <div
        className={`drag-drop-zone ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <div className="drag-drop-content">
          <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
          </svg>
          <h3>Drag & drop your PDFs here</h3>
          <p>or click the button below to select files</p>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />

      <div className="button-group">
        <button
          className="btn btn-select"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
        >
          Select PDFs
        </button>

        {selectedFiles.length > 0 && (
          <>
            <button
              className="btn btn-upload"
              onClick={handleUpload}
              disabled={disabled}
            >
              Upload & Classify ({selectedFiles.length})
            </button>
            <button
              className="btn btn-clear"
              onClick={handleClear}
              disabled={disabled}
            >
              Clear
            </button>
          </>
        )}
      </div>

      {selectedFiles.length > 0 && (
        <div className="file-list">
          <h4>Selected Files ({selectedFiles.length}):</h4>
          <ul>
            {selectedFiles.map((file, index) => (
              <li key={index}>
                <span className="file-icon">📄</span>
                <span className="file-name">{file.name}</span>
                <span className="file-size">({(file.size / 1024).toFixed(2)} KB)</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default FileUpload
