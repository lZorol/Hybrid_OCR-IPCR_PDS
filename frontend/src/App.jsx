import { useState } from 'react'
import './App.css'
import FileUpload from './components/FileUpload'
import Results from './components/Results'

function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileUpload = async (files) => {
    if (files.length === 0) {
      setError('Please select at least one PDF file')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      files.forEach((file) => {
        formData.append('files', file)
      })

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to classify PDFs')
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err.message || 'An error occurred while processing files')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleRunExport = async () => {
  try {
    await fetch('http://localhost:8000/run-export', {
      method: 'POST',
    })
  } catch (err) {
    alert('Failed to run export script')
  }
}


  return (
    <div className="app-container">
      <div className="app-card">
        <h1 className="app-title">📄 PDF Classifier</h1>
        <p className="app-subtitle">Upload multiple PDFs to classify them by category</p>

        <FileUpload onUpload={handleFileUpload} disabled={loading} />
        <button class="btn btn-secondary" onClick={handleRunExport}>Export</button>

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Classifying PDFs...</p>
          </div>
        )}

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}
        {results && !loading && (
          <Results data={results} />
        )}
      </div>
    </div>
  )
}

export default App
