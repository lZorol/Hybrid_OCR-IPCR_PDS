import './Results.css'

function Results({ data }) {
  const { total_pdfs, category_counts } = data
  const categories = Object.entries(category_counts).sort((a, b) => b[1] - a[1])
  const maxCount = Math.max(...Object.values(category_counts), 1)

  return (
    <div className="results-section">
      <h2 className="results-title">✅ Classification Results</h2>

      <div className="summary-card">
        <div className="summary-stat">
          <div className="stat-number">{total_pdfs}</div>
          <div className="stat-label">Total PDFs Processed</div>
        </div>
        <div className="summary-stat">
          <div className="stat-number">{categories.length}</div>
          <div className="stat-label">Categories Found</div>
        </div>
      </div>

      <div className="categories-list">
        {categories.map(([category, count]) => {
          const percentage = (count / total_pdfs) * 100
          const barWidth = (count / maxCount) * 100

          return (
            <div key={category} className="category-item">
              <div className="category-header">
                <span className="category-name">{category}</span>
                <span className="category-count">{count} ({percentage.toFixed(1)}%)</span>
              </div>
              <div className="progress-bar-container">
                <div
                  className="progress-bar"
                  style={{ width: `${barWidth}%` }}
                ></div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="results-chart">
        <h3>Category Distribution</h3>
        <div className="chart-container">
          {categories.map(([category, count]) => (
            <div key={category} className="chart-bar">
              <div className="bar-container">
                <div
                  className="bar"
                  style={{
                    height: `${(count / maxCount) * 200}px`,
                  }}
                  title={`${category}: ${count}`}
                ></div>
              </div>
              <div className="bar-label">{category}</div>
              <div className="bar-value">{count}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default Results
