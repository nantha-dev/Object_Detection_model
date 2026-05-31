import React, { useState } from 'react';
import './App.css';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';

const API_URL = 'http://localhost:5000';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedImage(file);
      const previewUrl = URL.createObjectURL(file);
      setPreview(previewUrl);
      setResult(null);
      setError(null);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.webp']
    },
    maxFiles: 1,
    multiple: false
  });

  const handleUpload = async () => {
    if (!selectedImage) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000
      });

      if (response.data.success) {
        setResult(response.data);
      } else {
        setError(response.data.error || 'Failed to process');
      }
    } catch (err) {
      setError(err.message || 'Failed to process image');
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    if (preview) URL.revokeObjectURL(preview);
    setSelectedImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const downloadImage = () => {
    if (result?.output_image) {
      const link = document.createElement('a');
      link.href = `data:image/jpeg;base64,${result.output_image}`;
      link.download = 'detected_image.jpg';
      link.click();
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Object Detection System</h1>
        <p>YOLOv8 Segmentation | Color Detection | Face Recognition</p>
      </header>

      <main className="main">
        {!result && !preview && (
          <div className="upload-area">
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
              <input {...getInputProps()} />
              <div className="upload-icon">📸</div>
              <p>{isDragActive ? 'Drop image here' : 'Drag & drop or click to upload'}</p>
              <small>JPG, PNG, BMP, WebP</small>
            </div>
          </div>
        )}

        {(preview || result) && (
          <div className="content">
            <div className="image-grid">
              <div className="image-card">
                <div className="card-header">
                  <h3>Original</h3>
                  {!result && (
                    <button onClick={clearImage} className="clear-btn">Clear</button>
                  )}
                </div>
                <img src={preview} alt="Original" />
              </div>

              {result && (
                <div className="image-card">
                  <div className="card-header">
                    <h3>Detection Result</h3>
                    <div className="card-actions">
                      <button onClick={downloadImage} className="download-btn">Download</button>
                      <button onClick={clearImage} className="new-btn">New</button>
                    </div>
                  </div>
                  <img src={`data:image/jpeg;base64,${result.output_image}`} alt="Result" />
                </div>
              )}
            </div>

            {result && (
              <div className="results-card">
                <div className="results-header">
                  <h3>Detection Results</h3>
                  <span className="total-badge">{result.total_detections} objects</span>
                </div>
                
                <div className="stats-grid">
                  <div className="stat">
                    <div className="stat-value">{result.total_detections}</div>
                    <div className="stat-label">Total Objects</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">
                      {result.detections.filter(d => d.confidence >= 80).length}
                    </div>
                    <div className="stat-label">High Confidence</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">
                      {[...new Set(result.detections.filter(d => d.color).map(d => d.color))].length}
                    </div>
                    <div className="stat-label">Colors</div>
                  </div>
                </div>

                <div className="table-container">
                  <table className="detection-table">
                    <thead>
                      <tr>
                        <th>Object</th>
                        <th>Confidence</th>
                        <th>Category</th>
                        <th>Color</th>
                        <th>Details</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.detections.map((detection, idx) => (
                        <tr key={idx}>
                          <td className="object-name">
                            {detection.label}
                            {detection.type && <small>({detection.type})</small>}
                          </td>
                          <td>
                            <div className="confidence">
                              
                              <span>{detection.confidence}%</span>
                            </div>
                          </td>
                          <td>{detection.category || 'General'}</td>
                          <td>
                            {detection.color && detection.color !== 'Unknown' ? (
                              <div className="color-chip">
                                <span className="color-dot" style={{ backgroundColor: detection.color.toLowerCase() }}></span>
                                {detection.color}
                              </div>
                            ) : '—'}
                          </td>
                          <td>
                            {detection.label === 'person' ? (
                              <div className="person-details">
                                {detection.gender && detection.gender !== 'Unknown' && (
                                  <h3>{detection.gender}</h3>
                                )}
                                {detection.age > 0 && (
                                  <h3>{detection.age}y</h3>
                                )}
                                {detection.recognized_person && detection.recognized_person !== 'Unknown' && (
                                  <h3 className="recognized"> {detection.recognized_person}</h3>
                                )}
                              </div>
                            ) : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {loading && (
              <div className="loading-overlay">
                <div className="spinner"></div>
                <p>Processing image...</p>
              </div>
            )}

            {error && (
              <div className="error-message">
                <span>❌ {error}</span>
                <button onClick={clearImage}>Try Again</button>
              </div>
            )}
          </div>
        )}

        {preview && !result && !loading && (
          <div className="action-buttons">
            <button onClick={clearImage} className="secondary">Cancel</button>
            <button onClick={handleUpload} className="primary">Detect Objects</button>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>YOLOv8 | Segmentation Masks | Color Detection | Face Analysis</p>
      </footer>
    </div>
  );
}

export default App;