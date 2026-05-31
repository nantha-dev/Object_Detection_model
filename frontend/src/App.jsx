// App.jsx
import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:5000';

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filterConfidence, setFilterConfidence] = useState(0);
  const [sortBy, setSortBy] = useState('confidence'); // 'confidence', 'label'

  // Cleanup preview URL
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedImage(file);
      const previewUrl = URL.createObjectURL(file);
      setPreview(previewUrl);
      setResult(null);
      setError(null);
      setFilterConfidence(0);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.webp'] },
    maxFiles: 1,
    multiple: false,
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
        timeout: 60000,
      });

      if (response.data.success) {
        setResult(response.data);
      } else {
        setError(response.data.error || 'Failed to process image');
      }
    } catch (err) {
      setError(err.message || 'Network error. Make sure the backend is running.');
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
      link.download = 'detection_result.jpg';
      link.click();
    }
  };

  // Filter and sort detections
  const getFilteredDetections = () => {
    if (!result) return [];
    let detections = [...result.detections];
    if (filterConfidence > 0) {
      detections = detections.filter(d => d.confidence >= filterConfidence);
    }
    if (sortBy === 'confidence') {
      detections.sort((a, b) => b.confidence - a.confidence);
    } else if (sortBy === 'label') {
      detections.sort((a, b) => a.label.localeCompare(b.label));
    }
    return detections;
  };

  const filteredDetections = getFilteredDetections();
  const highConfCount = result?.detections?.filter(d => d.confidence >= 80).length || 0;
  const uniqueColors = result?.detections?.filter(d => d.color && d.color !== 'Unknown').map(d => d.color).filter((v, i, a) => a.indexOf(v) === i).length || 0;

  return (
    <div className="app">
      <div className="bg-gradient"></div>
      
      <header className="header">
        <div className="container">
          <h1>🔍 VisionAI</h1>
          <p>Advanced Object Detection | Segmentation | Face Analysis</p>
        </div>
      </header>

      <main className="main container">
        {!preview && !result && (
          <section className="upload-section">
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
              <input {...getInputProps()} />
              <div className="dropzone-icon"></div>
              <p>{isDragActive ? 'Drop your image here' : 'Drag & drop or click to upload'}</p>
              <span className="dropzone-hint">Supports JPG, PNG, BMP, WebP</span>
            </div>
          </section>
        )}

        {(preview || result) && (
          <div className="content-wrapper">
            {/* Image comparison grid */}
            <div className="image-grid">
              <div className="image-card">
                <div className="card-header">
                  <h3> Original</h3>
                  {!result && (
                    <button onClick={clearImage} className="btn-icon" aria-label="Clear">
                      ✕
                    </button>
                  )}
                </div>
                <div className="image-container">
                  <img src={preview} alt="Original" />
                </div>
              </div>

              {result && (
                <div className="image-card">
                  <div className="card-header">
                    <h3> Detection Result</h3>
                    <div className="card-actions">
                      <button onClick={downloadImage} className="btn-download">⬇ Download</button>
                      <button onClick={clearImage} className="btn-new"> New Image</button>
                    </div>
                  </div>
                  <div className="image-container">
                    <img src={`data:image/jpeg;base64,${result.output_image}`} alt="Detection Result" />
                  </div>
                </div>
              )}
            </div>

            {/* Action buttons when only preview exists */}
            {preview && !result && !loading && (
              <div className="action-bar">
                <button onClick={clearImage} className="btn-secondary">Cancel</button>
                <button onClick={handleUpload} className="btn-primary"> Detect Objects</button>
              </div>
            )}

            {/* Detection results */}
            {result && (
              <div className="results-panel">
                <div className="stats-row">
                  <div className="stat-card">
                    <div className="stat-value">{result.total_detections}</div>
                    <div className="stat-label">Total Objects</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{highConfCount}</div>
                    <div className="stat-label">High Confidence ≥80%</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{uniqueColors}</div>
                    <div className="stat-label">Unique Colors</div>
                  </div>
                </div>

                <div className="detection-header">
                  <h3> Detection Details</h3>
                  <div className="filter-controls">
                    <select 
                      value={filterConfidence} 
                      onChange={(e) => setFilterConfidence(Number(e.target.value))}
                      className="filter-select"
                    >
                      <option value={0}>All confidences</option>
                      <option value={50}>≥50%</option>
                      <option value={70}>≥70%</option>
                      <option value={80}>≥80%</option>
                      <option value={90}>≥90%</option>
                    </select>
                    <select 
                      value={sortBy} 
                      onChange={(e) => setSortBy(e.target.value)}
                      className="filter-select"
                    >
                      <option value="confidence">Sort by Confidence ↓</option>
                      <option value="label">Sort by Object Name</option>
                    </select>
                  </div>
                </div>

                <div className="detection-list">
                  {filteredDetections.map((detection, idx) => (
                    <div key={idx} className="detection-item">
                      <div className="detection-header-row">
                        <div className="detection-title">
                          <span className="object-icon">
                            {detection.label === 'person' ? '' : 
                             detection.label === 'car' ? '' :
                             detection.label === 'dog' ? '' :
                             detection.label === 'cat' ? '' : ''}
                          </span>
                          <span className="object-name">{detection.label}</span>
                          {detection.type && <span className="object-type">({detection.type})</span>}
                        </div>
                        <div className="detection-confidence">
                          <div className="confidence-bar">
                            <div className="confidence-fill" style={{ width: `${detection.confidence}%` }}></div>
                          </div>
                          <span className="confidence-text">{detection.confidence}%</span>
                        </div>
                      </div>

                      <div className="detection-details">
                        {detection.category && detection.category !== 'General' && (
                          <div className="detail-chip">
                            <span className="detail-label">Category</span>
                            <span>{detection.category}</span>
                          </div>
                        )}
                        {detection.color && detection.color !== 'Unknown' && (
                          <div className="detail-chip color-chip">
                            <span className="detail-label">Color</span>
                            <span className="color-value">
                              <span className="color-dot" style={{ backgroundColor: detection.color.toLowerCase() }}></span>
                              {detection.color}
                            </span>
                          </div>
                        )}
                        {detection.label === 'person' && (
                          <>
                            {detection.gender && detection.gender !== 'Unknown' && (
                              <div className="detail-chip">
                                <span className="detail-label">Gender</span>
                                <span>{detection.gender}</span>
                              </div>
                            )}
                            {detection.age > 0 && (
                              <div className="detail-chip">
                                <span className="detail-label">Age</span>
                                <span>{detection.age} years</span>
                              </div>
                            )}
                            {detection.recognized_person && detection.recognized_person !== 'Unknown' && (
                              <div className="detail-chip recognized">
                                <span className="detail-label">Recognized as</span>
                                <span>✨ {detection.recognized_person}</span>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                  {filteredDetections.length === 0 && (
                    <div className="empty-state">No detections match the current filter.</div>
                  )}
                </div>
              </div>
            )}

            {/* Loading overlay */}
            {loading && (
              <div className="loading-overlay">
                <div className="loading-spinner"></div>
                <p>Analyzing image with YOLOv8 & AI models...</p>
              </div>
            )}

            {/* Error message */}
            {error && (
              <div className="error-message">
                <span> {error}</span>
                <button onClick={clearImage} className="btn-retry">Try Again</button>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Powered by YOLOv8 | Segmentation Masks | Color Detection | Face Recognition</p>
      </footer>
    </div>
  );
};

export default App;