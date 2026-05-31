import { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);

    // create image preview URL
    setPreview(URL.createObjectURL(selectedFile));
  };

  const uploadImage = async () => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await axios.post("http://127.0.0.1:5000/predict", formData);

    setResult(
      res.data.prediction +
        " (confidence: " +
        res.data.confidence +
        ")"
    );
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>AI Image Classifier</h1>

      <input type="file" onChange={handleFileChange} />

      <br /><br />

      {/* IMAGE PREVIEW */}
      {preview && (
        <img
          src={preview}
          alt="preview"
          style={{ width: "250px", borderRadius: "10px" }}
        />
      )}

      <br /><br />

      <button onClick={uploadImage} disabled={!file}>
        Predict
      </button>

      <h2>{result}</h2>
    </div>
  );
}

export default App;