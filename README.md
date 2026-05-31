# AI Image Classifier (Cat vs Dog)

A full-stack deep learning project that classifies images of cats and dogs using a TensorFlow CNN model with a Flask backend and React frontend.

##  Features
- Upload image and get prediction (Cat / Dog)
- Shows uploaded image preview
- Displays prediction confidence
- REST API using Flask
- CNN model trained on Kaggle dataset

## Tech Stack
Frontend: React.js, Axios  
Backend: Flask, TensorFlow, NumPy, Pillow  
ML: CNN (Keras)

## API
POST /predict  
Form-data: file=image

Response:
{
  "prediction": "Dog",
  "confidence": 0.93
}

##  Run Backend
pip install flask flask-cors tensorflow numpy pillow
python app.py

##  Run Frontend
npm install
npm start
