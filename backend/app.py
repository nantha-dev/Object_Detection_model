from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from deepface import DeepFace
import cv2
import os
import uuid
import numpy as np
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
KNOWN_FACES_FOLDER = "known_faces"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(KNOWN_FACES_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n-seg.pt")

# Hierarchy definitions
hierarchy = {
    'dog': {'category': 'Animal', 'type': 'Dog'},
    'cat': {'category': 'Animal', 'type': 'Cat'},
    'banana': {'category': 'Fruit', 'type': 'Banana'},
    'apple': {'category': 'Fruit', 'type': 'Apple'},
    'orange': {'category': 'Fruit', 'type': 'Orange'},
    'person': {'category': 'Human', 'type': 'Person'},
    'car': {'category': 'Vehicle', 'type': 'Car'},
    'bicycle': {'category': 'Vehicle', 'type': 'Bicycle'},
    'motorbike': {'category': 'Vehicle', 'type': 'Motorcycle'},
    'bus': {'category': 'Vehicle', 'type': 'Bus'},
    'truck': {'category': 'Vehicle', 'type': 'Truck'},
    'chair': {'category': 'Furniture', 'type': 'Chair'},
    'sofa': {'category': 'Furniture', 'type': 'Sofa'},
    'table': {'category': 'Furniture', 'type': 'Table'},
    'bottle': {'category': 'Container', 'type': 'Bottle'},
    'cup': {'category': 'Container', 'type': 'Cup'}
}

# Color detection
def get_color_name(bgr):
    b, g, r = bgr
    r, g, b = int(r), int(g), int(b)
    
    if r < 50 and g < 50 and b < 50:
        return "Black"
    if r > 200 and g > 200 and b > 200:
        return "White"
    if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
        return "Gray"
    
    colors = {
        "Red": r > 150 and g < 100 and b < 100,
        "Green": g > 150 and r < 100 and b < 100,
        "Blue": b > 150 and r < 100 and g < 100,
        "Yellow": r > 150 and g > 150 and b < 100,
        "Cyan": g > 150 and b > 150 and r < 100,
        "Magenta": r > 150 and b > 150 and g < 100,
        "Orange": r > 150 and g > 100 and g < 150 and b < 100,
        "Purple": r > 100 and b > 100 and r < 150 and g < 100,
        "Pink": r > 180 and g < 150 and b > 100,
        "Brown": r > 100 and g > 50 and g < 100 and b < 80
    }
    
    for color_name, condition in colors.items():
        if condition:
            return color_name
    return "Unknown"

def safe_convert(obj):
    if isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_convert(i) for i in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int64):
        return int(obj)
    return obj

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files["image"]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        filename = secure_filename(str(uuid.uuid4()) + ".jpg")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        annotated_image = image.copy()
        results = model(filepath)
        
        detections = []
        
        for r in results:
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
            else:
                masks = []
            
            for idx, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                confidence = float(box.conf[0])
                
                item = {
                    "label": label,
                    "confidence": round(confidence * 100, 2),
                    "bbox": [x1, y1, x2, y2]
                }
                
                if label in hierarchy:
                    item["category"] = hierarchy[label]["category"]
                    item["type"] = hierarchy[label]["type"]
                
                # Draw segmentation mask
                if idx < len(masks):
                    mask = masks[idx]
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8)
                    
                    colored_mask = np.zeros_like(image)
                    colored_mask[mask == 1] = (255, 0, 0)
                    
                    alpha = 0.3
                    mask_indices = mask == 1
                    annotated_image[mask_indices] = cv2.addWeighted(
                        annotated_image[mask_indices], 1 - alpha,
                        colored_mask[mask_indices], alpha, 0
                    )
                    
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_image, contours, -1, (255, 0, 0), 2)
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Color detection
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    avg_color = np.mean(crop, axis=(0, 1))
                    detected_color = get_color_name(avg_color)
                    item["color"] = detected_color
                
                # Person features
                if label == "person":
                    try:
                        face_roi = image[y1:y2, x1:x2]
                        if face_roi.size > 0:
                            face_path = os.path.join(UPLOAD_FOLDER, f"face_{uuid.uuid4()}.jpg")
                            cv2.imwrite(face_path, face_roi)
                            
                            try:
                                analysis = DeepFace.analyze(
                                    img_path=face_path,
                                    actions=["age", "gender"],
                                    enforce_detection=False,
                                    silent=True
                                )
                                
                                if analysis and len(analysis) > 0:
                                    if isinstance(analysis, list):
                                        analysis = analysis[0]
                                    
                                    item["age"] = int(analysis.get("age", 0))
                                    item["gender"] = analysis.get("gender", "Unknown")
                                    
                                    if isinstance(item["gender"], dict):
                                        gender_dict = {k: float(v) for k, v in item["gender"].items()}
                                        item["gender"] = max(gender_dict, key=gender_dict.get)
                            except:
                                item["age"] = 0
                                item["gender"] = "Unknown"
                            
                            item["recognized_person"] = "Unknown"
                            try:
                                if os.path.exists(KNOWN_FACES_FOLDER) and os.listdir(KNOWN_FACES_FOLDER):
                                    result_face = DeepFace.find(
                                        img_path=face_path,
                                        db_path=KNOWN_FACES_FOLDER,
                                        enforce_detection=False,
                                        silent=True
                                    )
                                    
                                    if result_face and len(result_face) > 0 and len(result_face[0]) > 0:
                                        df = result_face[0].sort_values(by="distance")
                                        best = df.iloc[0]
                                        name = os.path.basename(best["identity"]).split(".")[0]
                                        
                                        if best["distance"] < 0.40:
                                            item["recognized_person"] = name
                            except:
                                pass
                            
                            try:
                                os.remove(face_path)
                            except:
                                pass
                    except:
                        pass
                
                detections.append(item)
        
        # Save output image
        output_filename = f"output_{uuid.uuid4()}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, annotated_image)
        
        with open(output_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        try:
            os.remove(filepath)
            os.remove(output_path)
        except:
            pass
        
        response = {
            "success": True,
            "detections": safe_convert(detections),
            "total_detections": len(detections),
            "output_image": img_base64
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)