# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import pytesseract
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)  # izinkan frontend akses

# Load model YOLO (pastikan modelnya ada)
model = YOLO('yolov8n.pt')  # contoh pakai model kecil default

@app.route('/detect', methods=['POST'])
def detect_serial():
    file = request.files['image']
    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    # YOLO detection
    results = model.predict(source=filepath)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2,conf]
    detected_serials = []

    image = cv2.imread(filepath)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cropped = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(cropped)
        detected_serials.append(text.strip())

    return jsonify({
        'serial_numbers': detected_serials
    })

if __name__ == '__main__':
    app.run(debug=True)
