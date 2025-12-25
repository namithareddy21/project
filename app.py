from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

model = YOLO("yolov8n.pt")

object_descriptions = {
    "person": "A human being detected in front of the camera",
    "car": "A road vehicle used for transportation",
    "bottle": "A container typically used to hold liquids",
    "chair": "Furniture designed for sitting",
    "laptop": "A portable computing device",
    "cell phone": "A handheld communication device",
    "cup": "A small container used for drinking",
    "dog": "A domesticated animal commonly kept as a pet"
}

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("frame")
def handle_frame(data):
    img_data = base64.b64decode(data.split(",")[1])
    np_img = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls)]
        w = x2 - x1
        h = y2 - y1

        detections.append({
            "label": label,
            "x": x1,
            "y": y1,
            "w": w,
            "h": h,
            "description": object_descriptions.get(
                label, "A commonly detected real-world object"
            )
        })

    socketio.emit("detections", {
        "count": len(detections),
        "objects": detections
    })

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=10000)
