# from ultralytics import YOLO
# import torch

# def main():
#     # Detect device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # Load YOLOv8 nano pretrained model
#     model = YOLO("yolov8n.pt")

#     # Train model
#     results = model.train(
#         data="/app/workspace/projects/cars2/exports/yolo_20260217_070559/data.yaml",   # path to data.yaml
#         epochs=10,
#         imgsz=640,
#         batch=16,
#         device=device,
#         workers=4,
#         project="runs/train",
#         name="yolov8n_custom",
#         pretrained=True,
#         optimizer="auto",
#         patience=20
#     )

#     print("Training completed.")
#     print(results)

# if __name__ == "__main__":
#     main()

import cv2
import torch
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = YOLO("/app/runs/train/yolov8n_custom2/weights/best.pt")

video_path = "cr2.mp4"
cap = cv2.VideoCapture(video_path)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, device=device)
        annotated_frame = results[0].plot()

        _, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

@app.get("/")
def home():
    return {"message": "YOLOv8 Stream Running"}

@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

