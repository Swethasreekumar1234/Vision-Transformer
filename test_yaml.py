from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(data='data.yaml', epochs=30, imgsz=1024, batch=4,patience=10,optimizer='AdamW')
