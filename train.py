from ultralytics import YOLO
import os

# Обязательно для Windows!
if __name__ == '__main__':
    # 1. Загружаем базовую модель YOLOv8s
    model = YOLO('yolov8s.pt')

    # 2. Запускаем обучение
    results = model.train(
        data='data.yaml',      
        epochs=50,             
        imgsz=640,             
        batch=8,              
        device=0,              # У тебя же видеокарта работает? Если да, оставляй 0.
        project='runs/train',  
        name='body_type_v1',   
        patience=15,           
        plots=True             
    )