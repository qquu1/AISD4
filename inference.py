import cv2
from pathlib import Path
from ultralytics import YOLO
import os

# --- НАСТРОЙКИ ---
MODEL_PATH = 'runs/detect/runs/train/body_type_v1-4/weights/best.pt' 
IMAGE_PATH = 'test/images' # Путь к папке с тестовыми картинками (или одна картинка)
OUT_DIR = 'runs/predict'   # Куда сохранять результаты

# Цвета для классов (BGR формат): Hatchback (зеленый), Sedan (синий), Wagon (красный)
COLORS = {
    0: (0, 255, 0),   # Hatchback
    1: (255, 0, 0),   # Sedan
    2: (0, 0, 255)    # Wagon
}

def run_inference():
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Модель не найдена по пути {MODEL_PATH}. Дождись окончания обучения!")
        return

    model = YOLO(MODEL_PATH)
    out_path = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    # Если передан путь к папке, ищем все картинки в ней
    if os.path.isdir(IMAGE_PATH):
        images = list(Path(IMAGE_PATH).glob('*.jpg')) + list(Path(IMAGE_PATH).glob('*.png'))
    else:
        images = [Path(IMAGE_PATH)]

    print(f"Найдено картинок: {len(images)}")

    for img_path in images:
        # Делаем предсказание
        results = model.predict(source=str(img_path), conf=0.25, save=False, verbose=False)
        
        # Читаем картинку для рисования
        img = cv2.imread(str(img_path))
        
        # Рисуем рамки
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            color = COLORS.get(cls_id, (200, 200, 200))
            
            # Рисуем прямоугольник
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Подготавливаем текст (Класс + уверенность)
            label = f"{class_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Рисуем фон для текста
            cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w + 4, y1), color, -1)
            # Пишем текст
            cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Сохраняем результат
        save_path = out_path / f"pred_{img_path.name}"
        cv2.imwrite(str(save_path), img)
        print(f"Сохранено: {save_path}")

if __name__ == '__main__':
    run_inference()