import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import matplotlib.pyplot as plt

# 1. Инициализация модели
model = YOLO("yolov8s.pt")

# 2. Обучение (автоматическая валидация на val после каждой эпохи)
results = model.train(
    data="recycling_data.yaml",
    epochs=5,
    batch=8,
    project='recycling_project', # имя папки с результатами обучения
    name='exp1',  
    val=True,             # валидация во время обучения
    verbose=True
)

# 3. Оценка модели на тестовом наборе (после окончания обучения)
print("Оценка модели на тестовом наборе (test)...")
test_metrics = model.val(data="recycling_data.yaml", split="test")

# Вывод основных метрик
print(f"mAP50:  {test_metrics.box.map50:.4f}")
print(f"mAP50-95: {test_metrics.box.map:.4f}")
print(f"Precision: {test_metrics.box.mp:.4f}")
print(f"Recall: {test_metrics.box.mr:.4f}")

test_img_path = r"C:\Users\Вадим\Desktop\DeepNeuro\Лабораторные2026\6_Image_detection\data\images\test"
results_pred = model.predict(source=test_img_path, conf=0.4, save=True)
for res in results_pred:
     res.show()
     