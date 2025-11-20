import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os

# Укажите пути к модели и видео
MODEL_PATH = "best_float32.tflite"
VIDEO_PATH = "input_video2.mp4"

# Загрузка модели TensorFlow Lite
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    print("✅ Модель TensorFlow Lite успешно загружена!")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    exit(1)

# Выделение памяти для тензоров
interpreter.allocate_tensors()

# Получение информации о входном и выходном тензорах
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Размеры модели
model_height = input_details[0]['shape'][1]
model_width = input_details[0]['shape'][2]
print(f"Размер модели: {model_width}x{model_height}")

# Проверка типа входных данных
input_dtype = input_details[0]['dtype']
print(f"Тип входных данных: {input_dtype}")

# Открытие видеофайла
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Не удалось открыть видео: {VIDEO_PATH}")
    exit(1)

# Получение параметров исходного видео
fps_input = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Исходное видео: {frame_width}x{frame_height}, FPS: {fps_input}, Всего кадров: {total_frames}")

# Функция для вычисления центра bounding box
def get_center(bbox, image_size):
    x, y, w, h = bbox
    orig_w, orig_h = image_size
    
    # Преобразуем относительные координаты в абсолютные
    x_abs = x * orig_w
    y_abs = y * orig_h
    w_abs = w * orig_w
    h_abs = h * orig_h
    
    # Вычисляем центр
    center_x = x_abs + w_abs / 2
    center_y = y_abs + h_abs / 2
    
    return (center_x, center_y)

# Функция для вычисления расстояния между центрами
def distance_between_centers(center1, center2):
    return np.sqrt((center1[0] - center2[0])2 + (center1[1] - center2[1])2)

# Функция NMS по центрам bounding boxes
def center_based_nms(detections, image_size, distance_threshold=50):
    """
    Non-Maximum Suppression на основе расстояния между центрами bounding boxes
    distance_threshold: максимальное расстояние в пикселях для слияния детекций
    """
    if not detections:
        return []
    
    # Группируем детекции по классам
    classes_dict = {}
    for detection in detections:
        class_id = detection['class_id']
        if class_id not in classes_dict:
            classes_dict[class_id] = []
        classes_dict[class_id].append(detection)
    
    # Применяем NMS для каждого класса отдельно
    filtered_detections = []
    
    for class_id, class_detections in classes_dict.items():
        # Сортируем по уверенности (от высокой к низкой)
        class_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Вычисляем центры для всех детекций этого класса
        centers = [get_center(det['bbox'], image_size) for det in class_detections]
        
        # Флаги для отслеживания, какие детекции уже обработаны
        keep = [True] * len(class_detections)
        
        for i in range(len(class_detections)):
            if not keep[i]:
                continue
                
            for j in range(i + 1, len(class_detections)):
                if not keep[j]:
                    continue
                
                # Вычисляем расстояние между центрами
                dist = distance_between_centers(centers[i], centers[j])
                
                # Если центры слишком близки, оставляем только детекцию с большей уверенностью
                if dist < distance_threshold:
                    keep[j] = False
        
        # Добавляем только те детекции, которые прошли фильтрацию
        for i, detection in enumerate(class_detections):
            if keep[i]:
                filtered_detections.append(detection)
    
    return filtered_detections
