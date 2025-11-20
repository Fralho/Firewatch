import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
from collections import deque

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
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

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

class TemporalDetectionFilter:
    """
    Класс для временной фильтрации ложных срабатываний
    Анализирует последовательные кадры для подтверждения детекций
    """
    
    def __init__(self, sequence_length=5, iou_threshold=0.3, class_names=None):
        """
        Инициализация фильтра
        
        Args:
            sequence_length: количество последовательных кадров для анализа
            iou_threshold: порог IoU для сопоставления bounding box между кадрами
            class_names: словарь с названиями классов
        """
        self.sequence_length = sequence_length
        self.iou_threshold = iou_threshold
        self.class_names = class_names or {0: "smoke", 1: "fire"}
        
        # История детекций для каждого класса
        self.detection_history = {
            0: deque(maxlen=sequence_length),  # smoke
            1: deque(maxlen=sequence_length)   # fire
        }
        
        # Статистика
        self.true_positives = 0
        self.false_positives = 0
        
    def calculate_iou(self, bbox1, bbox2):
        """Вычисление Intersection over Union для двух bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Вычисляем координаты прямоугольников
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2
        
        # Вычисляем площадь пересечения
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        intersection = inter_width * inter_height
        
        # Вычисляем площадь объединения
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def match_detections(self, current_detections, previous_detections):
        """
        Сопоставление детекций между кадрами на основе IoU
        """
        matches = []
        used_indices = set()
        
        for i, curr_det in enumerate(current_detections):
            best_match = None
            best_iou = 0
            
            for j, prev_det in enumerate(previous_detections):
                if j in used_indices:
                    continue
                
                # Сравниваем только детекции одного класса
                if curr_det['class_id'] == prev_det['class_id']:
                    iou = self.calculate_iou(curr_det['bbox'], prev_det['bbox'])
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_match = j
            
            if best_match is not None:
                matches.append((i, best_match))
                used_indices.add(best_match)
        
        return matches
    
    def update_history(self, detections):
        """
        Обновление истории детекций и фильтрация ложных срабатываний
        """
        current_frame_detections = {0: [], 1: []}
        confirmed_detections = []
        
        # Группируем текущие детекции по классам
        for det in detections:
            class_id = det['class_id']
            if class_id in current_frame_detections:
                current_frame_detections[class_id].append(det)
        
        # Анализируем каждый класс отдельно
        for class_id in [0, 1]:
            current_dets = current_frame_detections[class_id]
            
            if not current_dets:
                # Нет детекций в текущем кадре - добавляем пустой список в историю
                self.detection_history[class_id].append([])
                continue
            
            # Получаем предыдущие детекции этого класса
            previous_detections = []
            if len(self.detection_history[class_id]) > 0:
                for hist_frame in self.detection_history[class_id]:
                    if hist_frame:  # если в кадре были детекции
                        previous_detections.extend(hist_frame)
            
            if not previous_detections:
                # Первая детекция в последовательности
                self.detection_history[class_id].append(current_dets)
                continue
            
            # Сопоставляем детекции с предыдущими кадрами
            matches = self.match_detections(current_dets, previous_detections)
            
            # Подтвержденные детекции - те, которые имеют соответствия в предыдущих кадрах
            confirmed_class_detections = []
            for curr_idx, _ in matches:
                confirmed_class_detections.append(current_dets[curr_idx])
            
            # Обновляем историю
            self.detection_history[class_id].append(confirmed_class_detections)
            
            # Проверяем, является ли детекция истинной (есть в sequence_length кадрах подряд)
            if len(self.detection_history[class_id]) == self.sequence_length:
                sequence_complete = all(
                    len(frame_dets) > 0 
                    for frame_dets in list(self.detection_history[class_id])
                )
                
                if sequence_complete:
                    # Это истинная детекция
                    self.true_positives += 1
                    confirmed_detections.extend(confirmed_class_detections)
                    print(f"✅ Подтверждена детекция {self.class_names[class_id]} (последовательность из {self.sequence_length} кадров)")
                else:
                    # Ложное срабатывание
                    self.false_positives += 1
                    print(f"❌ Ложное срабатывание {self.class_names[class_id]} (прерванная последовательность)")
            else:
                # Пока недостаточно кадров для окончательного решения
                confirmed_detections.extend(confirmed_class_detections)
        
        return confirmed_detections
    
    def get_stats(self):
        """Получение статистики фильтрации"""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "total_verified": self.true_positives + self.false_positives
        }

# Инициализация временного фильтра
temporal_filter = TemporalDetectionFilter(sequence_length=5)

# Основной цикл обработки видео
frame_count = 0
confirmed_detections_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    print(f"\n--- Обработка кадра {frame_count} ---")
    
    # Здесь должен быть ваш код для получения детекций от модели
    # Для примера, создадим mock детекции
    # Замените этот блок на реальную инференс-логику
    
    # Mock детекции (замените на реальные вызовы модели)
    mock_detections = [
        # Пример детекции дыма
        {
            'class_id': 0,
            'confidence': 0.85,
            'bbox': [0.4, 0.3, 0.1, 0.1]  # [x, y, w, h] в относительных координатах
        },
        # Пример детекции огня
        {
            'class_id': 1, 
            'confidence': 0.78,
            'bbox': [0.6, 0.5, 0.08, 0.12]
        }
    ]
    
    # Применяем NMS
    filtered_detections = center_based_nms(
        mock_detections, 
        (frame_width, frame_height)
    )
    
    # Применяем временную фильтрацию
    confirmed_detections = temporal_filter.update_history(filtered_detections)
    
    if confirmed_detections:
        confirmed_detections_count += len(confirmed_detections)
        print(f"Подтвержденные детекции в кадре {frame_count}: {len(confirmed_detections)}")
    
    # Визуализация результатов (опционально)
    for det in confirmed_detections:
        class_name = temporal_filter.class_names[det['class_id']]
        confidence = det['confidence']
        bbox = det['bbox']
        
        # Конвертируем относительные координаты в абсолютные
        x_abs = int(bbox[0] * frame_width)
        y_abs = int(bbox[1] * frame_height)
        w_abs = int(bbox[2] * frame_width)
        h_abs = int(bbox[3] * frame_height)
        
        # Рисуем bounding box
        color = (0, 255, 0) if det['class_id'] == 0 else (0, 0, 255)  # зеленый для дыма, красный для огня
        cv2.rectangle(frame, (x_abs, y_abs), (x_abs + w_abs, y_abs + h_abs), color, 2)
        cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                   (x_abs, y_abs - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Показываем кадр (опционально)
    cv2.imshow('Fire/Smoke Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Выводим статистику
stats = temporal_filter.get_stats()
print(f"\n=== СТАТИСТИКА ФИЛЬТРАЦИИ ===")
print(f"Истинные срабатывания: {stats['true_positives']}")
print(f"Ложные срабатывания: {stats['false_positives']}")
print(f"Всего проверено последовательностей: {stats['total_verified']}")
print(f"Всего подтвержденных детекций: {confirmed_detections_count}")

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
