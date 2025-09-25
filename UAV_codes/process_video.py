import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from PIL import Image
import os, time

# Укажите пути к модели и видео
MODEL_PATH = "best_full_integer_quant_edgetpu.tflite"
VIDEO_PATH = "input_video.mp4"  # Ваше видео
OUTPUT_PATH = "output_video.mp4"

try:
    # Создание интерпретатора с делегатом Edge TPU
    interpreter = tflite.Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[
            tflite.load_delegate('libedgetpu.so.1')
        ]
    )
    print("Делегат Edge TPU успешно загружен!")
except Exception as e:
    print(f"Ошибка загрузки делегата: {e}")
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

# Открытие видеофайла
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Не удалось открыть видео: {VIDEO_PATH}")
    exit(1)

# Получение параметров исходного видео
fps_input = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Исходное видео: {frame_width}x{frame_height}, FPS: {fps_input}, Всего кадров: {total_frames}")

fps_output = 24
print(f"Выходное видео будет с FPS: {fps_output}")

# Вычисляем интервал пропуска кадров
skip_interval = max(1, int(fps_input / fps_output))
print(f"Пропуск кадров: каждый {skip_interval}-й кадр")

# Создание VideoWriter для сохранения результата с FPS = 5
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_output, (frame_width, frame_height))

# Функция обработки вывода YOLO
def process_yolo_output(outputs, confidence_threshold=0.5):
    detections = []
    predictions = np.squeeze(outputs[0]).T
    
    for i, prediction in enumerate(predictions):
        # Извлекаем координаты bounding box (относительные координаты)
        bbox = prediction[:4]
        
        # Извлекаем уверенности классов
        class_confidences = prediction[4:]
        class_id = np.argmax(class_confidences)
        confidence = class_confidences[class_id]
        
        if confidence > confidence_threshold:
            detections.append({
                'class_id': class_id,
                'confidence': confidence,
                'bbox': bbox  # x, y, w, h в относительных координатах
            })
    
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections

# Функция для преобразования координат из размера модели в исходный размер
def scale_coordinates(bbox, model_size, original_size):
    x, y, w, h = bbox
    model_w, model_h = model_size
    orig_w, orig_h = original_size
    
    # Преобразуем относительные координаты в абсолютные (пиксели)
    x_abs = int(x * orig_w)
    y_abs = int(y * orig_h)
    w_abs = int(w * orig_w)
    h_abs = int(h * orig_h)
    
    # Рассчитываем координаты bounding box
    x1 = int(x_abs - w_abs / 2)
    y1 = int(y_abs - h_abs / 2)
    x2 = int(x_abs + w_abs / 2)
    y2 = int(y_abs + h_abs / 2)
    
    # Ограничиваем координаты рамками изображения
    x1 = max(0, min(x1, orig_w - 1))
    y1 = max(0, min(y1, orig_h - 1))
    x2 = max(0, min(x2, orig_w - 1))
    y2 = max(0, min(y2, orig_h - 1))
    
    return x1, y1, x2, y2

# Цвета для разных классов
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128)
]

# Обработка видео по кадрам с пониженным FPS
frame_count = 0
processed_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % skip_interval != 0:
        continue
    start_time=time.time()
    processed_count += 1
    if processed_count % 10 == 0:
        print(f"Обработано кадров: {processed_count} (пропущено {frame_count - processed_count})")
    
    # Сохраняем копию исходного кадра для отрисовки
    output_frame = frame.copy()
    
    # Преобразуем BGR (OpenCV) в RGB (для модели)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Изменяем размер до размера модели
    img_resized = cv2.resize(frame_rgb, (model_width, model_height))
    
    # Конвертируем в int8 и добавляем размерность батча
    img_input = img_resized.astype(np.int8)
    img_input = np.expand_dims(img_input, axis=0)
    
    # Установка входных данных и выполнение инференса
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    
    # Получение результатов
    outputs = []
    for i in range(len(output_details)):
        output_data = interpreter.get_tensor(output_details[i]['index'])
        outputs.append(output_data)
    
    # Обработка детекций
    detections = process_yolo_output(outputs, confidence_threshold=0.5)
    
    # Отрисовка bounding boxes на исходном кадре
    for detection in detections[:10]:  # Ограничиваем количество боксов для производительности
        class_id = detection['class_id']
        confidence = detection['confidence']
        bbox = detection['bbox']
        
        # Преобразуем координаты к исходному размеру
        x1, y1, x2, y2 = scale_coordinates(bbox, (model_width, model_height), (frame_width, frame_height))
        
        # Выбираем цвет для класса
        color = colors[class_id % len(colors)]
        
        # Рисуем bounding box
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        
        # Добавляем подпись с классом и уверенностью
        label = f"Class {class_id}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Рисуем подложку для текста
        cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 5), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Рисуем текст
        cv2.putText(output_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    print(time.time()-start_time)
    # Записываем кадр в выходное видео
    out.write(output_frame)

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Обработка завершена! Результат сохранен в: {OUTPUT_PATH}")
print(f"Всего прочитано кадров: {frame_count}")
print(f"Обработано кадров: {processed_count}")
print(f"Выходное видео имеет FPS: {fps_output}")