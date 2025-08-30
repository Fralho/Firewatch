import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Загрузка модели
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Функция для предобработки изображения
def preprocess_image(image_path, input_size=(640, 640)):
    image = Image.open(image_path).convert("RGB")  # убедимся, что RGB
    image = image.resize(input_size)
    image = np.array(image) / 255.0  # Нормализуем
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Добавляем batch размер
    return image

# Функция для извлечения выходных данных из интерпретатора
def postprocess(output_data, image_shape, threshold=0.05):
    boxes, class_ids, confidences = [], [], []

    output = output_data[0]  # (1, 84, 8400)
    output = np.squeeze(output)  # → (84, 8400)
    output = output.T  # → (8400, 84)

    for detection in output:
        if len(detection) < 6:
            continue  # safety

        x_center, y_center, width, height = detection[0:4]
        objectness = detection[4]
        class_scores = detection[5:]

        class_id = np.argmax(class_scores)
        class_score = class_scores[class_id]
        confidence = objectness * class_score  # YOLO confidence formula

        if confidence > threshold:
            # Координаты bbox
            x_min = int((x_center - width / 2) * image_shape[1])
            y_min = int((y_center - height / 2) * image_shape[0])
            x_max = int((x_center + width / 2) * image_shape[1])
            y_max = int((y_center + height / 2) * image_shape[0])

            boxes.append([x_min, y_min, x_max, y_max])
            class_ids.append(class_id)
            confidences.append(float(confidence))

    return boxes, class_ids, confidences

# Функция для визуализации боксов
def draw_boxes(image_path, boxes, class_ids, confidences):
    image = cv2.imread(image_path)
    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        print(box)
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"Class {class_id}: {confidence:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Загрузка и предобработка изображения
image_path = "image_2025-08-30_23-53-06.png" # Вот тут путь к изображению с возгаранием
image = preprocess_image(image_path)

# Подготовка входных данных для модели
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Подача изображения в модель
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

# Получаем результаты
output_data = [interpreter.get_tensor(output_detail['index']) for output_detail in output_details]

# Обработка выходных данных
boxes, class_ids, confidences = postprocess(output_data, image.shape[1:3])

# Визуализация результатов
draw_boxes(image_path, boxes, class_ids, confidences)
