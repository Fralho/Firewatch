import numpy as np
import tensorflow as tf
import cv2

# Пути к файлам
tflite_model_path = 'C:/Users/onofr/Downloads/model.tflite'
image_path = 'dataset/images/test/1.jpg'                

# Загрузка и настройка интерпретатора TFLite
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Определение размера входного тензора
input_shape = input_details[0]['shape']
input_height = input_shape[2]  # Высота (640)
input_width = input_shape[3]   # Ширина (640)
input_channels = input_shape[1]  # Каналы (3)
print(f"Expected input shape: {input_shape}")

# Чтение и подготовка изображения
img = cv2.imread(image_path)
if img is None:
    print("Ошибка: не удалось загрузить изображение 2.jpg. Проверь путь!")
else:
    # Изменение размера в соответствии с ожидаемым входом модели
    img_resized = cv2.resize(img, (input_width, input_height))  # (640, 640)
    input_array = np.array(img_resized, dtype=np.float32) / 255.0  # Нормализация

    # Перестановка каналов для соответствия [1, 3, 640, 640]
    input_array = np.transpose(input_array, (2, 0, 1))  # Из [height, width, channels] в [channels, height, width]
    input_array = np.expand_dims(input_array, axis=0)    # Добавляем батч [1, channels, height, width]

    # Проверка формы входного тензора
    print(f"Input array shape: {input_array.shape}")

    # Установка входного тензора
    interpreter.set_tensor(input_details[0]['index'], input_array)

    # Выполнение предсказания
    interpreter.invoke()

    # Получение вывода и вывод формы
    output = interpreter.get_tensor(output_details[0]['index'])[0]  # [N, 6] или аналогично
    print(f"Output shape: {output.shape}")  # Добавлен вывод формы вывода

    # Обработка детекций
    for det in output:
        if len(det) >= 6:  # Проверяем минимальную длину для [x_center, y_center, width, height, conf, class_id]
            x_center, y_center, width, height, conf, class_id = det[:6]
            if conf >= 0.6:  # Порог уверенности
                # Преобразование нормализованных координат в пиксельные
                x1 = int((x_center - width / 2) * input_width)
                y1 = int((y_center - height / 2) * input_height)
                x2 = int((x_center + width / 2) * input_width)
                y2 = int((y_center + height / 2) * input_height)

                # Ограничение координат рамки в пределах изображения
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(input_width - 1, x2)
                y2 = min(input_height - 1, y2)

                # Рисование bounding box и текста (только индекс класса и уверенность)
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_resized, f'Class {int(class_id)} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отображение результата
    cv2.imshow('Detection Result', img_resized)
    cv2.waitKey(0)  # Ждем нажатия любой клавиши для закрытия окна
    cv2.destroyAllWindows()