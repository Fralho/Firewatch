import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

tflite_model_path = "model.tflite"
video_path = "5_видео_БАС_торфяники_с_дымом_2_АРМ_11.mp4"  # Укажите путь к вашему видеофайлу

def pred_on_video(tflite_model_path, video_path, input_size=(640, 640), conf_threshold=0.02):
    # Загрузка модели
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)
    
    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Создаем окно для отображения
    cv2.namedWindow('Video Detection', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Выход из цикла если видео закончилось
        
        # Конвертируем кадр из BGR (OpenCV) в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        
        # Подготавливаем изображение для модели
        img_resized = img_pil.resize(input_size)
        input_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Проверяем порядок входных каналов (CHW или HWC)
        if input_details[0]['shape'][-1] == 3:  # HWC
            input_array = np.expand_dims(input_array, axis=0)
        else:  # CHW
            input_array = np.transpose(input_array, (2, 0, 1))
            input_array = np.expand_dims(input_array, axis=0)
        
        # Выполняем инференс
        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        
        # Получаем выходные данные
        output = interpreter.get_tensor(output_details[0]['index'])
        output = output[0].T  # Преобразуем в [num_detections, 6]
        print(output.shape)
        
        # Обрабатываем детекции

        # for det in output:
        #     if len(det) >= 6:
        #         x_center, y_center, width, height, conf, class_id = det[:6]
        #         if conf >= conf_threshold:
        #             x1 = int(x_center - width / 2)
        #             y1 = int(y_center - height / 2)
        #             x2 = int(x_center + width / 2)
        #             y2 = int(y_center + height / 2)
        #             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        for det in output:
            # print(len(det))
            continue
            x1, y1, x2, y2, conf, class_id = det
            
            if conf >= conf_threshold:
                # Масштабируем координаты под исходный кадр
                x1 = int(x1 * width / input_size[0])
                y1 = int(y1 * height / input_size[1])
                x2 = int(x2 * width / input_size[0])
                y2 = int(y2 * height / input_size[1])
                
                # Рисуем bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Добавляем текст с уверенностью и классом
                label = f"Class {int(class_id)}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Отображаем кадр с детекциями
        cv2.imshow('Video Detection', frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

# Запускаем обработку видео
pred_on_video(tflite_model_path, video_path)