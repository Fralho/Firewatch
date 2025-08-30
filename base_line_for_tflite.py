import numpy as np
from PIL import Image
import tensorflow as tf

import cv2


# Пример использования
tflite_model_path = "C:/Users/onofr/Downloads/model.tflite"

numberFile = '1'
image_path = f'dataset/images/test/{numberFile}.jpg'

def pred(tflite_model_path, image_path, input_size=(640, 640), conf_threshold=0.6):
    # Загрузка модели
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Загрузка и подготовка изображения
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(input_size)
    input_array = np.array(img_resized, dtype=np.float32) / 255.0
    if input_details[0]['shape'][1] == 3:
        input_array = np.transpose(input_array, (2, 0, 1))
    input_array = np.expand_dims(input_array, 0)

    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]  # [6, N] → [N,6]
    output = output.T  # теперь shape [num_boxes,6]


    img = cv2.imread(image_path)


    for det in output:
        x1, y1, x2, y2, conf, class_id = det
        if conf >= conf_threshold:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)

    cv2.imshow('fsdf', img)
    while True:
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()



pred(tflite_model_path, image_path)
