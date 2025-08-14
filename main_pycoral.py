import time
import cv2
import numpy as np
from picamera2 import Picamera2, Preview
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
from pycoral.adapters import visualize

# Параметры
MODEL_PATH = "model_edgetpu.tflite"   # путь к модели
LABELS_PATH = "labels.txt"            # файл с метками
INPUT_SIZE = (1080, 1080)              
SCORE_THRESHOLD = 0.4
NMS_IOU = 0.45

# Загрузить метки
def load_labels(path):
    labels = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                labels[i] = line.strip()
    except Exception:
        pass
    return labels

labels = load_labels(LABELS_PATH)

# Инициализация ускорителя Edge Coral
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

# Picamera2 init (MIPI)
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration({"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

def preprocess_frame(frame, target_size):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, target_size)
    # Если модель ожидает RGB:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Если модель ожидает uint8 [0..255]
    return img

def run_inference(image):
    common.set_input(interpreter, image)
    interpreter.invoke()
    
    objs = detect.get_objects(interpreter, SCORE_THRESHOLD, top_k=50)
    return objs

def draw_results(frame, objs, labels):
    for obj in objs:
        bbox = obj.bbox
        x0, y0, x1, y1 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0,255,0), 2)
        cls = int(obj.id)
        score = obj.score
        label = labels.get(cls, str(cls))
        cv2.putText(frame, f"{label} {score:.2f}", (int(x0), int(y0)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame

def main_loop():
    try:
        while True:
            frame = picam2.capture_array()  # BGR uint8
            inp = preprocess_frame(frame, INPUT_SIZE)
            objs = run_inference(inp)
            out_frame = draw_results(frame.copy(), objs, labels)
       
            cv2.imshow("EdgeTPU YOLO", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
