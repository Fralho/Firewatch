import cv2
import numpy as np
import glob
import os
import json
from datetime import datetime

class CameraCalibrator:
    def __init__(self, calibration_images_path='shots/', chessboard_size=(9, 6), square_size=0.025):
        """
        Инициализация калибратора камеры для Raspberry Pi без GUI
        """
        self.calibration_images_path = calibration_images_path
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Критерии для алгоритма поиска углов шахматной доски
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Подготовка объектных точек
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Массивы для хранения точек
        self.objpoints = []
        self.imgpoints = []
        
        # Параметры калибровки
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
        
    def find_chessboard_corners(self, image):
        """
        Поиск углов шахматной доски на изображении
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Поиск углов шахматной доски
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            # Уточнение позиции углов
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            return True, corners_refined
        return False, None
    
    def calibrate_from_images(self, image_size=None, save_corners=True):
        """
        Калибровка камеры на основе изображений в папке
        """
        # Получаем список изображений
        images = glob.glob(os.path.join(self.calibration_images_path, '*.jpg'))
        images.sort()
        
        if not images:
            print("Не найдено изображений для калибровки!")
            return False
        
        print(f"Найдено {len(images)} изображений для калибровки")
        
        successful_calibrations = 0
        
        for i, fname in enumerate(images):
            print(f"Обработка изображения {i+1}/{len(images)}: {os.path.basename(fname)}")
            
            img = cv2.imread(fname)
            if img is None:
                print(f"Не удалось загрузить изображение: {fname}")
                continue
            
            if image_size is None:
                image_size = (img.shape[1], img.shape[0])
            
            success, corners = self.find_chessboard_corners(img)
            
            if success:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
                successful_calibrations += 1
                print(f"  ✓ Углы шахматной доски найдены")
                
                if save_corners:
                    # Сохраняем изображение с отмеченными углами
                    img_with_corners = img.copy()
                    cv2.drawChessboardCorners(img_with_corners, self.chessboard_size, corners, success)
                    
                    corners_dir = os.path.join(self.calibration_images_path, 'corners_detected')
                    os.makedirs(corners_dir, exist_ok=True)
                    corners_filename = os.path.join(corners_dir, f'corners_{os.path.basename(fname)}')
                    cv2.imwrite(corners_filename, img_with_corners)
                    print(f"  ✓ Изображение с углами сохранено: {corners_filename}")
            else:
                print(f"  ✗ Углы шахматной доски не найдены")
        
        print(f"\nУспешно обработано изображений: {successful_calibrations}/{len(images)}")
        
        if successful_calibrations < 3:
            print("Для калибровки необходимо как минимум 3 изображения с обнаруженной шахматной доской!")
            return False
        
        # Калибровка камеры
        print("Выполняется калибровка камеры...")
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None)
        
        self.calibration_error = ret
        print(f"Средняя ошибка репроекции: {self.calibration_error:.3f} пикселей")
        
        return True
    
    def save_calibration(self, filename='camera_calibration.json'):
        """
        Сохранение параметров калибровки в JSON файл
        """
        if self.camera_matrix is None:
            print("Сначала выполните калибровку!")
            return False
        
        calibration_data = {
            'calibration_date': datetime.now().isoformat(),
            'image_count': len(self.objpoints),
            'chessboard_size': self.chessboard_size,
            'square_size': self.square_size,
            'calibration_error': float(self.calibration_error),
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.dist_coeffs.tolist(),
            'image_size': [int(self.camera_matrix[0, 2] * 2), int(self.camera_matrix[1, 2] * 2)]  # Примерный размер
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Параметры калибровки сохранены в {filename}")
        return True
    
    def load_calibration(self, filename='camera_calibration.json'):
        """
        Загрузка параметров калибровки из JSON файла
        """
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            self.camera_matrix = np.array(calibration_data['camera_matrix'])
            self.dist_coeffs = np.array(calibration_data['distortion_coefficients'])
            self.calibration_error = calibration_data['calibration_error']
            
            print(f"Параметры калибровки загружены из {filename}")
            print(f"Дата калибровки: {calibration_data['calibration_date']}")
            print(f"Ошибка репроекции: {self.calibration_error:.3f} пикселей")
            
            return True
        except Exception as e:
            print(f"Ошибка загрузки калибровки: {e}")
            return False
    
    def undistort_image(self, image):
        """
        Коррекция искажения на изображении
        """
        if self.camera_matrix is None:
            print("Сначала выполните калибровку!")
            return image
        
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        
        # Коррекция искажения
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        
        # Обрезка изображения
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def create_undistorted_examples(self, num_examples=3):
        """
        Создание примеров коррекции искажения без отображения
        """
        images = glob.glob(os.path.join(self.calibration_images_path, '*.jpg'))
        images.sort()
        
        if not images:
            print("Нет изображений для создания примеров")
            return
        
        examples_dir = os.path.join(self.calibration_images_path, 'undistorted_examples')
        os.makedirs(examples_dir, exist_ok=True)
        
        for i in range(min(num_examples, len(images))):
            original_image = cv2.imread(images[i])
            undistorted_image = self.undistort_image(original_image)
            
            # Сохраняем оригинал и исправленное изображение
            original_filename = os.path.join(examples_dir, f'original_{i+1}.jpg')
            undistorted_filename = os.path.join(examples_dir, f'undistorted_{i+1}.jpg')
            
            cv2.imwrite(original_filename, original_image)
            cv2.imwrite(undistorted_filename, undistorted_image)
            
            print(f"Создан пример {i+1}: {os.path.basename(original_filename)} -> {os.path.basename(undistorted_filename)}")
    
    def print_calibration_info(self):
        """
        Вывод информации о калибровке
        """
        if self.camera_matrix is None:
            print("Калибровка не выполнена!")
            return
        
        print("\n" + "="*50)
        print("ИНФОРМАЦИЯ О КАЛИБРОВКЕ КАМЕРЫ")
        print("="*50)
        print(f"Матрица камеры:")
        print(self.camera_matrix)
        print(f"\nКоэффициенты дисторсии:")
        print(self.dist_coeffs.ravel())
        print(f"\nОшибка репроекции: {self.calibration_error:.3f} пикселей")
        print(f"Количество использованных изображений: {len(self.objpoints)}")

def main():
    """
    Основная функция калибровки
    """
    # Создаем экземпляр калибратора
    calibrator = CameraCalibrator(
        calibration_images_path='shots/',
        chessboard_size=(7, 9),  # Размер шахматной доски
        square_size=0.02  # Размер квадрата в метрах
    )
    
    # Выполняем калибровку
    print("Начинаем калибровку камеры...")
    success = calibrator.calibrate_from_images(save_corners=True)
    
    if success:
        # Выводим информацию о калибровке
        calibrator.print_calibration_info()
        
        # Сохраняем параметры калибровки
        calibrator.save_calibration('camera_calibration.json')
        
        # Создаем примеры коррекции
        print("\nСоздание примеров коррекции искажения...")
        calibrator.create_undistorted_examples(3)
        
        print("\nКалибровка завершена успешно!")
        print("Результаты сохранены в:")
        print("- camera_calibration.json - параметры калибровки")
        print("- shots/corners_detected/ - изображения с обнаруженными углами")
        print("- shots/undistorted_examples/ - примеры коррекции искажения")
    else:
        print("Калибровка не удалась!")

if __name__ == "__main__":
    main()
