import serial
import struct
import time
import threading

class Storm32Controller:
    def __init__(self, port="/dev/ttyAMA4", baudrate=9600):
        """
        Инициализация контроллера STorM32
        """
        self.uart = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        
        # Диапазоны значений для осей (в градусах)
        self.pitch_range = (-90, 90)
        self.roll_range = (-90, 90)
        self.yaw_range = (-180, 179)
        
        # Диапазон значений STorM32
        self.storm32_min = 700
        self.storm32_max = 2300
        self.storm32_center = 1500
        
        # Текущие углы
        self.current_pitch = 0
        self.current_roll = 0
        self.current_yaw = 0
        
        self.lock = threading.Lock()
        print(f"Подключение к {port} установлено")

    def calc_crc(self, data):
        """Вычисление CRC-16/X25"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0x8408
                else:
                    crc >>= 1
        return crc

    def degrees_to_storm32(self, degrees, axis_range):
        """Преобразование градусов в значения STorM32"""
        min_deg, max_deg = axis_range
        degrees = max(min(degrees, max_deg), min_deg)
        percentage = (degrees - min_deg) / (max_deg - min_deg)
        return int(self.storm32_min + percentage * (self.storm32_max - self.storm32_min))

    def set_angle(self, pitch, roll, yaw):
        """
        Установка углов в значениях STorM32
        """
        with self.lock:
            for axis, value in [('pitch', pitch), ('roll', roll), ('yaw', yaw)]:
                if value != 0 and (value < 700 or value > 2300):
                    raise ValueError(f"{axis} value must be between 700 and 2300 or 0, but got {value}")

            payload = struct.pack('<HHH', pitch, roll, yaw)
            start_byte = 0xFA
            length = 0x06
            command = 0x12
            
            crc_data = bytes([length, command]) + payload
            crc = self.calc_crc(crc_data)
            message = bytes([start_byte]) + crc_data + struct.pack('<H', crc)
            
            self.uart.write(message)
            return 0

    def set_angle_degrees(self, pitch_deg, roll_deg, yaw_deg):
        """Установка углов в градусах"""
        pitch_val = self.degrees_to_storm32(pitch_deg, self.pitch_range)
        roll_val = self.degrees_to_storm32(roll_deg, self.roll_range)
        yaw_val = self.degrees_to_storm32(yaw_deg, self.yaw_range)
        
        result = self.set_angle(pitch_val, roll_val, yaw_val)
        
        # Обновляем текущие углы
        self.current_pitch = pitch_deg
        self.current_roll = roll_deg
        self.current_yaw = yaw_deg
        
        return result

    def get_current_angles(self):
        """Получение текущих углов"""
        return self.current_pitch, self.current_roll, self.current_yaw

    def recenter(self):
        """Рецентрирование подвеса"""
        return self.set_angle_degrees(0, 0, 0)

    def close(self):
        """Закрытие соединения"""
        if self.uart.is_open:
            self.uart.close()
        print("Соединение с подвесом закрыто")


    def set_angle_advanced(self, pitch_deg, roll_deg, yaw_deg, flags=0x00):
        """
        Установка углов с использованием команды CMD_SETANGLE (#17)
        Поддерживает неограниченный режим через флаги
        """
        # Преобразуем углы в байты (float, 4 байта каждый)
        payload = struct.pack('<fff', pitch_deg, roll_deg, yaw_deg)
        self.current_pitch = pitch_deg
        self.current_roll = roll_deg
        self.current_yaw = yaw_deg
        
        # Добавляем флаги и тип (type byte всегда 0)
        payload += bytes([flags, 0x00])
        
        # Формируем сообщение для команды CMD_SETANGLE (#17)
        start_byte = 0xFA
        length = 0x0E  # 14 байт полезной нагрузки (3 float × 4 + 2 байта)
        command = 0x11  # CMD_SETANGLE
        
        crc_data = bytes([length, command]) + payload
        crc = self.calc_crc(crc_data)
        
        message = bytes([start_byte]) + crc_data + struct.pack('<H', crc)
        
        self.uart.write(message)


    # def set_yaw_360(self, yaw_deg):
    #     """
    #     Установка угла yaw с поддержкой полного вращения на 360° через неограниченный режим
    #     yaw_deg: угол yaw в градусах (от -180 до 179 или от 0 до 359)
    #     """
    #     # Нормализация угла в диапазон -180° до 179°
    #     normalized_yaw = yaw_deg
    #     while normalized_yaw > 180:
    #         normalized_yaw -= 360
    #     while normalized_yaw < -180:
    #         normalized_yaw += 360
            
    #     # Установка угла с использованием неограниченного режима (flags=0x00)
    #     self.set_angle_advanced(0, 0, normalized_yaw, flags=0x00)

    # def setup_360_yaw(self):
    #     """
    #     Настройка параметров подвеса для поддержки полного вращения на 360° по yaw
    #     Требует точных индексов параметров для вашей версии прошивки!
    #     """
    #     try:
    #         # ВАЖНО: Эти индексы параметров могут отличаться! Уточните для вашей прошивки.
    #         # Примерные индексы (нужно проверить в GUI или документации):
    #         YAW_MIN_INDEX = 50    # Параметр минимального угла yaw
    #         YAW_MAX_INDEX = 51    # Параметр максимального угла yaw
    #         YAW_PAN_METHOD_INDEX = 52  # Параметр метода панорамирования yaw
            
    #         # Установка минимального и максимального углов для yaw (в сотых долях градуса)
    #         self.set_parameter(YAW_MIN_INDEX, -18000)  # -180.00°
    #         self.set_parameter(YAW_MAX_INDEX, 17900)   # 179.00°
            
    #         # Настройка режима работы yaw оси (если доступно)
    #         # 2 = режим непрерывного вращения (значение может отличаться)
    #         self.set_parameter(YAW_PAN_METHOD_INDEX, 2)
            
    #         print("Настройка полного вращения по yaw выполнена")
    #         return True
    #     except Exception as e:
    #         print(f"Ошибка настройки 360° yaw: {e}")
    #         print("Уточните индексы параметров для вашей версии прошивки")
    #         return False

    # def set_parameter(self, param_index, param_value):
    #     """
    #     Установка параметра подвеса (CMD_SETPARAMETER #4)
    #     """
    #     payload = struct.pack('<HH', param_index, param_value)
    #     self.send_rc_command(0x04, payload)

    # def send_rc_command(self, command_code, payload=b''):
    #     """
    #     Общая функция для отправки RC команд
    #     """
    #     start_byte = 0xFA
    #     length = len(payload)
        
    #     crc_data = bytes([length, command_code]) + payload
    #     crc = self.calc_crc(crc_data)
        
    #     message = bytes([start_byte]) + crc_data + struct.pack('<H', crc)
