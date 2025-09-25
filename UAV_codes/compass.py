import time
import math
import smbus2
import threading

class Compass:
    def __init__(self, bus_number=8, address=0x1E):
        """
        Инициализация компаса HMC5883L
        """
        self.bus = smbus2.SMBus(bus_number)
        self.address = address
        self.lock = threading.Lock()
        
        # Настройка компаса
        self.setup()
        print("Компас HMC5883L инициализирован")

    def setup(self):
        """Настройка регистров компаса"""
        with self.lock:
            self.bus.write_byte_data(self.address, 0x00, 0x70)  # 8 samples @ 15Hz
            self.bus.write_byte_data(self.address, 0x01, 0x20)  # 1.3 gain
            self.bus.write_byte_data(self.address, 0x02, 0x00)  # Continuous mode

    def read_raw_data(self, addr):
        """Чтение сырых данных с компаса"""
        with self.lock:
            high = self.bus.read_byte_data(self.address, addr)
            low = self.bus.read_byte_data(self.address, addr + 1)
        
        value = (high << 8) + low
        if value > 32768:
            value = value - 65536
        return value

    def get_heading(self):
        """Получение направления в градусах"""
        x = self.read_raw_data(0x03)  # X
        y = self.read_raw_data(0x07)  # Y
        z = self.read_raw_data(0x05)  # Z
        
        # Расчет направления в радианах
        heading_rad = math.atan2(y, x)
        
        # Поправка на магнитное склонение (например, 0.22 для ~13 градусов)
        declination_angle = 0.22
        heading_rad += declination_angle
        
        # Коррекция отрицательных значений
        if heading_rad < 0:
            heading_rad += 2 * math.pi
        
        # Конвертация в градусы
        heading_deg = heading_rad * (180.0 / math.pi)
        
        return heading_deg

    def get_direction_name(self, heading):
        """Получение названия направления"""
        directions = ["С", "СВ", "В", "ЮВ", "Ю", "ЮЗ", "З", "СЗ"]
        index = round(heading / 45) % 8
        return directions[index]

    def get_compass_data(self):
        """Получение полных данных компаса"""
        heading = self.get_heading()
        direction = self.get_direction_name(heading)
        return heading, direction
