import math

import numpy as np


class MainViewModel:
    t = 0
    l_st = 0
    l_ot = 0
    r_vn = 0

    r_os = 0
    a = 0
    s_x = 0
    s_y = 0
    x_c = 0
    y_c = 0
    j_x = 0
    j_y = 0
    j_p = 0
    w_x = 0
    w_y = 0
    i_x = 0
    i_y = 0
    h = 0
    b = 0

    __alpha1 = math.radians(60)  # Центральный угол гиба (60°)
    __alpha2 = math.radians(90)  # Угол гиба отвеса (90°)

    def calculate_parameters(self):
        # 1. Радиус сектора по оси листа
        self.r_os = self.r_vn + self.t / 2

        self.__calculate_profile_points()

        self.__calculate_area()

        self.__calculate_static_moments()

        self.__calculate_center_of_gravity()

    # Определение координат точек профиля для дальнейших расчетов
    def __calculate_profile_points(self):
        """Рассчитать координаты точек профиля для дальнейших вычислений"""
        # Рассчитываем координаты основных точек профиля
        # Начнём с левого нижнего угла как начала координат (0,0)

        # Точка 1: начало координат
        p1 = (0, 0)

        # Точка 2: конец первой стенки снизу
        p2 = (self.l_st, 0)

        # Точка 3: конец первого гиба (α1 = 60°)
        # Расчет координат после поворота на угол __alpha1 вокруг центра окружности
        center1_x = self.l_st
        center1_y = self.r_os
        p3_x = center1_x + self.r_os * math.cos(math.pi / 2 - self.__alpha1)
        p3_y = center1_y + self.r_os * math.sin(math.pi / 2 - self.__alpha1)
        p3 = (p3_x, p3_y)

        # Точка 4: конец второй стенки
        p4_x = p3_x + self.l_st * math.cos(self.__alpha1)
        p4_y = p3_y + self.l_st * math.sin(self.__alpha1)
        p4 = (p4_x, p4_y)

        # Точка 5: конец второго гиба (α2 = 90°)
        center2_x = p4_x
        center2_y = p4_y - self.r_os
        p5_x = center2_x + self.r_os * math.cos(self.__alpha1 + math.pi / 2)
        p5_y = center2_y + self.r_os * math.sin(self.__alpha1 + math.pi / 2)
        p5 = (p5_x, p5_y)

        # Точка 6: конец отгиба
        p6_x = p5_x + self.l_ot * math.cos(self.__alpha1 + self.__alpha2)
        p6_y = p5_y + self.l_ot * math.sin(self.__alpha1 + self.__alpha2)
        p6 = (p6_x, p6_y)

        # Точки для внутренней линии профиля (с учетом толщины)
        # Точка 7: начало внутренней линии первой стенки
        p7 = (0, self.t)

        # Точка 8: конец внутренней линии первой стенки
        p8 = (self.l_st - (self.r_os - self.r_vn), self.t)

        # Точка 9: конец внутреннего первого гиба
        center1_in_x = self.l_st - (self.r_os - self.r_vn)
        center1_in_y = self.r_vn + self.t
        p9_x = center1_in_x + self.r_vn * math.cos(math.pi / 2 - self.__alpha1)
        p9_y = center1_in_y + self.r_vn * math.sin(math.pi / 2 - self.__alpha1)
        p9 = (p9_x, p9_y)

        # Точка 10: конец внутренней линии второй стенки
        p10_x = p9_x + (self.l_st - (self.r_os - self.r_vn)) * math.cos(self.__alpha1)
        p10_y = p9_y + (self.l_st - (self.r_os - self.r_vn)) * math.sin(self.__alpha1)
        p10 = (p10_x, p10_y)

        # Точка 11: конец внутреннего второго гиба
        center2_in_x = p10_x
        center2_in_y = p10_y - self.r_vn
        p11_x = center2_in_x + self.r_vn * math.cos(self.__alpha1 + math.pi / 2)
        p11_y = center2_in_y + self.r_vn * math.sin(self.__alpha1 + math.pi / 2)
        p11 = (p11_x, p11_y)

        # Точка 12: конец внутренней линии отгиба
        p12_x = p11_x + (self.l_ot - (self.r_os - self.r_vn)) * math.cos(self.__alpha1 + self.__alpha2)
        p12_y = p11_y + (self.l_ot - (self.r_os - self.r_vn)) * math.sin(self.__alpha1 + self.__alpha2)
        p12 = (p12_x, p12_y)

        # Сохраняем все точки
        self.__points = [p1, p2, p3, p4, p5, p6, p12, p11, p10, p9, p8, p7]

    def __calculate_area(self):
        """Расчет площади поперечного сечения"""
        # Разбиваем профиль на простые фигуры:
        # 1. Прямоугольник первой стенки
        a1 = self.l_st * self.t

        # 2. Сектор первого гиба
        a2 = self.__alpha1 * (self.r_os ** 2 - self.r_vn ** 2) / 2

        # 3. Прямоугольник второй стенки
        a3 = self.l_st * self.t

        # 4. Сектор второго гиба
        a4 = self.__alpha2 * (self.r_os ** 2 - self.r_vn ** 2) / 2

        # 5. Прямоугольник отгиба
        a5 = self.l_ot * self.t

        self.__a = (a1 + a2 + a3 + a4 + a5) / 100  # переводим в см²

    def __calculate_static_moments(self):
        """Расчет статических моментов площади поперечного сечения"""
        # Метод: разбиваем на простые фигуры, находим статические моменты каждой,
        # и суммируем с учетом положения каждой фигуры

        # Для простоты будем использовать метод интегрирования по контуру (метод Грина)
        # Формула: Sx = ∫y dA, Sy = ∫x dA

        points = np.array(self.__points)
        x = points[:, 0]
        y = points[:, 1]

        # Используем формулу площади многоугольника
        # A = 0.5 * |∑(x_i * y_{i+1} - x_{i+1} * y_i)|

        # Для статических моментов используем:
        # Sx = 1/6 * ∑[(y_i + y_{i+1}) * (x_i * y_{i+1} - x_{i+1} * y_i)]
        # Sy = 1/6 * ∑[(x_i + x_{i+1}) * (x_i * y_{i+1} - x_{i+1} * y_i)]

        s_x = 0
        s_y = 0

        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            s_x += (y[i] + y[j]) * (x[i] * y[j] - x[j] * y[i])
            s_y += (x[i] + x[j]) * (x[i] * y[j] - x[j] * y[i])

        # Делим на 6 и берем абсолютное значение
        self.s_x = abs(s_x) / 6 / 1000  # переводим в см³
        self.s_y = abs(s_y) / 6 / 1000  # переводим в см³

    def __calculate_center_of_gravity(self):
        """Расчет координат центра тяжести"""
        # Формулы: xc = Sy/A, yc = Sx/A
        self.x_c = self.s_y / self.a  # см
        self.y_c = self.s_x / self.a  # см
