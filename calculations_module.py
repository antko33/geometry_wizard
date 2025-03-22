import math

import numpy as np
from shapely.geometry import Point
from shapely.geometry.linestring import LineString


class CalculationsModule:
    def __init__(self):
        self.t = 0
        self.l_st = 0
        self.l_ot = 0
        self.r_vn = 0

        self.r_os = 0
        self.a = 0
        self.s_x = 0
        self.s_y = 0
        self.x_c = 0
        self.y_c = 0
        self.j_x = 0
        self.j_y = 0
        self.j_p = 0
        self.w_x = 0
        self.w_y = 0
        self.i_x = 0
        self.i_y = 0
        self.h = 0
        self.b = 0

        self.__alpha1 = math.radians(60)  # Центральный угол гиба (60°)
        self.__alpha2 = math.radians(90)  # Угол гиба отвеса (90°)

    __alpha1 = math.radians(60)  # Центральный угол гиба (60°)
    __alpha2 = math.radians(90)  # Угол гиба отвеса (90°)

    def calculate_parameters(self):
        # 1. Радиус сектора по оси листа
        self.r_os = self.r_vn + self.t / 2

        self.__calculate_profile_points()
        self.__calculate_area()
        self.__calculate_static_moments()
        self.__calculate_center_of_gravity()
        self.__calculate_moments_of_inertia()
        self.__calculate_section_modulus()
        self.i_x = math.sqrt(self.j_x / self.a)
        self.i_y = math.sqrt(self.j_y / self.a)
        self.__calculate_dimensions()

    # Определение координат точек профиля для дальнейших расчетов
    def __calculate_profile_points(self):
        fi = math.pi - self.__alpha1
        r = self.r_vn + self.t / 2.0

        ax = 0
        ay = 0

        bx = r * math.sin(fi / 2.0)
        by = -r * math.cos(fi / 2.0)

        alpha2 = self.__alpha2 - fi / 2.0

        cx = bx + self.l_st * math.sin(alpha2)
        cy = by - self.l_st * math.cos(alpha2)

        alpha3 = math.pi + alpha2

        centerx = cx + r * math.cos(alpha3)
        centery = cy + r * math.sin(alpha3)

        beta = alpha2 - self.__alpha2

        dx = centerx + r * math.cos(beta)
        dy = centery + r * math.sin(beta)

        ex = dx + self.l_ot * math.sin(beta)
        ey = dy - self.l_ot * math.cos(beta)

        points_ab = self.__generate_arc_by_radius(Point(ax, ay), Point(bx, by), r)
        points_bc = self.__interpolate_line(Point(bx, by), Point(cx, cy))
        points_cd = self.__generate_arc_by_radius(Point(cx, cy), Point(dx, dy), r)
        points_de = self.__interpolate_line(Point(dx, dy), Point(ex, ey))

        points = points_ab + points_bc + points_cd + points_de

        # points — твой список точек [(x, y), ...]
        centerline = LineString(points)

        # Построим профиль с толщиной t
        profile = centerline.buffer(self.t / 2.0, cap_style="square", join_style="bevel")

        # Получим контур полигона как список точек
        contour = list(profile.exterior.coords)

        other_contour = [(-x, y) for (x, y) in contour]
        self.__points = other_contour + contour

        print(self.__points)

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
        # 1. Прямоугольник стенки
        a1 = self.l_st * self.t

        # 2. Прямоугольник отгиба
        a2 = self.l_ot * self.t

        # 3. Центральный гиб
        alpha11 = math.pi - self.__alpha1
        a3 = alpha11 / 2 * ((self.r_vn + self.t) ** 2 - self.r_vn ** 2)

        # 4. Боковой гиб
        a4 = self.__alpha2 / 2 * ((self.r_vn + self.t) ** 2 - self.r_vn ** 2)

        self.a = (2 * a1 + 2 * a2 + a3 + 2 * a4) / 100  # переводим в см²

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

    def __calculate_moments_of_inertia(self):
        """Расчет осевых моментов инерции"""
        # Метод: разбиваем на простые фигуры, находим моменты инерции каждой,
        # и суммируем с учетом положения каждой фигуры относительно центра тяжести

        # Для осевых моментов инерции используем:
        # Jx = 1/12 * ∑[(y_i² + y_i*y_{i+1} + y_{i+1}²) * (x_i*y_{i+1} - x_{i+1}*y_i)]
        # Jy = 1/12 * ∑[(x_i² + x_i*x_{i+1} + x_{i+1}²) * (x_i*y_{i+1} - x_{i+1}*y_i)]

        points = np.array(self.__points)
        x = points[:, 0]
        y = points[:, 1]

        j_x = 0
        j_y = 0

        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            j_x += (y[i] ** 2 + y[i] * y[j] + y[j] ** 2) * (x[i] * y[j] - x[j] * y[i])
            j_y += (x[i] ** 2 + x[i] * x[j] + x[j] ** 2) * (x[i] * y[j] - x[j] * y[i])

        # Делим на 12 и берем абсолютное значение
        self.j_x = abs(j_x) / 12 / 10000  # переводим в см⁴
        self.j_y = abs(j_y) / 12 / 10000  # переводим в см⁴
        self.j_p = self.j_x + self.j_y

        # Перенос моментов инерции к центральным осям (если нужно)
        # Jx_c = Jx - A * yc²
        # Jy_c = Jy - A * xc²
        self.j_x = self.j_x - self.a * (self.y_c ** 2)
        self.j_y = self.j_y - self.a * (self.x_c ** 2)

    def __calculate_section_modulus(self):
        """Расчет моментов сопротивления"""
        # Находим наиболее удаленные точки от центра тяжести
        points = np.array(self.__points)
        x = points[:, 0] / 10  # переводим в см
        y = points[:, 1] / 10  # переводим в см

        x_dists = abs(x - self.x_c)
        y_dists = abs(y - self.y_c)

        x_max = max(x_dists)
        y_max = max(y_dists)

        # Расчет моментов сопротивления
        self.w_x = self.j_x / y_max  # см³
        self.w_y = self.j_y / x_max  # см³

    def __calculate_dimensions(self):
        """Расчет высоты и ширины профиля"""
        points = np.array(self.__points)
        x = points[:, 0]
        y = points[:, 1]

        self.h = max(y) - min(y)  # мм
        self.b = max(x) - min(x)  # мм

    def __generate_arc_by_radius(self, p1, p2, radius, direction='cw', n_points=10):
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        dx, dy = x2 - x1, y2 - y1
        chord_len = math.hypot(dx, dy)

        # Проверка: можно ли построить окружность с таким радиусом
        if chord_len > 2 * radius:
            raise ValueError("Радиус слишком мал: не удаётся построить окружность через 2 точки.")

        # Середина хорды
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        # Расстояние от середины хорды до центра окружности
        h = math.sqrt(radius ** 2 - (chord_len / 2) ** 2)

        # Нормаль к хорде
        nx, ny = -dy / chord_len, dx / chord_len

        # Центр окружности: два возможных решения (влево/вправо от хорды)
        if direction == 'ccw':
            cx, cy = mx + nx * h, my + ny * h
        elif direction == 'cw':
            cx, cy = mx - nx * h, my - ny * h
        else:
            raise ValueError("direction должен быть 'ccw' или 'cw'")

        # Углы на окружности
        angle1 = math.atan2(y1 - cy, x1 - cx)
        angle2 = math.atan2(y2 - cy, x2 - cx)

        # Упорядочим углы
        if direction == 'ccw' and angle2 < angle1:
            angle2 += 2 * math.pi
        elif direction == 'cw' and angle2 > angle1:
            angle2 -= 2 * math.pi

        # Углы точек
        angles = np.linspace(angle1, angle2, n_points)
        arc_points = [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]

        return arc_points

    def __interpolate_line(self, p1, p2, n_points=10):
        """
        Возвращает n_points точек на прямом отрезке от p1 до p2, включая сами точки.

        :param p1: tuple (x1, y1) — начальная точка
        :param p2: tuple (x2, y2) — конечная точка
        :param n_points: сколько точек нужно (включая начальную и конечную)
        :return: список кортежей [(x, y), ...]
        """
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        points = [
            (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
            for t in [i / (n_points - 1) for i in range(n_points)]
        ]
        return points
