import math

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.geometry.linestring import LineString
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union


def generate_arc_by_radius(p1, p2, radius, direction='cw', n_points=500):
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


def interpolate_line(p1, p2, n_points=10):
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


def visualize(line: LineString, size=6):
    x, y = line.xy

    # Рисуем
    plt.figure(figsize=(size, size))
    plt.plot(x, y, '-', color='blue', linewidth=2)
    # plt.scatter(x, y, color='red')  # точки, по которым проходит линия
    plt.axis('equal')
    plt.grid(True)
    plt.show()


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

        alpha2 = (math.pi - fi) / 2.0

        cx = bx + self.l_st * math.sin(alpha2)
        cy = by - self.l_st * math.cos(alpha2)

        alpha3 = math.pi + alpha2

        center_x = cx + r * math.cos(alpha3)
        center_y = cy + r * math.sin(alpha3)

        beta = alpha2 - self.__alpha2

        dx = center_x + r * math.cos(beta)
        dy = center_y + r * math.sin(beta)

        ex = dx + self.l_ot * math.sin(beta)
        ey = dy - self.l_ot * math.cos(beta)

        points_ab = [(ax, ay)] + generate_arc_by_radius(Point(ax, ay), Point(bx, by), r) + [(bx, by)]
        points_bc = [(bx, by)] + interpolate_line(Point(bx, by), Point(cx, cy)) + [(cx, cy)]
        points_cd = [(cx, cy)] + generate_arc_by_radius(Point(cx, cy), Point(dx, dy), r) + [(dx, dy)]
        points_de = [(dx, dy)] + interpolate_line(Point(dx, dy), Point(ex, ey)) + [(ex, ey)]

        points = points_ab + points_bc + points_cd + points_de

        # points — список точек [(x, y), ...]
        centerline = LineString(points)

        # Построим профиль с толщиной t
        profile = centerline.buffer(self.t / 2.0, cap_style="flat", join_style="bevel")

        # Получим контур полигона как список точек
        contour_right = list(profile.exterior.coords)
        contour_left = [(-x, y) for (x, y) in contour_right]

        merged = unary_union([Polygon(contour_left), Polygon(contour_right)])
        self.__poly = Polygon(merged)
        self.__points = self.__poly.exterior.coords

    def __calculate_area(self):
        self.a = self.__poly.area / 100  # переводим в см²

    def __calculate_static_moments(self):
        x, y = self.__poly.exterior.xy

        s_x = 0.0
        s_y = 0.0

        for i in range(len(x) - 1):
            common_term = x[i] * y[i + 1] - x[i + 1] * y[i]
            s_x += (y[i] + y[i + 1]) * common_term
            s_y += (x[i] + x[i + 1]) * common_term

        self.s_x = abs(s_x) / 6 / 1000
        self.s_y = abs(s_y) / 6 / 1000

    def __calculate_center_of_gravity(self):
        self.x_c = self.__poly.centroid.x / 10
        self.y_c = self.__poly.centroid.y / 10

    def __calculate_moments_of_inertia(self):
        x, y = self.__poly.exterior.xy

        i_x = 0.0
        i_y = 0.0

        for i in range(len(x) - 1):
            common = x[i] * y[i + 1] - x[i + 1] * y[i]
            i_x += common * (y[i] ** 2 + y[i] * y[i + 1] + y[i + 1] ** 2)
            i_y += common * (x[i] ** 2 + x[i] * x[i + 1] + x[i + 1] ** 2)

        self.j_x = abs(i_x) / 12 / 10000
        self.j_y = abs(i_y) / 12 / 10000

        self.j_x = self.j_x - self.a * (self.y_c ** 2)
        self.j_y = self.j_y - self.a * (self.x_c ** 2)
        self.j_p = self.j_x + self.j_y

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
