"""
==============================================================================
Module:      forecasting.py
Description: Time-series forecasting algorithms and error evaluation metrics.
Includes:    Lagrange Polynomial Interpolation, Moving Averages (SMA, WMA),
             Single Exponential Smoothing (SES), Holt's Linear Trend Method,
             and Accuracy Diagnostics (MAD, MSE).
==============================================================================
"""

import numpy as np
from typing import List, Tuple, Union


def lagrange_interpolation(x_points: List[float], y_points: List[float], x: float) -> float:
    """
    Вычисляет значение интерполяционного полинома Лагранжа в заданной точке x.

    :param x_points: Известные узлы интерполяции (x-координаты).
    :param y_points: Значения функции в узлах (y-координаты).
    :param x: Точка, в которой требуется оценить значение.
    :return: Интерполированное значение.
    """
    n = len(x_points)
    if n != len(y_points):
        raise ValueError("Количество точек x и y должно совпадать.")

    result = 0.0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return float(result)


def moving_average(data: List[float], window_size: int) -> float:
    """
    Простое скользящее среднее (Simple Moving Average, SMA).

    :param data: Временной ряд наблюдений.
    :param window_size: Размер окна сглаживания.
    :return: Прогноз на следующий период.
    """
    if len(data) < window_size:
        raise ValueError(f"Недостаточно данных ({len(data)}) для окна размером {window_size}.")
    return float(np.mean(data[-window_size:]))


def weighted_moving_average(data: List[float], weights: List[float]) -> float:
    """
    Взвешенное скользящее среднее (Weighted Moving Average, WMA).

    :param data: Временной ряд наблюдений.
    :param weights: Весовые коэффициенты (от наиболее свежих к старым).
    :return: Прогноз на следующий период.
    """
    k = len(weights)
    if len(data) < k:
        raise ValueError(f"Недостаточно данных ({len(data)}) для вектора весов длины {k}.")

    weighted_sum = sum(data[-i - 1] * weights[i] for i in range(k))
    return float(weighted_sum)


def exponential_smoothing(data: List[float], alpha: float) -> float:
    """
    Простое экспоненциальное сглаживание (Single Exponential Smoothing, SES).

    :param data: Временной ряд наблюдений.
    :param alpha: Коэффициент сглаживания (0 < alpha < 1).
    :return: Прогноз на следующий шаг.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("Коэффициент alpha должен находиться в диапазоне (0, 1].")

    forecast = data[0]
    for actual in data[1:]:
        forecast = forecast + alpha * (actual - forecast)
    return float(forecast)


def holt_linear_trend(data: List[float], alpha: float, beta: float, horizon: int = 1) -> List[float]:
    """
    Двухпараметрический метод Хольта (линейное экспоненциальное сглаживание с трендом).

    :param data: Временной ряд наблюдений.
    :param alpha: Коэффициент сглаживания уровня (Level).
    :param beta: Коэффициент сглаживания тренда (Trend).
    :param horizon: Горизонт прогнозирования (количество шагов вперед).
    :return: Список прогнозов на horizon периодов вперед.
    """
    if len(data) < 2:
        raise ValueError("Для метода Хольта требуется как минимум 2 наблюдения.")

    s = [data[0]]  # Начальный уровень
    t = [data[1] - data[0]]  # Начальный тренд

    for i in range(1, len(data)):
        s_val = alpha * data[i] + (1.0 - alpha) * (s[i - 1] + t[i - 1])
        t_val = beta * (s_val - s[i - 1]) + (1.0 - beta) * t[i - 1]
        s.append(s_val)
        t.append(t_val)

    return [float(s[-1] + m * t[-1]) for m in range(1, horizon + 1)]


def calculate_mad_mse(actual: List[float], forecast: List[float]) -> Tuple[float, float]:
    """
    Расчет метрик точности прогноза:
    - MAD (Mean Absolute Deviation) — среднее абсолютное отклонение.
    - MSE (Mean Squared Error) — среднеквадратичная ошибка.
    """
    a = np.array(actual[1:], dtype=float)
    f = np.array(forecast[1:], dtype=float)

    mad = float(np.mean(np.abs(a - f)))
    mse = float(np.mean((a - f) ** 2))
    return mad, mse


if __name__ == "__main__":
    print("=== ТЕСТИРОВАНИЕ МОДУЛЯ ПРОГНОЗИРОВАНИЯ ===")

    # Тест 1: Полином Лагранжа
    x_nodes = [1, 3, 4]
    y_nodes = [100, 150, 130]
    print(f"Лагранж (x = 2): {lagrange_interpolation(x_nodes, y_nodes, 2):.2f}")

    # Тест 2: Скользящие средние
    visitors = [100, 120, 130, 90, 110]
    print(f"SMA (3 дня): {moving_average(visitors, 3):.2f}")
    print(f"WMA (веса 0.5, 0.3, 0.2): {weighted_moving_average(visitors, [0.5, 0.3, 0.2]):.2f}")

    # Тест 3: Экспоненциальное сглаживание
    sales = [120, 130, 128, 135, 133]
    print(f"SES (alpha=0.3): {exponential_smoothing(sales, 0.3):.2f}")

    # Тест 4: Метод Хольта
    sales_12m = [120, 130, 150, 160, 170, 175, 180, 190, 210, 220, 230, 240]
    h_preds = holt_linear_trend(sales_12m, alpha=0.6, beta=0.3, horizon=3)
    print(f"Хольт (прогноз на 3 месяца вперед): {[round(p, 2) for p in h_preds]}")

    # Тест 5: MAD / MSE
    ma_f = [120, 120, 125, 126, 131]
    mad_val, mse_val = calculate_mad_mse(sales, ma_f)
    print(f"Метрики SMA -> MAD: {mad_val:.2f}, MSE: {mse_val:.2f}")
