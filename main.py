"""
==============================================================================
Project:     Decision Support & Quantitative Forecasting Toolkit
File:        main.py
Author:      Yevhen Biezhyn
Description: Master CLI orchestrator importing modular solvers:
             - Time Series Forecasting & Error Evaluation (modules.forecasting)
             - Ordinary Least Squares Regression (modules.regression)
             - Multi-Criteria Decision Analysis (modules.mcda)
             - Transportation Logistics Optimization (modules.transportation)
==============================================================================
"""

import argparse
import numpy as np

# Чистый импорт из локальных модулей
from modules.forecasting import (
    lagrange_interpolation,
    moving_average,
    weighted_moving_average,
    exponential_smoothing,
    holt_linear_trend,
    calculate_mad_mse
)
from modules.regression import LinearRegressionOLS
from modules.mcda import score_matrix, score_dict
from modules.transportation import TransportationSolver


def print_banner(title: str) -> None:
    """Вспомогательная функция для форматирования консольных заголовков."""
    print("\n" + "=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)


def run_forecasting_suite() -> None:
    """Запуск комплекса моделей прогнозирования временных рядов."""
    print_banner("1. Модуль: Прогнозирование временных рядов и оценка точности")

    # 1.1 Полиномиальная интерполяция Лагранжа
    x_nodes = [1, 3, 4]
    y_nodes = [100, 150, 130]
    target_day = 2
    pred_lagrange = lagrange_interpolation(x_nodes, y_nodes, target_day)
    print(f"[*] Интерполяция Лагранжа (день x = {target_day}): {pred_lagrange:.2f}")

    # 1.2 Скользящие средние (SMA & WMA)
    traffic_data = [100, 120, 130, 90, 110]
    sma_val = moving_average(traffic_data, window_size=3)
    wma_val = weighted_moving_average(traffic_data, weights=[0.5, 0.3, 0.2])
    print(f"[*] Простое скользящее среднее (SMA, n = 3)    : {sma_val:.2f}")
    print(f"[*] Взвешенное скользящее среднее (WMA, k = 3) : {wma_val:.2f}")

    # 1.3 Простое экспоненциальное сглаживание (SES)
    sales_5m = [120, 130, 128, 135, 133]
    ses_val = exponential_smoothing(sales_5m, alpha=0.3)
    print(f"[*] Экспоненциальное сглаживание (SES, a = 0.3): {ses_val:.2f}")

    # 1.4 Двухпараметрический метод Хольта (уровень + тренд)
    sales_12m = [120, 130, 150, 160, 170, 175, 180, 190, 210, 220, 230, 240]
    holt_preds = holt_linear_trend(sales_12m, alpha=0.6, beta=0.3, horizon=3)
    print(f"[*] Метод Хольта (прогноз на мес. 13, 14, 15)  : {', '.join(f'{v:.2f}' for v in holt_preds)}")

    # 1.5 Сравнение метрик погрешности (MAD / MSE)
    print("\n  Диагностика точности моделей (MAD / MSE):")
    actual_series = [120, 130, 128, 135, 133]
    forecast_candidates = {
        "SMA": [120, 120, 125, 126, 131],
        "WMA": [120, 121, 124, 127, 130],
        "SES": [120, 123, 124.5, 127.65, 129.255]
    }
    for model_name, f_vals in forecast_candidates.items():
        mad, mse = calculate_mad_mse(actual_series, f_vals)
        print(f"    - {model_name:<4} -> MAD: {mad:5.2f} | MSE: {mse:6.2f}")


def run_regression_suite() -> None:
    """Запуск аналитической регрессии методом наименьших квадратов (OLS)."""
    print_banner("2. Модуль: Линейная регрессия методом наименьших квадратов (OLS)")

    mileage = np.array([50, 70, 90, 120, 180])
    car_price = np.array([30, 28, 25, 20, 15])

    model = LinearRegressionOLS().fit(mileage, car_price)
    target_km = 150.0
    predicted_price = model.predict(target_km)

    print(f"[*] Уравнение регрессии : Цена = {model.beta_0:.3f} + ({model.beta_1:.4f}) * Пробег")
    print(f"[*] Коэффициент R²      : {model.r_squared:.4f}")
    print(f"[*] Прогноз цены ({target_km:.0f} тыс. км): {predicted_price:.2f} тыс. $")


def run_mcda_suite() -> None:
    """Запуск моделей многокритериальной оценки решений (MCDA / WSM)."""
    print_banner("3. Модуль: Многокритериальный анализ решений (MCDA)")

    # 3.1 Матричный подход (выбор ноутбука)
    laptop_options = ["MacBook Pro", "Dell XPS", "Lenovo ThinkPad", "HP Spectre"]
    ratings_matrix = np.array([
        [9, 7, 5, 9],
        [8, 8, 6, 8],
        [7, 9, 8, 7],
        [8, 6, 7, 8]
    ])
    criteria_weights = np.array([0.4, 0.3, 0.2, 0.1])
    ranking = score_matrix(laptop_options, ratings_matrix, criteria_weights)

    print("[*] Рейтинг альтернатив (Матричная модель):")
    for place, (name, score) in enumerate(ranking, start=1):
        print(f"    Место {place}: {name:<16} | Итоговый балл: {score:.3f}")

    # 3.2 Словарный подход (выбор автомобиля)
    cars_data = {
        "Car A": {"economy": 0.9, "comfort": 0.4, "power": 0.3},
        "Car B": {"economy": 0.6, "comfort": 0.8, "power": 0.4},
        "Car C": {"economy": 0.3, "comfort": 0.6, "power": 0.9}
    }
    car_weights = {"economy": 0.4, "comfort": 0.3, "power": 0.3}
    best_car, all_scores = score_dict(cars_data, car_weights)
    print(f"\n[*] Оптимальный выбор по критериям: {best_car} (балл: {all_scores[best_car]:.2f})")


def run_transportation_suite() -> None:
    """Запуск оптимизации сбалансированной транспортной задачи."""
    print_banner("4. Модуль: Оптимизация транспортных потоков (Логистика)")

    tariff_matrix = np.array([
        [5, 4, 3],
        [8, 4, 3],
        [9, 7, 5]
    ])
    warehouse_supply = np.array([100, 300, 300])
    store_demand = np.array([300, 200, 200])

    solver = TransportationSolver(tariff_matrix, warehouse_supply, store_demand)
    allocation_plan, min_cost = solver.solve_stochastic(iterations=1000)

    print("[*] Сформированный оптимальный план перевозок:")
    print(allocation_plan)
    print(f"[*] Минимальные суммарные затраты: {min_cost:.2f} ден. ед.")


def main():
    parser = argparse.ArgumentParser(
        description="Decision Support & Quantitative Forecasting Suite",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--module",
        choices=["forecasting", "regression", "mcda", "transport", "all"],
        default="all",
        help=(
            "forecasting : Запуск алгоритмов прогнозирования (SMA, WMA, SES, Holt, Lagrange)\n"
            "regression  : Запуск аналитической OLS-регрессии\n"
            "mcda        : Запуск многокритериального выбора (WSM)\n"
            "transport   : Запуск решения транспортной задачи\n"
            "all         : Запуск всех модулей последовательно (по умолчанию)"
        )
    )
    args = parser.parse_args()

    suite_router = {
        "forecasting": run_forecasting_suite,
        "regression": run_regression_suite,
        "mcda": run_mcda_suite,
        "transport": run_transportation_suite,
    }

    if args.module == "all":
        for execute_suite in suite_router.values():
            execute_suite()
    else:
        suite_router[args.module]()

    print("\n" + "=" * 75)
    print("  РАСЧЕТЫ УСПЕШНО ЗАВЕРШЕНЫ.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
