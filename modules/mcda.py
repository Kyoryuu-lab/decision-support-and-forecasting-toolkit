"""
==============================================================================
Module:      mcda.py
Description: Multi-Criteria Decision Analysis (MCDA).
             Implements the Weighted Sum Model (WSM) via vector-matrix
             algebra and structured dictionary scoring for ranking alternatives.
==============================================================================
"""

import numpy as np
from typing import List, Dict, Tuple


def score_matrix(alternatives: List[str], ratings: np.ndarray, weights: np.ndarray) -> List[Tuple[str, float]]:
    """
    Оценка альтернатив на основе матрицы решений и вектора весов (Matrix Dot Product).

    :param alternatives: Список названий альтернатив (например, модели ноутбуков).
    :param ratings: Двумерный массив оценок размерности (n_alternatives, n_criteria).
    :param weights: Одномерный массив весов критериев (сумма = 1.0).
    :return: Отсортированный список кортежей (альтернатива, итоговый_балл).
    """
    ratings = np.asarray(ratings, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if ratings.shape[1] != len(weights):
        raise ValueError("Число столбцов матрицы оценок должно совпадать с числом весов.")
    if len(alternatives) != ratings.shape[0]:
        raise ValueError("Количество альтернатив не совпадает со строками матрицы.")

    scores = ratings @ weights
    ranking = sorted(zip(alternatives, scores), key=lambda x: x[1], reverse=True)
    return ranking


def score_dict(candidates: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    """
    Оценка альтернатив, заданных в виде словаря атрибутов.

    :param candidates: Словарь вида {"Название": {"критерий_1": оценка, ...}}.
    :param weights: Словарь весов вида {"критерий_1": вес, ...}.
    :return: Кортеж из (лучшая_альтернатива, словарь_баллов_всех_кандидатов).
    """
    total_scores = {}
    for name, criteria in candidates.items():
        score = sum(criteria.get(crit, 0.0) * weight for crit, weight in weights.items())
        total_scores[name] = float(score)

    best_candidate = max(total_scores, key=total_scores.get)
    return best_candidate, total_scores


if __name__ == "__main__":
    print("=== ТЕСТИРОВАНИЕ МОДУЛЯ ПРИНЯТИЯ РЕШЕНИЙ (MCDA) ===")

    # Тест 1: Матричный подход (Выбор ноутбука)
    laptop_names = ["MacBook Pro", "Dell XPS", "Lenovo ThinkPad", "HP Spectre"]
    ratings_matrix = np.array([
        [9, 7, 5, 9],
        [8, 8, 6, 8],
        [7, 9, 8, 7],
        [8, 6, 7, 8]
    ])
    crit_weights = np.array([0.4, 0.3, 0.2, 0.1])

    print("Рейтинг ноутбуков:")
    for rank, (name, sc) in enumerate(score_matrix(laptop_names, ratings_matrix, crit_weights), 1):
        print(f"  {rank}. {name:<16}: {sc:.3f}")

    # Тест 2: Словарный подход (Выбор автомобиля)
    cars_data = {
        "Car A": {"economy": 0.9, "comfort": 0.4, "power": 0.3},
        "Car B": {"economy": 0.6, "comfort": 0.8, "power": 0.4},
        "Car C": {"economy": 0.3, "comfort": 0.6, "power": 0.9}
    }
    car_weights = {"economy": 0.4, "comfort": 0.3, "power": 0.3}

    best, all_sc = score_dict(cars_data, car_weights)
    print(f"\nЛучший автомобиль: {best} с баллом {all_sc[best]:.2f}")
