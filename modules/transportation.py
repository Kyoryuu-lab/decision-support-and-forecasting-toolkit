"""
==============================================================================
Module:      transportation.py
Description: Logistics Transportation Problem solver.
             Computes balanced supply-demand allocation matrices
             and minimizes total logistics freight costs.
==============================================================================
"""

import numpy as np
from typing import Tuple


class TransportationSolver:
    """
    Решатель классической сбалансированной транспортной задачи линейного программирования.
    Целевая функция: min sum(c_ij * x_ij) при ограничениях по запасам и потребностям.
    """

    def __init__(self, costs: np.ndarray, supply: np.ndarray, demand: np.ndarray):
        self.costs = np.asarray(costs, dtype=float)
        self.supply = np.asarray(supply, dtype=int)
        self.demand = np.asarray(demand, dtype=int)

        if np.sum(self.supply) != np.sum(self.demand):
            raise ValueError("Задача не сбалансирована: суммарный объем складов не равен спросу.")

    def calculate_cost(self, allocation: np.ndarray) -> float:
        """Расчет суммарных транспортных расходов для заданной схемы распределения."""
        return float(np.sum(allocation * self.costs))

    def solve_stochastic(self, iterations: int = 2000, seed: int = 42) -> Tuple[np.ndarray, float]:
        """
        Стохастический поиск допустимого базисного плана с минимальными затратами.

        :param iterations: Количество стохастических итераций поиска.
        :param seed: Зерно генератора случайных чисел для воспроизводимости.
        :return: (лучшая_матрица_перевозок, минимальная_стоимость).
        """
        np.random.seed(seed)
        best_cost = float("inf")
        best_solution = np.zeros_like(self.costs, dtype=int)

        m, n = self.costs.shape

        for _ in range(iterations):
            solution = np.zeros_like(self.costs, dtype=int)
            cur_supply = self.supply.copy()
            cur_demand = self.demand.copy()

            for i in range(m):
                for j in range(n):
                    max_possible = min(cur_supply[i], cur_demand[j])
                    if max_possible > 0:
                        alloc = np.random.randint(0, max_possible + 1)
                        solution[i, j] = alloc
                        cur_supply[i] -= alloc
                        cur_demand[j] -= alloc

            # Проверка выполнения граничных условий (спрос и склад закрыты в 0)
            if np.sum(cur_supply) == 0 and np.sum(cur_demand) == 0:
                cost = self.calculate_cost(solution)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = solution

        return best_solution, best_cost


if __name__ == "__main__":
    print("=== ТЕСТИРОВАНИЕ ТРАНСПОРТНОЙ ЗАДАЧИ ===")
    tariff_matrix = np.array([
        [5, 4, 3],
        [8, 4, 3],
        [9, 7, 5]
    ])
    warehouse_supply = np.array([100, 300, 300])
    store_demand = np.array([300, 200, 200])

    solver = TransportationSolver(tariff_matrix, warehouse_supply, store_demand)
    plan, min_exp = solver.solve_stochastic(iterations=2000)

    print("Оптимальный план перевозок:")
    print(plan)
    print(f"Минимальные затраты: {min_exp:.2f}")
