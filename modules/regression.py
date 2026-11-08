"""
==============================================================================
Module:      regression.py
Description: Ordinary Least Squares (OLS) Linear Regression implementation
             from first principles with analytical parameter estimation,
             goodness-of-fit (R-squared), and diagnostic visualization.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class LinearRegressionOLS:
    """
    Классическая одномерная линейная регрессия (OLS) без сторонних ML-библиотек.
    Модель: y = beta_0 + beta_1 * x + epsilon
    """

    def __init__(self):
        self.beta_0: Optional[float] = None
        self.beta_1: Optional[float] = None
        self.r_squared: Optional[float] = None
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegressionOLS":
        """
        Обучение модели методом наименьших квадратов (МНК).
        """
        self.x_data = np.asarray(x, dtype=float)
        self.y_data = np.asarray(y, dtype=float)

        x_mean = np.mean(self.x_data)
        y_mean = np.mean(self.y_data)

        # Аналитический расчет коэффициентов регрессии
        numerator = np.sum((self.x_data - x_mean) * (self.y_data - y_mean))
        denominator = np.sum((self.x_data - x_mean) ** 2)

        if denominator == 0:
            raise ValueError("Дисперсия предиктора x равна нулю.")

        self.beta_1 = float(numerator / denominator)
        self.beta_0 = float(y_mean - self.beta_1 * x_mean)

        # Коэффициент детерминации R^2
        y_pred = self.predict(self.x_data)
        ss_res = np.sum((self.y_data - y_pred) ** 2)
        ss_tot = np.sum((self.y_data - y_mean) ** 2)
        self.r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

        return self

    def predict(self, x_new: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Точечный прогноз по обученной модели.
        """
        if self.beta_0 is None or self.beta_1 is None:
            raise RuntimeError("Модель должна быть обучена вызовом fit() перед прогнозированием.")
        return self.beta_0 + self.beta_1 * np.asarray(x_new, dtype=float)

    def plot(self, x_highlight: Optional[float] = None, save_path: Optional[str] = None):
        """
        Построение диаграммы рассеяния и линии тренда.
        """
        if self.x_data is None or self.y_data is None:
            raise RuntimeError("Нет данных для построения графика.")

        plt.figure(figsize=(8, 6))
        plt.scatter(self.x_data, self.y_data, color="blue", label="Фактические данные", zorder=5)

        x_line = np.linspace(np.min(self.x_data), np.max(self.x_data), 100)
        y_line = self.predict(x_line)
        plt.plot(x_line, y_line, color="red", label=f"Линия OLS: y = {self.beta_0:.2f} + ({self.beta_1:.3f})x")

        if x_highlight is not None:
            y_hl = float(self.predict(x_highlight))
            plt.scatter(x_highlight, y_hl, color="green", s=100, zorder=10,
                        label=f"Прогноз ({x_highlight}, {y_hl:.2f})")

        plt.title("Зависимость цены от пробега автомобиля (OLS Регрессия)", fontsize=13)
        plt.xlabel("Пробег (тыс. км)", fontsize=11)
        plt.ylabel("Цена (тыс. $)", fontsize=11)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    print("=== ТЕСТИРОВАНИЕ МОДУЛЯ ЛИНЕЙНОЙ РЕГРЕССИИ ===")
    mileage = np.array([50, 70, 90, 120, 180])
    car_price = np.array([30, 28, 25, 20, 15])

    model = LinearRegressionOLS().fit(mileage, car_price)
    print(f"Intercept (beta_0) : {model.beta_0:.4f}")
    print(f"Slope (beta_1)     : {model.beta_1:.4f}")
    print(f"R-squared          : {model.r_squared:.4f}")

    target_mileage = 150.0
    pred_price = model.predict(target_mileage)
    print(f"Прогноз цены для пробега {target_mileage} тыс. км: {pred_price:.2f} тыс. $")
