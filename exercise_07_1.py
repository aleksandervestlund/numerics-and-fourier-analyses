from collections.abc import Callable

import numpy as np


def f(t: float, y: float) -> float:
    return -2 * t * y


def actual_f(t: float) -> float:
    return np.exp(-(t**2))


def runge_kutta(
    func: Callable[[float, float], float],
    t_0: float,
    y_0: float,
    t_n: float,
    h: float,
) -> float:
    while t_0 < t_n:
        k_1 = func(t_0, y_0)
        k_2 = func(t_0 + h / 2, y_0 + h * k_1 / 2)
        k_3 = func(t_0 + h / 2, y_0 + h * k_2 / 2)
        k_4 = func(t_0 + h, y_0 + h * k_3)
        print(k_1, k_2, k_3, k_4)

        t_0 += h
        y_0 += h * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

    return y_0


def error_runge_kutta(
    func: Callable[[float, float], float],
    actual_func: Callable[[float], float],
    t_0: float,
    y_0: float,
    t_n: float,
    h: float,
):
    return abs(runge_kutta(func, t_0, y_0, t_n, h) - actual_func(t_n))


def main() -> None:
    h = 2.0
    for _ in range(10):
        next_h = h / 2
        error_1 = error_runge_kutta(f, actual_f, 0, 1, 2, h)
        error_2 = error_runge_kutta(f, actual_f, 0, 1, 2, next_h)

        print(
            f"Convergence: {np.log(error_1 / error_2) / np.log(h / next_h)} ≈ 4"
        )

        h = next_h


if __name__ == "__main__":
    main()
