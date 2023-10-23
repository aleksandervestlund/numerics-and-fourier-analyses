from collections.abc import Callable
import numpy as np

from exercise_07_1 import runge_kutta


def euler(
    func: Callable[[float, float], float],
    t_0: float,
    y_0: float,
    t_n: float,
    n: int,
) -> list[float]:
    h = (t_n - t_0) / n
    t = np.linspace(t_0, t_n, n + 1)
    y = [y_0]
    for i in range(n):
        y.append(y[i] + h * func(t[i], y[i]))
    return y[1:]


def heuns(
    func: Callable[[float, float], float],
    t_0: float,
    y_0: float,
    t_n: float,
    n: int,
) -> list[float]:
    h = (t_n - t_0) / n
    t = np.linspace(t_0, t_n, n + 1)
    y = [y_0]
    for i in range(n):
        k_1 = func(t[i], y[i])
        y_1_euler = y[i] + h * k_1
        k_2 = func(t[i + 1], y_1_euler)
        y.append(y[i] + h * (k_1 + k_2) / 2)

    return y[1:]


def main() -> None:
    args = (lambda t, y: -2 * t * y**2, 0.0, 1.0, 0.4)
    answer = 1 / (1 + 0.4**2)  # 25/29

    euler_answer = euler(*args, 4)
    print(euler_answer)
    print(abs(euler_answer[-1] - answer))
    print()

    heuns_answer = heuns(*args, 2)
    print(heuns_answer)
    print(abs(heuns_answer[-1] - answer))
    print()

    runge_kutta_answer = runge_kutta(*args, 0.4)
    print(runge_kutta_answer)
    print(abs(runge_kutta_answer - answer))
    print()


if __name__ == "__main__":
    main()
