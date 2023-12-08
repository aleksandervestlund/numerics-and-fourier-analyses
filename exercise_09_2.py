from collections.abc import Callable
from functools import partial
import numpy as np
import matplotlib.pyplot as plt


X_VALUES = np.linspace(-3 * np.pi, 3 * np.pi, 1000)


def convert_x_value(x: float) -> float:
    x %= 2 * np.pi
    if x > np.pi:
        x -= 2 * np.pi
    return x


# Define the truncated series function
def truncated_series(
    x: np.ndarray,
    n: int,
    a_0: float,
    a_n: Callable[[int], float],
    b_n: Callable[[int], float],
) -> np.ndarray:
    result = a_0 * np.ones_like(x)

    for m in range(1, n + 1):
        result += a_n(m) * np.cos(m * x) + b_n(m) * np.sin(m * x)

    return result


def exercise_a() -> partial[np.ndarray]:
    def f(x: float) -> float:
        x = convert_x_value(x)

        if 0 <= x < np.pi / 2:
            return x
        return 0.0

    def a_n(n: int) -> float:
        return (
            np.pi * n * np.sin(n * np.pi / 2) + 2 * np.cos(n * np.pi / 2) - 2
        ) / (2 * np.pi * n**2)

    def b_n(n: int) -> float:
        return (
            2 * np.sin(n * np.pi / 2) - np.pi * n * np.cos(n * np.pi / 2)
        ) / (2 * np.pi * n**2)

    plot_original_function(f)
    return partial(
        truncated_series, x=X_VALUES, a_0=np.pi / 16, a_n=a_n, b_n=b_n
    )


def exercise_b() -> partial[np.ndarray]:
    def f(x: float) -> float:
        x = convert_x_value(x)

        if x <= 0:
            return 0.0
        if x <= np.pi / 2:
            return x
        return np.pi - x

    def a_n(n: int) -> float:
        return (2 * np.cos(n * np.pi / 2) - (-1) ** n - 1) / (np.pi * n**2)

    def b_n(n: int) -> float:
        return 2 * np.sin(np.pi * n / 2) / (np.pi * n**2)

    plot_original_function(f)
    return partial(
        truncated_series, x=X_VALUES, a_0=np.pi / 8, a_n=a_n, b_n=b_n
    )


def exercise_c() -> partial[np.ndarray]:
    def f(x: float) -> float:
        x = convert_x_value(x)

        if x <= -np.pi / 2:
            return -(np.pi + x)
        if x <= np.pi / 2:
            return x
        return np.pi - x

    def a_n(n: int) -> float:
        return 0.0 * n

    def b_n(n: int) -> float:
        return (4 * np.sin(np.pi * n / 2)) / (np.pi * n**2)

    plot_original_function(f)
    return partial(truncated_series, x=X_VALUES, a_0=0.0, a_n=a_n, b_n=b_n)


def plot_original_function(function: Callable[[float], float]) -> None:
    plt.plot(
        X_VALUES, [function(x_value) for x_value in X_VALUES], label="f(x)"
    )


def plot_truncated_series(s_n: partial[np.ndarray]) -> None:
    for n in (5, 20, 100):
        plt.plot(X_VALUES, s_n(n=n), label=f"S_{n}(x)")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("S_N(x)")
    plt.title("Truncated Series")
    plt.show()


def main() -> None:
    plot_truncated_series(exercise_a())
    plot_truncated_series(exercise_b())
    plot_truncated_series(exercise_c())


if __name__ == "__main__":
    main()
