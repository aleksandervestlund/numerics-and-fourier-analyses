import math
import numpy as np


def f_prime_2(x: np.ndarray) -> np.ndarray:
    return -5 * np.sin(x) * np.cos(2 * x) - 4 * np.sin(2 * x) * np.cos(x)


def g(x: float) -> float:
    return np.exp(x)


def convert_to_gauss(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return (b - a) / 2 * x - (b + a) / 2


def compute_error(a: float, b: float, n: int) -> float:
    return (
        (b - a) ** (2 * n + 1)
        * math.factorial(n) ** 4
        * g(3)
        / ((2 * n + 1) * math.factorial(2 * n) ** 3)
    )


def main() -> None:
    # print(max(abs(f_prime_2(np.linspace(np.pi / 6, np.pi / 3, 100)))))

    x_values = np.array(
        [
            -math.sqrt(525 + 70 * math.sqrt(30)) / 35,
            -math.sqrt(525 - 70 * math.sqrt(30)) / 35,
            math.sqrt(525 - 70 * math.sqrt(30)) / 35,
            math.sqrt(525 + 70 * math.sqrt(30)) / 35,
        ]
    )
    y_values = np.array(
        [
            (18 - math.sqrt(30)) / 36,
            (18 + math.sqrt(30)) / 36,
            (18 + math.sqrt(30)) / 36,
            (18 - math.sqrt(30)) / 36,
        ]
    )
    print(
        sum(
            y_value * g(x_value)
            for x_value, y_value in zip(
                convert_to_gauss(x_values, -3, 3), y_values
            )
        )
        * 3
    )
    # print(compute_error(-3, 3, 4))


if __name__ == "__main__":
    main()
