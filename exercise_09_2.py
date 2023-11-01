import math
import numpy as np
import matplotlib.pyplot as plt


# Define the truncated series function
def truncated_series(x: np.ndarray, n: int) -> np.ndarray:
    result = 0 * x
    a_0 = math.pi / 4

    for m in range(1, n + 1):
        a_n = (
            -2
            + 2 * math.cos((math.pi * m) / 2)
            + math.pi * m * math.sin(math.pi * m / 2) / 2 * m**2
        ) + (
            -math.cos((math.pi * m) / 2)
            - math.pi * m * math.sin(math.pi * m / 2)
            + (-1) ** m
        ) / (
            math.pi * m**2
        )
        b_n = 0.0
        result += a_n * np.cos(m * x) + b_n * np.sin(m * x)

    return a_0 + result


def main() -> None:
    x_values = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 1000)

    for N in (5, 20, 100):
        plt.plot(x_values, truncated_series(x_values, N), label=f"{N = }")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("S_N(x)")
    plt.title("Truncated Series")
    plt.show()


if __name__ == "__main__":
    main()
