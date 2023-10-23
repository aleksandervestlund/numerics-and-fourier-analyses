from matplotlib import pyplot as plt
import numpy as np


def s(x, n):
    a_0, a_n, b_n = fourier_series_a(n)
    return a_0 + sum(
        a_n[i] * np.cos((i + 1) * x) + b_n[i] * np.sin((i + 1) * x)
        for i in range(n)
    )


def fourier_series_a(N):
    a_0 = np.pi / 16
    a_n = np.array(
        [
            np.sin(n * np.pi / 2) / (2 * n)
            + np.cos(n * np.pi / 2) / (np.pi * n**2)
            - np.sin(n * np.pi / 2) / (np.pi * n**2)
            for n in range(1, N + 1)
        ]
    )
    b_n = np.array(
        [
            1 / (np.pi * n**2) * np.sin(n * np.pi / 2)
            - 1 / (2 * n) * np.cos(n * np.pi / 2)
            for n in range(1, N + 1)
        ]
    )
    return a_0, a_n, b_n


def main() -> None:
    for n in (1000,):
        x_values = np.linspace(-3 * np.pi, 3 * np.pi, n)
        plt.plot(x_values, s(x_values, n), label=f"N = {n}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
