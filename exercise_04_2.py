import numpy as np


def g(x: float) -> float:
    return np.cos(np.exp(-x)) ** 2 / 4


def main() -> None:
    x = 0
    x_old = 1

    while abs(x_old - x) > 1e-6:
        x_old = x
        x = g(x)
        print(x)


if __name__ == "__main__":
    main()
