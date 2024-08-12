from collections.abc import Callable

import numpy as np


def f(x: float) -> float:
    return x**2


def trapezoidal_rule(
    function: Callable[[float], float], a: float, b: float, m: int
) -> float:
    xs = np.linspace(a, b, m + 1)
    ys = [function(x) for x in xs]
    s = ys[0] + ys[-1] + 2 * sum(ys[1:-1])
    return s * (b - a) / (2 * m)


def main() -> None:
    print(trapezoidal_rule(f, 0, 1, 2))
    print(trapezoidal_rule(f, 0, 1, 4))
    print(trapezoidal_rule(f, 0, 1, 10))
    print(trapezoidal_rule(f, 0, 1, 1_000_000))


if __name__ == "__main__":
    main()
