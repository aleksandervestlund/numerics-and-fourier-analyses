import numpy as np


def main() -> None:
    # 4b) y' = f(t, y) = 2y/t^2
    f = lambda t, y: 2 / t**2 * y

    t0, tend = 1, 2
    y0 = 1
    N = 10

    y = np.zeros(N + 1)
    y[0] = y0

    # 4a) Bug 3: t was equal to only zeros, except for the first element
    t = np.linspace(t0, tend, N + 1)

    # 4a) Bug 1: a was undefined
    t[0] = t0

    # 4a) Bug 2: h was undefined
    h = (tend - t0) / N

    # Runge-Kutta for s = 2
    for n in range(N):
        k1 = f(t[n], y[n])
        k2 = f(t[n] + 0.5 * h, y[n] + 0.5 * h * k1)
        y[n + 1] = y[n] + h * k2

    print("t=", t)
    print("y=", y)

    # 4d)
    print(f"y_0 = {y[0]}, y_1 = {y[1]}, y_2 = {y[2]}")


if __name__ == "__main__":
    main()
