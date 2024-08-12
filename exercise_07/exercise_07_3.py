from collections.abc import Callable
from functools import partial

import numpy as np
from matplotlib import pyplot as plt


def dydt(
    y: np.ndarray, k_1: float, k_2: float, m_1: float, m_2: float
) -> np.ndarray:
    return np.array(
        [
            y[1],
            -k_1 / m_1 * y[0] + k_2 / m_1 * (y[2] - y[0]),
            y[3],
            -k_2 / m_2 * (y[2] - y[0]),
        ]
    )


def euler(
    func: Callable[[np.ndarray], np.ndarray],
    t_0: float,
    y_0: np.ndarray,
    t_n: float,
    n: int,
) -> list[np.ndarray]:
    h = (t_n - t_0) / n
    y = [y_0]
    for i in range(n):
        y.append(y[i] + h * func(y[i]))

    return y


def heun(
    func: Callable[[np.ndarray], np.ndarray],
    t_0: float,
    y_0: np.ndarray,
    t_n: float,
    n: int,
) -> list[np.ndarray]:
    h = (t_n - t_0) / n
    y = [y_0]
    for i in range(n):
        k_1 = func(y[i])
        y_1_euler = y[i] + h * k_1
        k_2 = func(y_1_euler)
        y.append(y[i] + h * (k_1 + k_2) / 2)

    return y


def plot_euler_and_heun(
    t_0: float,
    t_n: float,
    h: float,
    k_1: float,
    k_2: float,
    m_1: float,
    m_2: float,
    y_0: np.ndarray,
) -> None:
    dy = partial(dydt, k_1=k_1, k_2=k_2, m_1=m_1, m_2=m_2)
    n = int((t_n - t_0) / h)

    # Extracting the columns of the returned arrays
    u_1, du_1, v_1, dv_1 = np.array(euler(dy, t_0, y_0, t_n, n)).transpose()
    u_2, du_2, v_2, dv_2 = np.array(heun(dy, t_0, y_0, t_n, n)).transpose()

    e_1 = (
        m_1 * du_1**2 + m_2 * dv_1**2 + k_1 * u_1**2 + k_2 * (v_1 - u_1) ** 2
    ) / 2
    e_2 = (
        m_1 * du_2**2 + m_2 * dv_2**2 + k_1 * u_2**2 + k_2 * (v_2 - u_2) ** 2
    ) / 2

    t = np.linspace(t_0, t_n, n + 1)
    mathematicians = ("Euler", "Heun")

    figure, axis = plt.subplots(3, 2, figsize=(7, 7))
    figure.tight_layout(pad=3.0)

    for i, elem in enumerate((u_1, u_2, v_1, v_2)):
        x, y = divmod(i, 2)
        axis[x, y].plot(t, elem)
        axis[x, y].grid()
        axis[x, y].set_ylim(-0.5, 0.5)
        axis[x, y].set_title(
            f"{mathematicians[y]} - {'u' if i < 2 else 'v'}(t)"
        )
        axis[x, y].set_xlabel("Time")
        axis[x, y].set_ylabel("Displacement")

    for i, elem in enumerate((e_1, e_2)):
        axis[2, i].plot(t, elem)
        axis[2, i].grid()
        axis[2, i].set_ylim(7, 10)
        axis[2, i].set_title(f"{mathematicians[i]} - E(t)")
        axis[2, i].set_xlabel("Time")
        axis[2, i].set_ylabel("Energy")

    plt.show()


def main() -> None:
    for h in (0.1, 0.01, 0.001):
        plot_euler_and_heun(
            t_0=0.0,
            t_n=3.0,
            h=h,
            k_1=100.0,
            k_2=200.0,
            m_1=10.0,
            m_2=5.0,
            y_0=np.array([0.0, 1.0, 0.0, 1.0]),
        )


if __name__ == "__main__":
    main()
