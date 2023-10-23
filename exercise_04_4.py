import numpy as np


def main() -> None:
    x = 0.5
    err = abs(np.cos(x) + np.log(x))

    # Newtons method, for cos(x)+ln(x)=0
    while err > 1e-6:
        dx = (np.cos(x) + np.log(x)) / (-np.sin(x) + 1 / x)
        x -= dx
        err = abs(dx)
        print(x)


if __name__ == "__main__":
    main()
