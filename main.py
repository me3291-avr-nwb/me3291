import numpy as np
import matplotlib.pyplot as plt


def generateBoundary(m):
    for index in range(1, len(m[0]) - 1):
        m[0][index] = 1


def plotMap(U, i, ax=None):
    cp = ax.imshow(U, cmap=plt.get_cmap("hot"), interpolation="spline16")
    ax.set_title("Iteration: {0}".format(i))
    plt.colorbar(cp, ax=ax, fraction=0.046, pad=0.04)


def laplacian(Z, dx):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2


def main():
    size = 15  # matrix size
    T = 1.0
    dt = 0.001  # time step
    n = 200  # iterations
    iter_to_plot = [1, 25, 75, 200]
    row_subplot_num = 2
    col_subplot_num = 2

    dx = 1 / size
    U = np.zeros((size, 1))
    plotted_ptr = 0
    plot_finish = False
    np.set_printoptions(precision=3)

    if row_subplot_num * col_subplot_num != len(iter_to_plot):
        raise Exception("Invalid subplot layout, check row and col nums")

    fig, axes = plt.subplots(col_subplot_num, row_subplot_num, figsize=(8, 8))
    generateBoundary(U)

    for i in range(1, n + 1):
        deltaU = laplacian(U, dx)
        Uc = U[1:-1, 1:-1]
        U[1:-1, 1:-1] = Uc + dt * deltaU

        if not plot_finish:
            if i == iter_to_plot[plotted_ptr]:
                print("Iteration: {0}\n{1}\n\n".format(i, U))
                ax = axes.flatten()
                plotMap(U, i, ax[plotted_ptr])
                plotted_ptr += 1
            if plotted_ptr == len(iter_to_plot):
                plot_finish = True

    plt.show()


if __name__ == "__main__":
    main()
