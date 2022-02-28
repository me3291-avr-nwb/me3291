from msilib.schema import Error
import numpy as np
import matplotlib.pyplot as plt

size = 50
U = np.zeros((size, size))
dx = 1 / size
T = 1.0
dt = 0.01
n = 100
iter_to_plot = [25, 50, 75, 100]


def plot_iter_check():
    try:
        iter_to_plot[-1] < n
        raise ValueError
    except Exception:
        print("Iter to print exceeds iteration number, check n")


def generateBoundary():
    for index in range(1, len(U[0])-1):
        U[0][index] = 1


def plotMap(U, ax=None):
    ax.imshow(U, cmap=plt.get_cmap('hot'), interpolation='spline16')
    ax.set_axis_off()


def laplacian(Z):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright -
            4 * Zcenter) / dx**2


def main():
    plotted_ptr = 0
    plot_finish = False
    np.set_printoptions(precision=3)
    row_len_plot = (len(iter_to_plot) // 2)

    fig, axes = plt.subplots(row_len_plot,
                             row_len_plot, figsize=(8, 8))
    # plot_iter_check()
    generateBoundary()

    for i in range(n):
        deltaU = laplacian(U)
        # print(deltaU)
        Uc = U[1:-1, 1:-1]
        U[1:-1, 1:-1] = Uc + dt * deltaU
        print("Iteration: {0}\n{1}\n\n".format(
            i, U))
        if not plot_finish:
            if i == iter_to_plot[plotted_ptr]:
                ax = axes.flatten()
                plotMap(U, ax[plotted_ptr])
                plotted_ptr += 1
                # print("Iteration: {0}\n{1}\n\n".format(
                #     i, tabulate(U, tablefmt="github")))

                # plt.imshow(U, cmap='hot', interpolation='nearest')
                # plt.contourf(x, -y, U, cmap='hot')
            if plotted_ptr == len(iter_to_plot) - 1:
                plot_finish = True

    plt.imshow(U, cmap='hot', interpolation='spline16')
    plt.show()


main()
