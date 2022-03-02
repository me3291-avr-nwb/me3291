import numpy as np
import matplotlib.pyplot as plt


def generateBoundary(m):
    for index in range(1, len(m[0]) - 1):
        m[0][index] = 1


def plotMap(U, i, ax=None):
    cp = ax.imshow(U, cmap=plt.get_cmap("hot"), interpolation="spline16")
    ax.set_title("Iteration: {0}".format(i))
    plt.colorbar(cp, ax=ax, fraction=0.046, pad=0.04)


def num_scheme(curr, ref, y, x, dt, dx):
    topNode = curr[y+1][x]
    bottomNode = curr[y-1][x]
    leftNode = curr[y][x-1]
    rightNode = curr[y][x+1]
    centerNode = curr[y][x]

    newNode = centerNode + \
        (topNode + leftNode + bottomNode + rightNode - 4*centerNode) * dt/pow(dx, 2)

    ref[y][x] = newNode


def main():
    size = 10  # matrix size
    T = 1.0
    dt = 0.001  # time step
    n = 600  # iterations
    iter_to_plot = [10, 100, 110, 120]
    row_subplot_num = 2
    col_subplot_num = 2
    conv_criteria = 5

    dx = 1 / size
    grid = np.zeros((size+1, size+1))
    gridRef = np.copy(grid)
    plotted_ptr = 0
    plot_finish = False
    # np.set_printoptions(precision=3)

    if row_subplot_num * col_subplot_num != len(iter_to_plot):
        raise Exception("Invalid subplot layout, check row and col nums")

    fig, axes = plt.subplots(col_subplot_num, row_subplot_num, figsize=(8, 8))
    generateBoundary(grid)
    converged = False
    i = 0
    while not converged:
        if i > n+1:
            print("Failed to converge at criteria: {0} dp".format(
                conv_criteria))
            break
        prevGrid = np.copy(grid)

        for y in range(1, size):
            for x in range(1, size):
                num_scheme(grid, gridRef, y, x, dt, dx)

        grid = np.copy(gridRef)
        diff = np.around(np.subtract(grid, prevGrid), conv_criteria)
        converged = not np.any(diff)

        print("Iteration: {0}\n{1}\n\n".format(i, grid))

        if not plot_finish:
            if i == iter_to_plot[plotted_ptr]:
                # print("Iteration: {0}\n{1}\n\n".format(i, U))
                ax = axes.flatten()
                plotMap(grid, i, ax[plotted_ptr])
                plotted_ptr += 1
            if plotted_ptr == len(iter_to_plot):
                plot_finish = True

        i += 1

    plt.show()


if __name__ == "__main__":
    main()
