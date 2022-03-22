import numpy as np
import matplotlib.pyplot as plt


def generateBoundary(m):
    for index in range(0, len(m[0])):
        m[0][index] = 1


tempx = np.arange(0, 1.1, 0.1)
tempy = np.flip(np.arange(0, 1.1, 0.1))

tempx, tempy = np.meshgrid(tempx, tempy)
# print(tempx)
# print(tempy)


def neumann_boundary(curr, ref, y):
    exLeftNode = curr[y][-3]
    leftNode = curr[y][-2]

    newNode = (4 * leftNode - exLeftNode)/3
    ref[y][-1] = newNode
    # print("Exleft :{0}, left: {1}, curr: {2}".format(
    #     exLeftNode, leftNode, newNode))


def plotMap(U, i, ax=None):
    # cp = ax.imshow(U, cmap=plt.get_cmap("hot"), interpolation="gaussian")
    cp = ax.plot_surface(tempx, tempy, U, cmap=plt.get_cmap(
        "hot"))
    # # ax.invert_yaxis()
    ax.set_title("Iteration: {0}".format(i))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('T(x,y,t)')
    plt.colorbar(cp, ax=ax, fraction=0.046, pad=0.2)


def num_scheme(curr, ref, y, x, dt, dx):
    topNode = curr[y+1][x]
    bottomNode = curr[y-1][x]
    leftNode = curr[y][x-1]
    rightNode = curr[y][x+1]
    centerNode = curr[y][x]

    newNode = centerNode + \
        (topNode + leftNode + bottomNode + rightNode - 4*centerNode) * dt/pow(dx, 2)

    ref[y][x] = newNode


def qn2():
    size = 25  # matrix size
    T = 1.0
    dt = 0.0020  # time step
    n = 500  # iterations
    iter_to_plot = [1, 5, 10, 347]
    row_subplot_num = 2
    col_subplot_num = 2
    conv_criteria = 5

    dx = 1 / size
    grid = np.zeros((size+1, size+1))

    plotted_ptr = 0
    plot_finish = False
    np.set_printoptions(precision=3)

    if row_subplot_num * col_subplot_num != len(iter_to_plot):
        raise Exception("Invalid subplot layout, check row and col nums")

    fig, axes = plt.subplots(col_subplot_num, row_subplot_num, figsize=(
        8, 8), subplot_kw={"projection": "3d"})
    generateBoundary(grid)
    gridRef = np.copy(grid)
    converged = False
    i = 0
    while not converged:
        i += 1
        if i > n:
            print("Failed to converge at criteria: {0} dp".format(
                conv_criteria))
            break
        prevGrid = np.copy(grid)

        for y in range(1, size):
            for x in range(1, size):
                num_scheme(grid, gridRef, y, x, dt, dx)
            neumann_boundary(grid, gridRef, y)
        grid = np.copy(gridRef)
        diff = np.around(np.subtract(grid, prevGrid), conv_criteria)
        converged = not np.any(diff)

        # print("Iteration: {0}\n{1}\n\n".format(i, grid))

        if not plot_finish:
            if i == iter_to_plot[plotted_ptr]:
                print("Iteration: {0}\n{1}\n\n".format(i, grid))
                ax = axes.flatten()
                plotMap(grid, i, ax[plotted_ptr])
                plotted_ptr += 1
            if plotted_ptr == len(iter_to_plot):
                plot_finish = True

    print("\nConverged at {0}".format(i))
    print("Iteration: {0}\n{1}\n\n".format(i, grid))
    plt.show()
