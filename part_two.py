import sys
import numpy as np
import math
from scipy.sparse import diags, linalg, csr_matrix
import matplotlib.pyplot as plt


def plotMap(U, tempx, tempy, ax=None):
    # For 2D plots
    cp = ax.imshow(U, cmap=plt.get_cmap("hot"), interpolation="gaussian")

    # For 3D plots
    # cp = ax.plot_surface(tempx, tempy, U, cmap=plt.get_cmap("hot"))
    # ax.set_zlabel("T(x,y,t)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=9)
    plt.colorbar(cp, ax=ax, fraction=0.046, pad=0.2)


def generateBoundary(m):
    for index in range(1, len(m[0]) - 1):
        m[0][index] = 1


def rhsLinearSys(n, grid):

    # Generating grid with boundary edges
    grid_ref = np.copy(grid)

    # RHS of linear system equation (converted to account for all size matrices)
    toprowCoeff = grid_ref[0][1:-1]

    # rhsCoeff_colVector all zeroes initially
    rhsRows = (n - 2) * (n - 2)
    initialRHS = np.zeros((rhsRows, 1))

    for k in range(1, len(toprowCoeff) + 1):
        for p in range(0, len(toprowCoeff)):
            rhsCoeff = toprowCoeff[p]
            initialRHS[-k] = rhsCoeff

    newRHS = initialRHS

    return newRHS


def qn1b():
    n = 11

    grid = np.zeros((n, n))

    tempx = np.arange(0, 1.1, 0.1)
    tempy = np.flip(np.arange(0, 1.1, 0.1))
    tempx, tempy = np.meshgrid(tempx, tempy)

    generateBoundary(grid)
    # Generating diagonal matrix - 5 point numerical scheme excluding boundary
    A = diags([1, 2, 1], [-1, 0, 1], shape=(n - 2, n - 2), dtype=int).toarray()
    B = diags([-1, 4, -1], [-1, 0, 1], shape=(n - 2, n - 2)).toarray()
    Z = np.zeros_like(B)
    S = -np.identity(n - 2, int)
    Q = np.stack([Z, S, B])
    B = Q[A]
    C = B.swapaxes(1, 2).reshape((n - 2) * (n - 2), -1)

    # Inverting matrix
    # Converting discretized matrix to a sparse type matrix
    sparseC = csr_matrix(C)
    invC = linalg.inv(sparseC)
    print(C.shape)
    # Generating RHS matrix of the system
    rhsSystem = rhsLinearSys(n, grid)

    # Solving for coefficients for the matrix system of Au=b
    centreNodes = invC @ rhsSystem

    # Rearrange to inner boundary size and replace it on the actual grid
    grid[1:-1, 1:-1] = np.flip(centreNodes).reshape(n - 2, -1)

    fig, ax = plt.subplots(
        figsize=(4, 3),
        constrained_layout=True,
        # subplot_kw={
        #     "projection": "3d"
        # },  # Projection argument is only for 3D, turn off if plotting 2D
    )

    plotMap(grid, tempx, tempy, ax)

    # print("Centre Nodes (Coefficients of Laplace equation): \n{0}".format(centreNodes))
    print("Final plate grid:\n{0}".format(grid))
    plt.show()
