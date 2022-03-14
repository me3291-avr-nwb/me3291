import sys
import numpy as np
import math
from scipy.sparse import diags, linalg, csr_matrix

np.set_printoptions(threshold=sys.maxsize, precision=3)


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
    n = 5

    grid = np.zeros((n, n))
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
    sparseC = csr_matrix(C)  # Converting discretized matrix to a sparse type matrix
    invC = linalg.inv(sparseC)

    # Generating RHS matrix of the system
    rhsSystem = rhsLinearSys(n, grid)

    # Solving for coefficients for the matrix system of Au=b
    centreNodes = invC @ rhsSystem

    # Rearrange to inner boundary size and replace it on the actual grid
    grid[1:-1, 1:-1] = np.flip(centreNodes).reshape(n - 2, -1)

    print("Centre Nodes (Coefficients of Laplace equation): \n{0}".format(centreNodes))
    print("Final plate grid:\n{0}".format(grid))
