import sys
import numpy as np
import math
from scipy.sparse import diags, linalg, csr_matrix

np.set_printoptions(threshold=sys.maxsize, precision=3)


def boundary(A, x, y):
    """
    Set up boundary conditions

    Input:
     - A: Matrix to set boundaries on
     - x: Array where x[i] = hx*i, x[last_element] = Lx
     - y: Eqivalent array for y

    Output:
     - A is initialized in-place (when this method returns)
    """

    # Boundaries implemented (condensator with plates at y={0,Lx}, DeltaV = 200):
    # A(x,0)  =  100*sin(2*pi*x/Lx)
    # A(x,Ly) = -100*sin(2*pi*x/Lx)
    # A(0,y)  = 0
    # A(Lx,y) = 0

    Nx = A.shape[1]
    Ny = A.shape[0]
    Lx = x[Nx-1]  # They *SHOULD* have same sizes!
    Ly = x[Nx-1]

    A[:, 0] = 0
    A[:, Nx-1] = 0
    A[0, :] = 1
    A[Ny-1, :] = 0
    print(A)

# Main program


# Input parameters
Nx = 2
Ny = 2
maxiter = 50

x = np.linspace(0, 1, num=Nx+2)  # Also include edges
y = np.linspace(0, 1, num=Ny+2)
A = np.zeros((Nx+2, Ny+2))

def generateBoundary(m):
    for index in range(1, len(m[0]) - 1):
        m[0][index] = 1

def rhsLinearSys(n):

    # Generating grid with boundary edges
    grid = np.zeros((n, n))
    generateBoundary(grid)
    grid_ref = np.copy(grid)

    # RHS of linear system equation (converted to account for all size matrices)
    toprowCoeff = grid_ref[0][1:-1]

    #rhsCoeff_colVector all zeroes initially
    rhsRows = (n-2)*(n-2)
    initialRHS = np.zeros((rhsRows, 1))

    for k in range(1, len(toprowCoeff)+1):
        for p in range (0, len(toprowCoeff)):
            rhsCoeff = toprowCoeff[p]
            initialRHS[-k] = rhsCoeff

    newRHS = initialRHS

    return (newRHS)



def qn1b():
    n = 4
    # Generating diagonal matrix - 5 point numerical scheme excluding boundary
    A = diags([1, 2, 1], [-1, 0, 1], shape=(n-2, n-2), dtype=int).toarray()
    B = diags([-1, 4, -1], [-1, 0, 1], shape=(n-2, n-2)).toarray()
    Z = np.zeros_like(B)
    S = -np.identity(n-2, int)
    Q = np.stack([Z, S, B])
    B = Q[A]
    C = B.swapaxes(1, 2).reshape((n-2)*(n-2), -1)

    # Inverting matrix
    sparseC = csr_matrix(C) # Converting discretized matrix to a sparse type matrix
    invC = linalg.inv(sparseC)

    # Generating RHS matrix of the system
    rhsSystem = rhsLinearSys(n)


    # Solving for coefficients for the matrix system of Au=b
    centreNodes = invC @ rhsSystem

    print("Centre Nodes (Coefficients of Laplace equation): \n")
    print(centreNodes)
