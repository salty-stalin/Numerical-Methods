# This function should safely invert a matrix, by checking its condition number
# and determinant
import numpy as np
import numpy.linalg

def TestInverse(A, tol):
    """
    Invert the matrix, assuming the condition number is smaller than the
    tolerance and the determinant is bigger than the inverse of the
    tolerance.

    Parameters
    ----------

    A : array of float
        matrix to be inverted
    tol : float
        tolerance

    Returns
    -------

    Inverse A : array of float
        inverse of A

    Notes
    -----

    This function relies heavily on numpy to do all real calculations.
    """


    invtol = 1.0 / tol

    assert(np.linalg.cond(A) < tol), 'The condition number is too big!'
    assert(np.linalg.det(A) > invtol), 'The determinant of the matrix is too small!'

    InverseA = np.linalg.inv(A)

    return InverseA
A = np.array([[5,3],[6,5]])
tol =14
invA = TestInverse(A, tol)
print(invA)
print(np.linalg.cond(A))
