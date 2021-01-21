__all__ = ["eigen_decomposition"]

import numpy as np
from numba import jit, njit, prange


def eigen_decomposition(t_matrix):
    """
    eW -> Eigenvalues w0 < w1 < w2 ... < wn
    eV -> Eigenvectors
    """
    w = np.zeros(
        (t_matrix.shape[0], t_matrix.shape[1], t_matrix.shape[2]), dtype=float
    )
    v = np.zeros(
        (
            t_matrix.shape[0],
            t_matrix.shape[1],
            t_matrix.shape[2],
            t_matrix.shape[3],
        ),
        dtype=complex,
    )

    for ix, iy in np.ndindex(t_matrix.shape[0:2]):
        w[ix, iy, :], v[ix, iy, :] = np.linalg.eigh(t_matrix[ix, iy, :, :])
    w[w <= 0] = 0.0
    return w, v


"""
@jit
def eigen_decomposition_jit(t_matrix):
    """ """
    eW -> Eigenvalues w0 < w1 < w2
    eV -> Eigenvectors
    """ """
    w = np.zeros(
        (t_matrix.shape[0], t_matrix.shape[1], t_matrix.shape[2]), dtype=float
    )
    v = np.zeros(
        (
            t_matrix.shape[0],
            t_matrix.shape[1],
            t_matrix.shape[2],
            t_matrix.shape[3],
        ),
        dtype=complex,
    )

    for ix, iy in np.ndindex(t_matrix.shape[0:2]):
        w[ix, iy, :], v[ix, iy, :] = np.linalg.eigh(t_matrix[ix, iy, :, :])
    w[w <= 0] = 0.0
    return w, v
"""


@jit(nopython=True, nogil=True, cache=True, parallel=False)
def eigen_decomposition_jit(A):
    """
    eW -> Eigenvalues w0 < w1 < w2 ... < wn
    eV -> Eigenvectors
    """
    # A
    nx = A.shape[0]
    ny = A.shape[1]
    t_size = A.shape[2:]
    w = np.zeros((nx, ny, t_size[0]), dtype=np.float_)
    v = np.zeros((nx, ny, t_size[0], t_size[1],), dtype=np.complex_,)
    for i in range(nx):
        for j in range(ny):

            w[i, j, :], v[i, j, :, :] = np.linalg.eigh(
                A[i, j, :, :]
            )  # , UPLO='U' numba not supported

    return w, v


@jit(nopython=True, nogil=True, cache=True, parallel=True)
def eigen_decomposition_jit_prange(A):
    """
    eW -> Eigenvalues w0 < w1 < w2 ... < wn
    eV -> Eigenvectors
    """
    # A
    nx = A.shape[0]
    ny = A.shape[1]
    # t_size = A.shape[2:4]
    w = np.zeros((nx, ny, A.shape[2]), dtype=np.float_)
    v = np.zeros((nx, ny, A.shape[2], A.shape[3],), dtype=np.complex_,)
    for i in prange(nx):
        for j in prange(ny):

            w[i, j, :], v[i, j, :, :] = np.linalg.eigh(
                A[i, j, :, :]
            )  # , UPLO='U' numba not supported

    return w, v
