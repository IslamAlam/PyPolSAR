__all__ = ["eigen_decomposition"]

import numpy as np


def eigen_decomposition(t_matrix):
    """
    eW -> Eigenvalues w0 < w1 < w2
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
