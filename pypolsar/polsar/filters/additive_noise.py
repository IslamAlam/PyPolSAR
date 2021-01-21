"""

Hajnsek I., Papathanassiou K.P. & Cloude S.R.
(2001), “Removal of Additive Noise in Polarimetric
Eigenvalue Processing”, Proc. IGARSS, Sidney,
Australia, pp.2778–2780.
"""

import numpy as np


def additive_noise(c_t_matrix_44):
    """
    Inputs: T or C Matrix 4x4
    Output: T or C Matrix 3x3 (Noise Removed)
    """
    from ..decomposition.eigen import eigen_decomposition

    assert c_t_matrix_44.shape[-1] == 4
    w_4, v_4 = eigen_decomposition(c_t_matrix_44)
    noise = w_4[:, :, 0]
    noise[noise < 0] = 0
    c_t_matrix_33 = np.zeros_like(c_t_matrix_44[:, :, :3, :3])
    c_t_matrix_33 = c_t_matrix_44[:, :, :3, :3]
    # c_t_matrix_33_diagonal = c_t_matrix_33.diagonal( axis1=-2, axis2=-1) - np.repeat(noise[:,:, np.newaxis], 3, axis=2)

    """
    c_t_matrix_33[:, :, 0, 0] = c_t_matrix_33[:, :, 0, 0] - c_t_matrix_33_diagonal[:, :, 0]
    c_t_matrix_33[:, :, 1, 1] = c_t_matrix_33[:, :, 1, 1] - c_t_matrix_33_diagonal[:, :, 1]
    c_t_matrix_33[:, :, 2, 2] = c_t_matrix_33[:, :, 2, 2] - c_t_matrix_33_diagonal[:, :, 2]
    """
    print("Additive T - 2*Noise")
    c_t_matrix_33[:, :, 0, 0] = c_t_matrix_33[:, :, 0, 0] - 2 * noise
    c_t_matrix_33[:, :, 1, 1] = c_t_matrix_33[:, :, 1, 1] - 2 * noise
    c_t_matrix_33[:, :, 2, 2] = c_t_matrix_33[:, :, 2, 2] - 2 * noise

    return c_t_matrix_33
