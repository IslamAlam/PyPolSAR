__all__ = ["lex_vec", "pauli_vec"]

import warnings

import numpy as np
from scipy import signal

# class lex_vec(s_matrix):

#    def __init__(self, *args, **kwargs):
#        self.name = "Lexicographic Scattering Vector"
#        self.allowed_ndim = [4]
#        self.blockprocess = True


def lex_vec(s_matrix, *args, **kwargs):
    """
    𝑘⃗_3𝐿 =[ 𝑆_𝐻𝐻, √2 𝑆_𝐻𝑉, 𝑆_𝑉𝑉 ]𝑇
    """
    if not np.array_equal(s_matrix[:, :, 1], s_matrix[:, :, 2]):
        """
        3.2.2 BISTATIC SCATTERING CASE
        4-D Lexicographic feature vector
        """
        print("4-D Lexicographic feature vector")
        lex_vector = np.zeros(
            (s_matrix.shape[0], s_matrix.shape[1], 4), dtype=complex
        )
        lex_vector[:, :, 0] = s_matrix[:, :, 0]
        lex_vector[:, :, 1] = s_matrix[:, :, 1]
        lex_vector[:, :, 2] = s_matrix[:, :, 2]
        lex_vector[:, :, 3] = s_matrix[:, :, 3]

    else:
        """
        3.2.3 MONOSTATIC BACKSCATTERING CASE
        3-D Lexicographic feature vector
        """
        print("3-D Lexicographic feature vector")
        lex_vector = np.zeros(
            (s_matrix.shape[0], s_matrix.shape[1], 3), dtype=complex
        )
        lex_vector[:, :, 0] = s_matrix[:, :, 0]
        lex_vector[:, :, 1] = np.sqrt(2) * s_matrix[:, :, 1]
        lex_vector[:, :, 2] = s_matrix[:, :, 3]
    return lex_vector


# class pauli_vec(*args, **kwargs):

# def __init__(*args, **kwargs):
#    self.name = "Pauli Scattering Vector"
#    self.allowed_ndim = [4]


def pauli_vec(s_matrix, *args, **kwargs):
    """
    Pauli scattering vector
    𝑘⃗_3𝑃 = 1/√2 [ 𝑆_𝐻𝐻+𝑆_𝑉𝑉, 𝑆_𝐻𝐻−𝑆_𝑉𝑉, 2𝑆_𝐻𝑉 ]𝑇
    """
    if not np.array_equal(s_matrix[:, :, 1], s_matrix[:, :, 2]):
        """
        3.2.2 BISTATIC SCATTERING CASE
        4-D Pauli feature vector
        𝑘⃗_4𝑃 = 
                ⎡   (S₁₁ + S₂₂)   ⎤
                ⎢                 ⎥
        1/√2 .  ⎢   (S₁₁ - S₂₂)   ⎥
                ⎢                 ⎥
                ⎢   (S₁₂ + S₂₁)   ⎥
                ⎢                 ⎥
                ⎣   ⅈ⋅(S₁₂ - S₂₁)  ⎦

        """
        print("4-D Pauli feature vector")
        pauli_vector = np.zeros(
            (s_matrix.shape[0], s_matrix.shape[1], 4), dtype=complex
        )
        pauli_vector[:, :, 0] = s_matrix[:, :, 0] + s_matrix[:, :, 3]
        pauli_vector[:, :, 1] = s_matrix[:, :, 0] - s_matrix[:, :, 3]
        pauli_vector[:, :, 2] = s_matrix[:, :, 1] + s_matrix[:, :, 2]
        pauli_vector[:, :, 3] = 1j * (s_matrix[:, :, 1] - s_matrix[:, :, 2])
        pauli_vector = pauli_vector / np.sqrt(2)
    else:
        """
        3.2.3 MONOSTATIC BACKSCATTERING CASE
        3-D Pauli feature vector
        𝑘⃗_3𝑃 = 
                ⎡   (Sxx + Syy)   ⎤
                ⎢                 ⎥
        1/√2 .  ⎢   (Sxx - Syy)   ⎥
                ⎢                 ⎥
                ⎣   2 . (Sxy )    ⎦
        """
        print("3-D Pauli feature vector")
        pauli_vector = np.zeros(
            (s_matrix.shape[0], s_matrix.shape[1], 3), dtype=complex
        )
        pauli_vector[:, :, 0] = s_matrix[:, :, 0] + s_matrix[:, :, 3]
        pauli_vector[:, :, 1] = s_matrix[:, :, 0] - s_matrix[:, :, 3]
        pauli_vector[:, :, 2] = 2 * s_matrix[:, :, 1]
        pauli_vector = pauli_vector / np.sqrt(2)
    return pauli_vector


def polarimetric_coherency_t_matrix(k_vector, kernel_size=None):
    """
    Calculate the Coherency Matrix [T] and visualize the elements T11, T22, T33 as powers and the elements T13, T23, T12 as powers and their phases. Plus calculate the histograms for everything
    3.5.2 ..................................................... pg 83
    Polarimetric coherency matrix [T 3x3]

    """

    # k_vector = pauli_scattering_vector(S_Matrix)
    pol_coherency_matrix = np.zeros(
        (
            k_vector.shape[0],
            k_vector.shape[1],
            k_vector.shape[2],
            k_vector.shape[2],
        ),
        dtype=complex,
    )

    pol_mat_temp = np.zeros(
        (k_vector.shape[2], k_vector.shape[2]), dtype=complex
    )

    k_vec_temp = np.zeros((k_vector.shape[2], 1), dtype=complex)

    for ix, iy in np.ndindex(k_vector.shape[0:2]):
        k_vec_temp = k_vector[ix, iy, :].reshape(-1, 1)
        pol_mat_temp = np.dot(k_vec_temp, k_vec_temp.T.conjugate())
        pol_coherency_matrix[ix, iy, :, :] = pol_mat_temp

    if kernel_size is None:
        kernel_size = 7
    mean_filter = np.ones((kernel_size, kernel_size))
    mean_filter /= sum(mean_filter)
    pol_coherency_matrix_filtered = np.zeros_like(pol_coherency_matrix)
    for ix, iy in np.ndindex(pol_coherency_matrix.shape[2:]):
        pol_coherency_matrix_filtered[:, :, ix, iy] = signal.convolve(
            pol_coherency_matrix[:, :, ix, iy], mean_filter, mode="same"
        )
    return pol_coherency_matrix, pol_coherency_matrix_filtered


def polarimetric_covariance_c_matrix(omega_vector):
    """
    Calculate the Covariance Matrix [C] and visualize the elements C1, C22, C33 as powers and the elements C13, C23, C12 as powers and their phases. Plus calculate the histograms for everything
    3.5.3 .................................................... 84
    Polarimetric covariance matrix [𝐶 3x3] or [C 4x4]
    """
    # omega_vector = lexicographic_scattering_vector(S_Matrix)
    pol_covariance_matrix = np.zeros(
        (
            omega_vector.shape[0],
            omega_vector.shape[1],
            omega_vector.shape[2],
            omega_vector.shape[2],
        ),
        dtype=complex,
    )

    pol_mat_temp = np.zeros(
        (omega_vector.shape[2], omega_vector.shape[2]), dtype=complex
    )

    omega_vec_temp = np.zeros((omega_vector.shape[2], 1), dtype=complex)
    for ix, iy in np.ndindex(omega_vector.shape[0:2]):
        omega_vec_temp = omega_vector[ix, iy, :].reshape(-1, 1)
        pol_mat_temp = np.dot(omega_vec_temp, omega_vec_temp.T.conjugate())
        pol_covariance_matrix[ix, iy, :, :] = pol_mat_temp

    n_window = 7
    mean_filter = np.ones((n_window, n_window))
    mean_filter /= sum(mean_filter)
    pol_covariance_matrix_filtered = np.zeros_like(pol_covariance_matrix)
    for ix, iy in np.ndindex(pol_covariance_matrix.shape[2:]):
        # print(i)
        pol_covariance_matrix_filtered[:, :, ix, iy] = signal.convolve(
            pol_covariance_matrix[:, :, ix, iy], mean_filter, mode="same"
        )
    return pol_covariance_matrix, pol_covariance_matrix_filtered


def mean_filter(im, mysize=None, noise=None):
    im = np.asarray(im)
    if mysize is None:
        mysize = [3] * im.ndim
    mysize = np.asarray(mysize)
    if mysize.shape == ():
        mysize = np.repeat(mysize.item(), im.ndim)
    mean_filter = np.ones(mysize)
    mean_filter /= sum(mean_filter)

    out = signal.convolve(im, mean_filter, mode="same")
    return out
