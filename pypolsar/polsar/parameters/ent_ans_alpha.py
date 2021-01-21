import numpy as np


def ent_ani_alp(ew, ev):
    p = pseudo_probabilities(ew)
    entropy_h = entropy(p)
    _, alpha_mean = alpha_angles(ev, p)
    anisotropy_a = anisotropy(p)
    print(
        "Eigenvalues shape: {0}\t Eigenvectors shape: {1}".format(
            ew.shape, ev.shape
        )
    )
    return np.stack([entropy_h, anisotropy_a, alpha_mean])


def pseudo_probabilities(ew):
    p = np.zeros(ew.shape, dtype=float)
    p = ew[:, :, :] / np.sum(ew, axis=2)[:, :, np.newaxis]

    assert np.allclose(
        np.sum(p[:, :, :], axis=2), 1
    ), "Pseudo Probabilities should be 1 along axis 2"
    return p


def entropy(p):
    entropy_h = -np.sum(p * np.log(p) / np.log(p.shape[2]), axis=2)
    assert (
        np.max(entropy_h) <= 1 and np.min(entropy_h) >= 0
    ), "Entopry should be in range 0 < H < 1"
    return entropy_h


def anisotropy(p):
    anisotropy_a = (p[:, :, -2] - p[:, :, -3]) / (
        p[:, :, -2] + p[:, :, -3]
    )  # (lamda1 - lamda 2) / (lamda1 + lamda2)
    return anisotropy_a


def alpha_angles(ev, p):
    alpha = np.zeros_like(p, dtype=np.float)
    # print(ev.shape, p.shape)
    assert np.allclose(np.sum(ev[:, :, 0, :].imag, axis=2), 0), (
        "Eigenvector" "s first element imag part should be almost zero."
    )
    alpha[:, :, :] = np.arccos(
        np.abs(ev[:, :, 0, :])
    )  # cos of each top element of each vector

    alpha_mean = np.sum(alpha * p, axis=2)

    return alpha, alpha_mean
