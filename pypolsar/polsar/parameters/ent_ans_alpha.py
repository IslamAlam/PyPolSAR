import numpy as np


def ent_ani_alp(ew, ev):
    p = pseudo_probabilities(ew)
    entropy_h = entropy(p)
    _, alpha_mean = alpha_angles(ev, p)
    anisotropy_a = anisotropy(p)
    return np.stack([entropy_h, alpha_mean, anisotropy_a])


def pseudo_probabilities(ew):
    p = np.zeros(ew.shape, dtype=float)
    for col in range(ew.shape[2]):
        print(col)
        p[:, :, col] = ew[:, :, col] / np.sum(ew, axis=2)
    return p


def entropy(p):
    entropy_h = -np.sum(p * np.log(p) / np.log(p.shape[2]), axis=2)

    return entropy_h


def anisotropy(p):
    anisotropy_a = (p[:, :, 1] - p[:, :, 0]) / (
        p[:, :, 0] + p[:, :, 1]
    )  # (lamda1 - lamda 2) / (lamda1 + lamda2)
    return anisotropy_a


def alpha_angles(ev, p):
    alpha = np.zeros_like(p, dtype=np.float)
    print(ev.shape, p.shape)
    alpha[:, :, 0] = np.arccos(np.abs(ev[:, :, 0, 0]))
    alpha[:, :, 1] = np.arccos(np.abs(ev[:, :, 0, 1]))
    alpha[:, :, 2] = np.arccos(np.abs(ev[:, :, 0, 2]))
    alpha_mean = (
        (alpha[:, :, 0] * p[:, :, 0])
        + (alpha[:, :, 1] * p[:, :, 1])
        + (alpha[:, :, 2] * p[:, :, 2])
    )

    return alpha, alpha_mean
