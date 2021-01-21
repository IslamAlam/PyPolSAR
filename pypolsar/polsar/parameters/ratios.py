import numpy as np
from scipy import signal


def calc_hh_vv_ratio_cpd(scat_matrix):
    from scipy import signal

    n_window = 7
    mean_filter = np.ones((n_window, n_window))
    mean_filter /= sum(mean_filter)
    P_hh_filtered = signal.convolve(
        scat_matrix[:, :, 0] * np.conj(scat_matrix[:, :, 0]),
        mean_filter,
        mode="same",
    )
    P_vv_filtered = signal.convolve(
        scat_matrix[:, :, 3] * np.conj(scat_matrix[:, :, 3]),
        mean_filter,
        mode="same",
    )
    p_hh_vv_ratio = (P_hh_filtered / P_vv_filtered).real
    cpd_multilook = signal.convolve(
        (scat_matrix[:, :, 0] * np.conj(scat_matrix[:, :, 3])),
        mean_filter,
        mode="same",
    )
    cpd_deg = (np.angle(cpd_multilook, deg=True)).real

    return p_hh_vv_ratio, cpd_deg
