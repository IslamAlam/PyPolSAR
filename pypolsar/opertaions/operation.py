import numpy as np

from ..filters.spatial import multilook


def coherence(in1, in2, window_size=7, *args, **kwargs):
    """[summary]

    calc coherence between two complex arrays

    Parameters
    ----------
    in1 : [type]
        [description]
    in2 : [type]
        [description]
    window_size : int, optional
        [description], by default 7
    """
    coh = multilook(in1 * np.conj(in2), window_size, *args, **kwargs) / (
        np.sqrt(
            multilook(in1 * np.conj(in1), window_size, *args, **kwargs)
            * multilook(in2 * np.conj(in2), window_size, *args, **kwargs)
        )
    )

    return coh  # np.clip(np.nan_to_num(coh), 0.0, 1.0)


def phase_diff(in1, in2, window_size=7, deg=False, *args, **kwargs):
    phase_dif = np.angle(
        multilook(in1 * np.conj(in2), window_size, *args, **kwargs), deg,
    )

    return phase_dif


def power_ratio(in1, in2, window_size=7, *args, **kwargs):
    p1 = multilook((in1) * np.conj(in1), window_size, *args, **kwargs)
    p2 = multilook((in2) * np.conj(in2), window_size, *args, **kwargs)
    p_ratio = (p1 / p2).real
    return p_ratio
