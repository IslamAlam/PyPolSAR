import numpy as np
from scipy.ndimage import filters


def multilook(inputs, window_size=3, output=None, *args, **kwargs):
    """Multi-look or a spatial averaging over an array

    Multidimensional mean filter over an array.

    Parameters
    ----------
    inputs : array_like
        [description]
    window_size : int, optional
        N-look over x-axis(Azimuth) and y-axis (Range) , by default 3 (Azimuth=3, Range=3)
    output : array or dtype, optional
        The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created. , by default None

    Returns
    -------
    mean_filter : ndarray
        Filtered array. Has the same shape as input.

    """

    # Window size specified as a single odd integer (3, 5, 7, …), or an iterable of length image.ndim containing only odd integers (e.g. (1, 5, 5)).
    # multi-look processing or spatial averaging

    inputs = np.asarray(inputs)

    window_size = np.asarray(window_size)
    if window_size.shape == ():
        window_size = np.repeat(window_size.item(), inputs.ndim)

    if np.iscomplexobj(inputs):
        return filters.uniform_filter(
            inputs.real, window_size, output=output, *args, **kwargs
        ) + 1j * filters.uniform_filter(
            inputs.imag, window_size, output=output, *args, **kwargs
        )
    else:
        return filters.uniform_filter(
            inputs.real, window_size, output=output, *args, **kwargs
        )
