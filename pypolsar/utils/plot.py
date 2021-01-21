import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

# # Configure plot universal property
plt.rcParams["figure.figsize"] = (16, 16)
plt.rcParams["font.size"] = 16


def image_dB(y):
    y[y == 0] = sys.float_info.epsilon
    return 10 * np.log10(y)


def stack_rgb(r, g, b):
    """    r_perc = np.percentile(np.abs(r), [0, 100])
    g_perc = np.percentile(np.abs(g), [0, 100])
    b_perc = np.percentile(np.abs(b), [0, 100])
    """
    rgb = np.stack((r, g, b), axis=2)  # stacks 3 h x w arrays -> h x w x 3
    # rgb[ rgb == 0 ] = sys.float_info.epsilon
    # rgb = dB(rgb)

    rgb = image_dB(np.abs(rgb))
    # rgb = np.interp(np.abs(rgb), (np.amin(np.abs(rgb)), 3*np.mean(np.abs(rgb))), (0, 1))
    return rgb


def stack_rgb_linear(r, g, b):
    rgb = np.stack((r, g, b), axis=2)  # stacks 3 h x w arrays -> h x w x 3
    # rgb[ rgb == 0 ] = sys.float_info.epsilon
    rgb = np.interp(
        np.abs(rgb), (np.amin(np.abs(rgb)), 3 * np.mean(np.abs(rgb))), (0, 1)
    )
    return rgb


def plot_pauli_lexi_rbg_hist(
    NAME,
    pauli_vector=None,
    lexicographic_vector=None,
    titles=["Pauli feature vector", "Lexicographic feature vector"],
    plot_type=["rbg"],
):
    Path("./results").mkdir(parents=True, exist_ok=True)

    n_images = 1
    n_plot = len(plot_type)
    print(n_images, n_plot)
    if pauli_vector is not None:
        pauli_vec_rgb = stack_rgb(
            r=pauli_vector[:, :, 1],
            g=pauli_vector[:, :, 2],
            b=pauli_vector[:, :, 0],
        )
        n_images += 1
    if lexicographic_vector is not None:
        lexi_vec_rgb = stack_rgb(
            r=lexicographic_vector[:, :, 0],
            g=lexicographic_vector[:, :, 1],
            b=lexicographic_vector[:, :, 2],
        )
        n_images += 1

    fig = plt.figure(constrained_layout=False, figsize=(24, 12 * n_images))
    gs = fig.add_gridspec(nrows=1, ncols=n_images)

    if "rbg" in plot_type:

        f_ax = fig.add_subplot(gs[0, 0])
        print(
            "Image min:",
            pauli_vec_rgb.min(),
            "Image mean:",
            pauli_vec_rgb.mean(),
            ",Image max:",
            pauli_vec_rgb.max(),
        )
        # pauli_vec_rgb_norm = np.interp(decibel_to_linear(np.abs(pauli_vec_rgb)), (decibel_to_linear(np.amin(np.abs(pauli_vec_rgb))), np.max(decibel_to_linear(np.abs(pauli_vec_rgb)))), (0, 1))
        rgb_pauli = stack_rgb_linear(
            r=pauli_vector[:, :, 1],
            g=pauli_vector[:, :, 2],
            b=pauli_vector[:, :, 0],
        )
        rgb_pauli = np.interp(
            rgb_pauli, (np.amin(rgb_pauli), np.amax(rgb_pauli)), (0, 1)
        )
        ax_ang = f_ax.imshow(
            rgb_pauli,  # vmin=pauli_vec_rgb.min(), vmax=pauli_vec_rgb.max(),
            # aspect=pauli_vec_rgb.shape[1] / pauli_vec_rgb.shape[0] * 2,
            origin="lower",
        )
        f_ax.axis("off")

        # Save just the portion _inside_ the second axis's boundaries
        extent = ax_ang.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        fig.savefig("./results/" + NAME + "_pauli.png", bbox_inches="tight")
        f_ax.set_title(titles[0])

        if lexicographic_vector is not None:
            f_ax = fig.add_subplot(gs[0, 1])
            print(
                "Image min:",
                lexi_vec_rgb.min(),
                "Image mean:",
                lexi_vec_rgb.mean(),
                ",Image max:",
                lexi_vec_rgb.max(),
            )
            lexi_vec_rgb_norm = np.interp(
                np.abs(lexi_vec_rgb),
                (np.amin(np.abs(lexi_vec_rgb)), np.max(np.abs(lexi_vec_rgb))),
                (0, 1),
            )

            rgb_lex = stack_rgb_linear(
                r=lexicographic_vector[:, :, 0],
                g=lexicographic_vector[:, :, 1],
                b=lexicographic_vector[:, :, 2],
            )
            ax_ang = f_ax.imshow(
                rgb_lex,  # vmin=10, vmax=200, # vmin=lexi_vec_rgb.min(), vmax=lexi_vec_rgb.max(),
                # aspect=lexi_vec_rgb.shape[1] / lexi_vec_rgb.shape[0] * 2,
                origin="lower",
            )
            f_ax.set_title(titles[1])
            f_ax.axis("off")

    if "hist" in plot_type:
        num_bins = 10000

        f_ax2 = fig.add_subplot(gs[0, -1])

        color = ("r", "g", "b")
        histogram = ["$S_{XX} - S_{YY}$", "$S_{XY}$", "$S_{XX} + S_{YY}$"]
        for i, col in enumerate(color):
            chan = pauli_vec_rgb[:, :, i].ravel()
            # f_ax2.hist(chan, bins='auto', color = col, histtype='step', label=histogram[i], density=True, linestyle=('solid'))
            #    f_cc = fig.add_subplot(gs[0, 1])
            g = sns.kdeplot(
                (chan.flatten()),
                shade=True,
                label=histogram[i],
                color=col,
                ax=f_ax2,
            )
        g.legend()
        g.legend_.set_title("Pauli")

        if lexicographic_vector != None:
            histogram = ["$S_{XX}$", r"$\sqrt{2} S_{XY}$", "$S_{YY}$"]
            for i, col in enumerate(color):
                chan = lexi_vec_rgb[:, :, i].ravel()
                f_ax2.hist(
                    chan,
                    bins="auto",
                    color=col,
                    histtype="step",
                    label=histogram[i],
                    density=True,
                    linestyle=(":"),
                )

        f_ax2.set_xlabel("Value (dB)")
        f_ax2.set_ylabel("Frequency")
        im_perc = np.percentile((pauli_vec_rgb), [0.01, 99.99])
        f_ax2.set_xlim(im_perc[0], im_perc[1])
        f_ax2.set_title("Histogram")

    fig.suptitle(NAME)  # or plt.suptitle('Main title')
    plt.savefig("./results/" + NAME + "_pauli_full.png")

    # sns.plt.show()

    return fig


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image). Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).     If the input image already has dtype uint8, no scaling
    is done.

    :author: taken from scipy.misc (deprecated in scipy, will disappear in version 1.2)

    :param data: image data array
    :type img1: ndarray
    :param cmin: Bias scaling of small values. Default is data.min()
    :type cmin: scalar, optional
    :param cmax: Bias scaling of large values. Default is data.max()
    :type cmax: scalar, optional
    :param high: Scale max value to high. Default is 255.
    :type high: scalar, optional
    :param low: Scale min value to low. Default is 0.
    :type low: scalar, optional
    :returns: The byte-scaled array as ndarray
    """

    if data.dtype == np.uint8:
        return data

    if high < low:
        raise ValueError("`high` should be larger than `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data * 1.0 - cmin) * scale + 0.4999
    bytedata[bytedata > high] = high
    bytedata[bytedata < 0] = 0
    return np.cast[np.uint8](bytedata) + np.cast[np.uint8](low)


def plot_pauli_lexi_rbg_hist_bytescale(
    NAME,
    pauli_vector=None,
    lexicographic_vector=None,
    im_bytescale=False,
    titles=["Pauli feature vector", "Lexicographic feature vector"],
    plot_type=["rbg"],
    *args,
    **kwargs,
):
    n_images = len(titles)
    n_plot = len(plot_type)

    fig = plt.figure(constrained_layout=False, figsize=(12, 24))
    gs = fig.add_gridspec(nrows=n_plot, ncols=n_images)

    if "rbg" in plot_type:
        if pauli_vector is not None:
            f_ax = fig.add_subplot(gs[0, 0])
            pauli_vec_rgb = stack_rgb(
                r=pauli_vector[:, :, 1],
                g=pauli_vector[:, :, 2],
                b=pauli_vector[:, :, 0],
            )
            print(
                "Image min:",
                pauli_vec_rgb.min(),
                "Image mean:",
                pauli_vec_rgb.mean(),
                ",Image max:",
                pauli_vec_rgb.max(),
            )
            # pauli_vec_rgb_norm = np.interp(decibel_to_linear(np.abs(pauli_vec_rgb)), (decibel_to_linear(np.amin(np.abs(pauli_vec_rgb))), np.max(decibel_to_linear(np.abs(pauli_vec_rgb)))), (0, 1))
            if im_bytescale is True:
                k_vec = np.abs(pauli_vector)
                k_sca = k_vec ** 0.7
                """
                vmin=2
                vmax=98
                im_perc = np.percentile(np.abs(k_sca), [vmin, vmax])
                """
                k_1 = bytescale(
                    k_sca[:, :, 0], cmin=k_sca.min(), cmax=k_sca.mean()
                )
                k_2 = bytescale(
                    k_sca[:, :, 1], cmin=k_sca.min(), cmax=k_sca.mean()
                )
                k_3 = bytescale(
                    k_sca[:, :, 2], cmin=k_sca.min(), cmax=k_sca.mean()
                )
                rgb = np.stack((k_2, k_3, k_1), axis=2)
                ax_ang = f_ax.imshow(
                    rgb,  # vmin=pauli_vec_rgb.min(), vmax=pauli_vec_rgb.max(),
                    *args,
                    **kwargs,
                )
            else:
                ax_ang = f_ax.imshow(
                    stack_rgb_linear(
                        r=pauli_vector[:, :, 1],
                        g=pauli_vector[:, :, 2],
                        b=pauli_vector[:, :, 0],
                    ),  # vmin=pauli_vec_rgb.min(), vmax=pauli_vec_rgb.max(),
                    *args,
                    **kwargs,
                )
            f_ax.set_title(titles[0])
            f_ax.axis("off")

        if lexicographic_vector is not None:
            lexi_vec_rgb = stack_rgb(
                r=lexicographic_vector[:, :, 0],
                g=lexicographic_vector[:, :, 1],
                b=lexicographic_vector[:, :, 2],
            )
            f_ax = fig.add_subplot(gs[0, 1])
            print(
                "Image min:",
                lexi_vec_rgb.min(),
                "Image mean:",
                lexi_vec_rgb.mean(),
                ",Image max:",
                lexi_vec_rgb.max(),
            )
            lexi_vec_rgb_norm = np.interp(
                np.abs(lexi_vec_rgb),
                (np.amin(np.abs(lexi_vec_rgb)), np.max(np.abs(lexi_vec_rgb))),
                (0, 1),
            )
            if im_bytescale is True:
                k_vec = np.abs(lexicographic_vector)
                k_sca = k_vec ** 0.7
                vmin = 0
                vmax = 98
                im_perc = np.percentile(np.abs(k_sca), [vmin, vmax])
                k_1 = bytescale(
                    abs(k_sca[:, :, 0]), cmin=im_perc[0], cmax=1 * k_sca.mean()
                )
                k_2 = bytescale(
                    abs(k_sca[:, :, 1]), cmin=im_perc[0], cmax=1 * k_sca.mean()
                )
                k_3 = bytescale(
                    abs(k_sca[:, :, 2]), cmin=im_perc[0], cmax=1 * k_sca.mean()
                )
                rgb = np.stack((k_1, k_2, k_3), axis=2)
                ax_ang = f_ax.imshow(rgb, *args, **kwargs)

            else:
                ax_ang = f_ax.imshow(
                    stack_rgb_linear(
                        r=lexicographic_vector[:, :, 0],
                        g=lexicographic_vector[:, :, 1],
                        b=lexicographic_vector[:, :, 2],
                    ),  # vmin=10, vmax=200, # vmin=lexi_vec_rgb.min(), vmax=lexi_vec_rgb.max(),
                    *args,
                    **kwargs,
                )
            f_ax.set_title(titles[-1])
            f_ax.axis("off")

    if "hist" in plot_type:
        num_bins = 10000
        colors = ["#E8000B", "#03ED3A", "#003FFF", "#1A1A1A"]  # R, G, B,Black
        palette = sns.color_palette(colors, 4)
        f_ax2 = fig.add_subplot(gs[1, :])

        if pauli_vector is not None:

            histogram = [
                "${S_{HH} - S_{VV}}_{(Dihedral)}$",
                "${2 S_{HV}}_{(Volume)}$",
                "${S_{HH} + S_{VV} }_{(Surface)}$",
            ]
            for i, col in zip(range(pauli_vec_rgb.shape[2]), palette):
                chan = pauli_vec_rgb[:, :, i].ravel()
                f_ax2.hist(
                    chan,
                    bins="auto",
                    color=col,
                    histtype="step",
                    label=histogram[i],
                    density=True,
                    linestyle=("solid"),
                )

        if lexicographic_vector is not None:
            histogram = ["$S_{HH}$", "$\sqrt{2} S_{HV}$", "$S_{VV}$"]
            for i, col in zip(range(lexi_vec_rgb.shape[2]), palette):
                chan = lexi_vec_rgb[:, :, i].ravel()
                f_ax2.hist(
                    chan,
                    bins="auto",
                    color=col,
                    histtype="step",
                    label=histogram[i],
                    density=True,
                    linestyle=(":"),
                )

        f_ax2.set_xlabel("Value")
        f_ax2.set_ylabel("Frequency")
        im_perc = np.percentile((pauli_vec_rgb), [0.01, 99.99])
        f_ax2.set_xlim(im_perc[0], im_perc[1])
        f_ax2.set_title("Histogram")

        plt.legend(loc="upper left")

    return fig
