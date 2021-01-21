import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter

from ..utils.plot import stack_rgb, stack_rgb_linear


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


def plot_pauli_entropy_ani_alpha_vv_hh_phi_diff(
    t_matrix,
    slc_ent_ani_alp_44_dic,
    p_hh_vv_ratio=None,
    cpd_deg=None,
    aspect=None,
    suptitle="Pauli, Entropy, Ani, Alpha, HH-VV Ratio, CPD",
    im_bytescale=True,
    *args,
    **kwargs,
):
    n_images = 6  # len(titles)
    n_plot = 1  # len(plot_type)

    # fig = plt.figure(constrained_layout=False, figsize=(5*n_images, 20))
    fig, axs = plt.subplots(
        1, 6, constrained_layout=True, figsize=(5 * n_images, 10)
    )
    # gs = fig.add_gridspec(nrows=n_plot, ncols=n_images)

    # Pauli
    pauli_vector = np.stack(
        (t_matrix[:, :, 0, 0], t_matrix[:, :, 1, 1], t_matrix[:, :, 2, 2]),
        axis=2,
    )
    if aspect is None:
        aspect = (pauli_vector.shape[1] / pauli_vector.shape[0]) / (1785 / 3140)
    elif aspect is None:
        aspect = (pauli_vector.shape[1] / pauli_vector.shape[0]) / aspect
    # f_ax = fig.add_subplot(gs[0, 0])
    f_ax = axs[0]
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
        k_1 = bytescale(k_sca[:, :, 0], cmin=k_sca.min(), cmax=k_sca.mean())
        k_2 = bytescale(k_sca[:, :, 1], cmin=k_sca.min(), cmax=k_sca.mean())
        k_3 = bytescale(k_sca[:, :, 2], cmin=k_sca.min(), cmax=k_sca.mean())
        rgb = np.stack((k_2, k_3, k_1), axis=2)
        ax_ang = f_ax.imshow(
            rgb,  # vmin=pauli_vec_rgb.min(), vmax=pauli_vec_rgb.max(),
            aspect=aspect,
        )
    else:
        ax_ang = f_ax.imshow(
            stack_rgb_linear(
                r=pauli_vector[:, :, 1],
                g=pauli_vector[:, :, 2],
                b=pauli_vector[:, :, 0],
            ),  # vmin=pauli_vec_rgb.min(), vmax=pauli_vec_rgb.max(),
            aspect=aspect,
            *args,
            **kwargs,
        )
    f_ax.set_title("Pauli", fontsize=24)
    f_ax.axis("off")
    f_ax.set_anchor(anchor="N", share=True)

    for idx, key in enumerate(slc_ent_ani_alp_44_dic):
        title = key
        array = slc_ent_ani_alp_44_dic[key]
        ax_im = axs[idx + 1]
        # ax_im = fig.add_subplot(gs[:, idx+1])
        """        
        if (np.max(array) - np.min(array)) < 1.1:
            ax_cc = ax_im.imshow(array,  cmap='jet',
                                vmin=0.0, vmax=1.)
        elif (np.max(array) - np.min(array)) < 100 and (np.max(array) - np.min(array)) > 10:
            ax_cc = ax_im.imshow(array, cmap='jet',
                                vmin=0.0, vmax=90.)
        else:
        """
        ax_cc = ax_im.imshow(
            array,
            cmap="jet",
            aspect=aspect,
            vmin=np.min(array),
            vmax=np.max(array),
        )

        ax_im.axis("off")
        ax_im.set_title(title, fontsize=24)
        # ax_im.clim(0, 10)
        if ax_cc.get_clim()[1] <= 1:

            ax_cc.set_clim(0, 1)
        elif ax_cc.get_clim()[1] >= 5:
            ax_cc.set_clim(0, 90)
        else:
            ax_cc.set_clim(ax_cc.get_clim()[0], ax_cc.get_clim()[1])

        # fig.colorbar(ax_ang, ax=f_ax2, shrink=0.6)
        cbar = fig.colorbar(
            ax_cc,
            ax=ax_im,  # ticks=[-np.pi*.99, 0, np.pi*.99],
            shrink=1,
            location="bottom",
        )
        cbar.ax.set_xlabel("", rotation=0)
        ax_im.set_anchor(anchor="N", share=True)
    cbar.ax.set_xlabel("$[deg]$", rotation=0)
    # cbar.ax.set_title ("$[deg]$")
    # fig.colorbar(ax_cc, ax=axs[1:3], shrink=0.6, location='bottom')
    # fig.colorbar(ax_cc, ax=axs[3], shrink=0.6, location='bottom')

    if p_hh_vv_ratio is not None:
        # ax_im = fig.add_subplot(gs[:, -2])
        ax_im = axs[-2]
        if aspect is None:
            aspect = (p_hh_vv_ratio.shape[1] / p_hh_vv_ratio.shape[0]) / (
                1785 / 3140
            )
        elif aspect is None:
            aspect = (p_hh_vv_ratio.shape[1] / p_hh_vv_ratio.shape[0]) / aspect
        im_perc = np.percentile(p_hh_vv_ratio, [2, 98])

        ax_cc = ax_im.imshow(
            p_hh_vv_ratio,
            cmap="jet",
            aspect=aspect,
            norm=LogNorm(vmin=im_perc[0], vmax=im_perc[1]),
        )
        ax_im.axis("off")
        ax_im.set_title("HH-VV Power Ratio", fontsize=24)
        cbar = fig.colorbar(
            ax_cc,
            ax=ax_im,  # ticks=[-np.pi*.99, 0, np.pi*.99],
            shrink=1,
            location="bottom",
        )
        cbar.ax.set_xlabel("$[dB]$", rotation=0)
        # cbar.ax.set_title ("$[dB]$")
        ax_im.set_anchor(anchor="N", share=True)
        # fig.colorbar(ax_cc, ax=axs[-2], shrink=0.6, location='bottom')

    if cpd_deg is not None:
        # ax_im = fig.add_subplot(gs[:, -1])
        ax_im = axs[-1]
        if aspect is None:
            aspect = (cpd_deg.shape[1] / cpd_deg.shape[0]) / (1785 / 3140)
        elif aspect is None:
            aspect = (cpd_deg.shape[1] / cpd_deg.shape[0]) / aspect

        im_perc = np.percentile(cpd_deg, [5, 95])

        ax_cc = ax_im.imshow(
            cpd_deg, cmap="jet", aspect=aspect, vmin=im_perc[0], vmax=im_perc[1]
        )
        ax_im.axis("off")
        ax_im.set_title("$\phi_{HH-VV}$", fontsize=24)
        # fig.colorbar(ax_cc, ax=axs[-1], shrink=0.6, location='bottom')
        cbar = fig.colorbar(
            ax_cc,
            ax=ax_im,  # ticks=[-np.pi*.99, 0, np.pi*.99],
            shrink=1,
            location="bottom",
        )
        cbar.ax.set_xlabel("$[deg]$", rotation=0)
        # cbar.ax.set_title ("$[deg]$")
        ax_im.set_anchor(anchor="N", share=True)
    fig.suptitle(suptitle)
    return fig


def plot_pauli_t_entropy_ani_alpha_vv_hh_phi_diff(
    pauli_vector,
    slc_ent_ani_alp_44_dic,
    p_hh_vv_ratio=None,
    cpd_deg=None,
    aspect=None,
    suptitle="Pauli, Entropy, Ani, Alpha, HH-VV Ratio, CPD",
    im_bytescale=True,
    *args,
    **kwargs,
):
    n_images = 6  # len(titles)
    n_plot = 1  # len(plot_type)

    # fig = plt.figure(constrained_layout=False, figsize=(5*n_images, 20))
    fig, axs = plt.subplots(
        1, 6, constrained_layout=True, figsize=(5 * n_images, 10)
    )
    # gs = fig.add_gridspec(nrows=n_plot, ncols=n_images)

    # Pauli

    # pauli_vector = np.stack(
    #    (t_matrix[:, :, 0, 0], t_matrix[:, :, 1, 1], t_matrix[:, :, 2, 2]),
    #    axis=2,
    # )

    pauli_vector = pauli_vector
    if aspect is None:
        aspect = (pauli_vector.shape[1] / pauli_vector.shape[0]) / (1785 / 3140)
    elif aspect is None:
        aspect = (pauli_vector.shape[1] / pauli_vector.shape[0]) / aspect
    # f_ax = fig.add_subplot(gs[0, 0])
    f_ax = axs[0]
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
        k_1 = bytescale(k_sca[:, :, 0], cmin=k_sca.min(), cmax=k_sca.mean())
        k_2 = bytescale(k_sca[:, :, 1], cmin=k_sca.min(), cmax=k_sca.mean())
        k_3 = bytescale(k_sca[:, :, 2], cmin=k_sca.min(), cmax=k_sca.mean())
        rgb = np.stack((k_2, k_3, k_1), axis=2)
        ax_ang = f_ax.imshow(
            rgb,  # vmin=pauli_vec_rgb.min(), vmax=pauli_vec_rgb.max(),
            aspect=aspect,
        )
    else:
        ax_ang = f_ax.imshow(
            stack_rgb_linear(
                r=pauli_vector[:, :, 1],
                g=pauli_vector[:, :, 2],
                b=pauli_vector[:, :, 0],
            ),  # vmin=pauli_vec_rgb.min(), vmax=pauli_vec_rgb.max(),
            aspect=aspect,
            *args,
            **kwargs,
        )
    f_ax.set_title("Pauli", fontsize=24)
    f_ax.axis("off")
    f_ax.set_anchor(anchor="N", share=True)

    for idx, key in enumerate(slc_ent_ani_alp_44_dic):
        title = key
        array = slc_ent_ani_alp_44_dic[key]
        ax_im = axs[idx + 1]
        # ax_im = fig.add_subplot(gs[:, idx+1])
        """        
        if (np.max(array) - np.min(array)) < 1.1:
            ax_cc = ax_im.imshow(array,  cmap='jet',
                                vmin=0.0, vmax=1.)
        elif (np.max(array) - np.min(array)) < 100 and (np.max(array) - np.min(array)) > 10:
            ax_cc = ax_im.imshow(array, cmap='jet',
                                vmin=0.0, vmax=90.)
        else:
        """
        ax_cc = ax_im.imshow(
            array,
            cmap="jet",
            aspect=aspect,
            vmin=np.min(array),
            vmax=np.max(array),
        )

        ax_im.axis("off")
        ax_im.set_title(title, fontsize=24)
        # ax_im.clim(0, 10)
        if ax_cc.get_clim()[1] <= 1:

            ax_cc.set_clim(0, 1)
        elif ax_cc.get_clim()[1] >= 5:
            ax_cc.set_clim(0, 90)
        else:
            ax_cc.set_clim(ax_cc.get_clim()[0], ax_cc.get_clim()[1])

        # fig.colorbar(ax_ang, ax=f_ax2, shrink=0.6)
        cbar = fig.colorbar(
            ax_cc,
            ax=ax_im,  # ticks=[-np.pi*.99, 0, np.pi*.99],
            shrink=1,
            location="bottom",
        )
        cbar.ax.set_xlabel("", rotation=0)
        ax_im.set_anchor(anchor="N", share=True)
    cbar.ax.set_xlabel("$[deg]$", rotation=0)
    # cbar.ax.set_title ("$[deg]$")
    # fig.colorbar(ax_cc, ax=axs[1:3], shrink=0.6, location='bottom')
    # fig.colorbar(ax_cc, ax=axs[3], shrink=0.6, location='bottom')

    if p_hh_vv_ratio is not None:
        # ax_im = fig.add_subplot(gs[:, -2])
        ax_im = axs[-2]
        if aspect is None:
            aspect = (p_hh_vv_ratio.shape[1] / p_hh_vv_ratio.shape[0]) / (
                1785 / 3140
            )
        elif aspect is None:
            aspect = (p_hh_vv_ratio.shape[1] / p_hh_vv_ratio.shape[0]) / aspect
        im_perc = np.percentile(p_hh_vv_ratio, [2, 98])

        ax_cc = ax_im.imshow(
            p_hh_vv_ratio,
            cmap="jet",
            aspect=aspect,
            # norm=LogNorm(vmin=im_perc[0], vmax=im_perc[1]),
            vmin=0.5,
            vmax=1.5,
        )
        ax_im.axis("off")
        ax_im.set_title("HH-VV Power Ratio", fontsize=24)
        cbar = fig.colorbar(
            ax_cc,
            ax=ax_im,  # ticks=[-np.pi*.99, 0, np.pi*.99],
            ticks=np.linspace(start=0.5, stop=1.5, num=5),
            shrink=1,
            location="bottom",
        )
        # cbar.ax.set_xlabel("$[dB]$", rotation=0)
        # cbar.ax.set_title ("$[dB]$")
        ax_im.set_anchor(anchor="N", share=True)
        # fig.colorbar(ax_cc, ax=axs[-2], shrink=0.6, location='bottom')

    if cpd_deg is not None:
        # ax_im = fig.add_subplot(gs[:, -1])
        ax_im = axs[-1]
        if aspect is None:
            aspect = (cpd_deg.shape[1] / cpd_deg.shape[0]) / (1785 / 3140)
        elif aspect is None:
            aspect = (cpd_deg.shape[1] / cpd_deg.shape[0]) / aspect

        im_perc = np.percentile(cpd_deg, [5, 95])

        ax_cc = ax_im.imshow(
            cpd_deg, cmap="jet", aspect=aspect, vmin=im_perc[0], vmax=im_perc[1]
        )
        ax_im.axis("off")
        ax_im.set_title("$\phi_{HH-VV}$", fontsize=24)
        # fig.colorbar(ax_cc, ax=axs[-1], shrink=0.6, location='bottom')
        cbar = fig.colorbar(
            ax_cc,
            ax=ax_im,  # ticks=[-np.pi*.99, 0, np.pi*.99],
            shrink=1,
            location="bottom",
        )
        cbar.ax.set_xlabel("$[deg]$", rotation=0)
        # cbar.ax.set_title ("$[deg]$")
        ax_im.set_anchor(anchor="N", share=True)
    fig.suptitle(suptitle)
    return fig


def plot_eigenvalues_hist_dB(
    array_dict,
    suptitle,
    vmin=2,
    vmax=98,
    aspect=None,
    origin="lower",
    *args,
    **kwargs,
):
    colors = [
        "#003FFF",
        "#E8000B",
        "#03ED3A",
        "#1A1A1A",
    ]  # Blue, Red, Green, Black
    palette = sns.color_palette(colors, 4)
    # sns.palplot(palette)

    n_images = len(array_dict.keys())
    fig = plt.figure(constrained_layout=False, figsize=(4 * n_images, 10))
    gs = fig.add_gridspec(nrows=2, ncols=n_images)
    f_img = {}

    ax_img = {}
    f_hist = fig.add_subplot(gs[1, :])

    for idx, key in enumerate(array_dict):
        legend = key
        array = array_dict[key]
        if aspect is None:
            aspect = array.shape[1] / array.shape[0] * 2

        f_img[idx] = fig.add_subplot(gs[0, idx])
        im_perc = np.percentile(array, [vmin, vmax])

        ax_img[idx] = f_img[idx].imshow(
            array,
            aspect=aspect,
            cmap="jet",
            vmin=0,
            vmax=im_perc[1],
            origin=origin,
        )
        f_img[idx].axis("off")
        f_img[idx].set_title(legend)
        # fig.colorbar(ax_ang, ax=f_ax2, shrink=0.6)
        cbar = fig.colorbar(
            ax_img[idx],
            ax=f_img[idx],  # ticks=[-np.pi*.99, 0, np.pi*.99],
            orientation="vertical",
            shrink=0.8,
            pad=0.1,
        )
        cbar.ax.set_xlabel("", rotation=0)

        f_hist.hist(
            image_dB(array.flatten()),
            color=palette[idx],
            bins="auto",
            histtype="step",
            label=legend,
            density=True,
        )  # , linestyle=(':')
        f_hist.legend()
        # ax = (sns.kdeplot((array.flatten()), shade=False, ax=f_cc, legend=True, label=title))
    f_hist.set_xlabel("$dB$")
    f_hist.set_ylabel("Normalized Frequency")
    fig.suptitle(suptitle)
    # now = datetime.datetime.now()
    # fig.savefig("eigenvalues_{}_{}.tiff".format(title, now.strftime("%Y-%m-%d-%H:%M:%S")))
    return fig


def plot_eigenvalues_hist_dB_2(
    array_dict,
    suptitle,
    array_dict2=None,
    vmin=2,
    vmax=98,
    aspect=None,
    origin="lower",
    *args,
    **kwargs,
):
    colors = [
        "#003FFF",
        "#E8000B",
        "#03ED3A",
        "#1A1A1A",
    ]  # Blue, Red, Green, Black
    palette = sns.color_palette(colors, 4)
    # sns.palplot(palette)

    n_images = len(array_dict.keys())
    fig = plt.figure(constrained_layout=False, figsize=(5 * n_images, 20))
    gs = fig.add_gridspec(nrows=3, ncols=n_images)
    f_img = {}

    ax_img = {}
    f_hist = fig.add_subplot(gs[-1, :])

    for idx, key in enumerate(array_dict):
        legend = key
        array = array_dict[key]
        if aspect is None:
            aspect = array.shape[1] / array.shape[0] * 2

        f_img[idx] = fig.add_subplot(gs[0, idx])
        im_perc = np.percentile(array, [vmin, vmax])

        ax_img[idx] = f_img[idx].imshow(
            array,
            aspect=aspect,
            cmap="jet",
            vmin=0,
            vmax=im_perc[1],
            origin=origin,
        )
        f_img[idx].axis("off")
        f_img[idx].set_title(legend)
        # fig.colorbar(ax_ang, ax=f_ax2, shrink=0.6)
        cbar = fig.colorbar(
            ax_img[idx],
            ax=f_img[idx],  # ticks=[-np.pi*.99, 0, np.pi*.99],
            orientation="vertical",
            shrink=0.8,
            pad=0.1,
        )
        cbar.ax.set_xlabel("", rotation=0)

        f_hist.hist(
            image_dB(array.flatten()),
            color=palette[idx],
            bins="auto",
            histtype="step",
            label=legend,
            density=True,
        )  # , linestyle=(':')
        f_hist.legend()
        # ax = (sns.kdeplot((array.flatten()), shade=False, ax=f_cc, legend=True, label=title))

    if array_dict2 is not None:
        for idx, key in enumerate(array_dict2):
            legend = key
            array = array_dict2[key]
            if aspect is None:
                aspect = array.shape[1] / array.shape[0] * 2

            f_img[idx] = fig.add_subplot(gs[-2, idx])
            im_perc = np.percentile(array, [vmin, vmax])

            ax_img[idx] = f_img[idx].imshow(
                array,
                aspect=aspect,
                cmap="jet",
                vmin=0,
                vmax=im_perc[1],
                origin=origin,
            )
            f_img[idx].axis("off")
            f_img[idx].set_title(legend)
            # fig.colorbar(ax_ang, ax=f_ax2, shrink=0.6)
            cbar = fig.colorbar(
                ax_img[idx],
                ax=f_img[idx],  # ticks=[-np.pi*.99, 0, np.pi*.99],
                orientation="vertical",
                shrink=0.8,
                pad=0.1,
            )
            cbar.ax.set_xlabel("", rotation=0)

            f_hist.hist(
                image_dB(array.flatten()),
                color=palette[idx],
                bins="auto",
                histtype="step",
                label=legend,
                density=True,
                linestyle=(":"),
            )  #
            f_hist.legend()

    f_hist.set_xlabel("$dB$")
    f_hist.set_ylabel("Normalized Frequency")
    fig.suptitle(suptitle)
    # now = datetime.datetime.now()
    # fig.savefig("eigenvalues_{}_{}.tiff".format(title, now.strftime("%Y-%m-%d-%H:%M:%S")))
    return fig


def plot_image_hist(
    array_dict, suptitle, aspect=None, origin="lower", *args, **kwargs
):
    colors = [
        "#003FFF",
        "#E8000B",
        "#03ED3A",
        "#1A1A1A",
    ]  # Blue, Red, Green, Black
    palette = sns.color_palette(colors, 4)

    n_images = len(array_dict.keys())
    fig = plt.figure(constrained_layout=False, figsize=(9 * n_images, 20))
    gs = fig.add_gridspec(nrows=2, ncols=n_images + 1)

    for idx, key in enumerate(array_dict):
        title = key
        array = array_dict[key]
        ax_im = fig.add_subplot(gs[0, idx])
        """        
        if (np.max(array) - np.min(array)) < 1.1:
            ax_cc = ax_im.imshow(array,  cmap='jet',
                                vmin=0.0, vmax=1.)
        elif (np.max(array) - np.min(array)) < 100 and (np.max(array) - np.min(array)) > 10:
            ax_cc = ax_im.imshow(array, cmap='jet',
                                vmin=0.0, vmax=90.)
        else:
        """
        ax_cc = ax_im.imshow(
            array, cmap="jet", vmin=np.min(array), vmax=np.max(array)
        )

        ax_im.axis("off")
        ax_im.set_title(title)
        # ax_im.clim(0, 10)
        ax_cc.set_clim(ax_cc.get_clim()[0], ax_cc.get_clim()[1])

        # fig.colorbar(ax_ang, ax=f_ax2, shrink=0.6)
        cbar = fig.colorbar(
            ax_cc,
            ax=ax_im,  # ticks=[-np.pi*.99, 0, np.pi*.99],
            orientation="vertical",
            shrink=1.0,
            pad=0.05,
        )
        cbar.ax.set_xlabel("", rotation=0)

        ax_hist = fig.add_subplot(gs[1, idx])
        ax_hist.hist(
            (array.ravel()),
            color=palette[idx],
            bins=1000,
            histtype="step",
            label=title,
            density=False,
            weights=100 * np.ones(len(array.ravel())) / len(array.ravel()),
        )  # , linestyle=(':')
        ax_hist.set_xlim(ax_cc.get_clim())
        # Now we format the y-axis to display percentage
        ax_hist.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        fig.suptitle(suptitle)

        # ax = (sns.kdeplot((array.flatten()), shade=False, ax=f_cc, legend=True, label=title))

    return fig


def plot_entropy_an_alpha_hist(
    array_dict, suptitle, aspect=None, origin="lower", *args, **kwargs
):
    colors = [
        "#003FFF",
        "#E8000B",
        "#03ED3A",
        "#1A1A1A",
    ]  # Blue, Red, Green, Black
    palette = sns.color_palette(colors, 4)

    n_images = len(array_dict.keys())
    fig = plt.figure(constrained_layout=False, figsize=(9 * n_images, 20))
    gs = fig.add_gridspec(nrows=3, ncols=n_images)

    for idx, key in enumerate(array_dict):
        title = key
        array = array_dict[key]
        if aspect is None:
            aspect = (array.shape[1] / array.shape[0]) / (1785 / 3140)
        elif aspect is None:
            aspect = (array.shape[1] / array.shape[0]) / aspect

        ax_im = fig.add_subplot(gs[:2, idx])
        """        
        if (np.max(array) - np.min(array)) < 1.1:
            ax_cc = ax_im.imshow(array,  cmap='jet',
                                vmin=0.0, vmax=1.)
        elif (np.max(array) - np.min(array)) < 100 and (np.max(array) - np.min(array)) > 10:
            ax_cc = ax_im.imshow(array, cmap='jet',
                                vmin=0.0, vmax=90.)
        else:
        """
        ax_cc = ax_im.imshow(
            array,
            cmap="jet",
            aspect=aspect,
            vmin=np.min(array),
            vmax=np.max(array),
        )

        ax_im.axis("off")
        ax_im.set_title(title)
        # ax_im.clim(0, 10)
        if ax_cc.get_clim()[1] <= 1:

            ax_cc.set_clim(0, 1)
        elif ax_cc.get_clim()[1] >= 5:
            ax_cc.set_clim(0, 90)
        else:
            ax_cc.set_clim(ax_cc.get_clim()[0], ax_cc.get_clim()[1])

        # fig.colorbar(ax_ang, ax=f_ax2, shrink=0.6)
        cbar = fig.colorbar(
            ax_cc,
            ax=ax_im,  # ticks=[-np.pi*.99, 0, np.pi*.99],
            orientation="vertical",
            shrink=0.7,
        )
        cbar.ax.set_xlabel("", rotation=0)

        ax_hist = fig.add_subplot(gs[2, idx])
        ax_hist.hist(
            (array.ravel()),
            color=palette[idx],
            bins=1000,
            histtype="step",
            label=title,
            density=False,
            weights=100 * np.ones(len(array.ravel())) / len(array.ravel()),
        )  # , linestyle=(':')
        ax_hist.set_xlim(ax_cc.get_clim())
        # Now we format the y-axis to display percentage
        ax_hist.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        fig.suptitle(suptitle)

        # ax = (sns.kdeplot((array.flatten()), shade=False, ax=f_cc, legend=True, label=title))

    return fig


def plot_eigenvalues_entropy_dB(
    array_dict,
    suptitle,
    array_dict2=None,
    array_entropy=None,
    vmin=2,
    vmax=98,
    aspect=None,
    origin="lower",
    *args,
    **kwargs,
):
    colors = [
        "#003FFF",
        "#E8000B",
        "#03ED3A",
        "#1A1A1A",
    ]  # Blue, Red, Green, Black
    palette = sns.color_palette(colors, 4)
    # sns.palplot(palette)

    n_images = len(array_dict.keys()) + 1
    fig = plt.figure(constrained_layout=False, figsize=(5 * n_images, 20))
    gs = fig.add_gridspec(nrows=3, ncols=n_images)
    f_img = {}

    ax_img = {}
    f_hist = fig.add_subplot(gs[-1, 0:4])

    for idx, key in enumerate(array_dict):
        legend = key
        array = array_dict[key]

        if aspect is None:
            aspect = (array.shape[1] / array.shape[0]) / (1785 / 3140)
        elif aspect is None:
            aspect = (array.shape[1] / array.shape[0]) / aspect

        f_img[idx] = fig.add_subplot(gs[0, idx])
        im_perc = np.percentile(array, [vmin, vmax])

        ax_img[idx] = f_img[idx].imshow(
            array,
            aspect=aspect,
            cmap="jet",
            vmin=0,
            vmax=im_perc[1],
            origin=origin,
        )
        f_img[idx].axis("off")
        f_img[idx].set_title(legend)
        # fig.colorbar(ax_ang, ax=f_ax2, shrink=0.6)
        cbar = fig.colorbar(
            ax_img[idx],
            ax=f_img[idx],  # ticks=[-np.pi*.99, 0, np.pi*.99],
            orientation="vertical",
            shrink=0.8,
            pad=0.1,
        )
        cbar.ax.set_xlabel("", rotation=0)

        f_hist.hist(
            image_dB(array.flatten()),
            color=palette[idx],
            bins="auto",
            histtype="step",
            label=legend,
            density=True,
        )  # , linestyle=(':')
        f_hist.legend()
        # ax = (sns.kdeplot((array.flatten()), shade=False, ax=f_cc, legend=True, label=title))

    if array_dict2 is not None:
        for idx, key in enumerate(array_dict2):
            legend = key
            array = array_dict2[key]
            if aspect is None:
                aspect = (array.shape[1] / array.shape[0]) / (1785 / 3140)
            elif aspect is None:
                aspect = (array.shape[1] / array.shape[0]) / aspect

            f_img[idx] = fig.add_subplot(gs[-2, idx])
            im_perc = np.percentile(array, [vmin, vmax])

            ax_img[idx] = f_img[idx].imshow(
                array,
                aspect=aspect,
                cmap="jet",
                vmin=0,
                vmax=im_perc[1],
                origin=origin,
            )
            f_img[idx].axis("off")
            f_img[idx].set_title(legend)
            # fig.colorbar(ax_ang, ax=f_ax2, shrink=0.6)
            cbar = fig.colorbar(
                ax_img[idx],
                ax=f_img[idx],  # ticks=[-np.pi*.99, 0, np.pi*.99],
                orientation="vertical",
                shrink=0.8,
                pad=0.1,
            )
            cbar.ax.set_xlabel("", rotation=0)

            f_hist.hist(
                image_dB(array.flatten()),
                color=palette[idx],
                bins="auto",
                histtype="step",
                label=legend,
                density=True,
                linestyle=(":"),
            )  #
            f_hist.legend()
    if array_entropy is not None:
        f_hist_entropy = fig.add_subplot(gs[-1, -1])
        for idx, key in enumerate(array_entropy):
            legend = key
            array = array_entropy[key]
            if aspect is None:
                aspect = (array.shape[1] / array.shape[0]) / (1785 / 3140)
            elif aspect is None:
                aspect = (array.shape[1] / array.shape[0]) / aspect

            f_img[idx] = fig.add_subplot(gs[idx, -1])

            ax_img[idx] = f_img[idx].imshow(
                array, aspect=aspect, cmap="jet", vmin=0, vmax=1, origin=origin
            )
            f_img[idx].axis("off")
            f_img[idx].set_title(legend)

            f_hist_entropy.hist(
                (array.flatten()),
                color=palette[idx],
                bins="auto",
                histtype="step",
                label=legend,
                density=True,
                linestyle=(":"),
            )  #
            f_hist_entropy.legend()
            cbar = fig.colorbar(
                ax_img[idx],
                ax=f_img[idx],  # ticks=[-np.pi*.99, 0, np.pi*.99],
                orientation="vertical",
                shrink=0.8,
                pad=0.1,
            )
            cbar.ax.set_xlabel("", rotation=0)

    f_hist.set_xlabel("$dB$")
    f_hist.set_ylabel("Normalized Frequency")
    fig.suptitle(suptitle)
    # now = datetime.datetime.now()
    # fig.savefig("eigenvalues_{}_{}.tiff".format(title, now.strftime("%Y-%m-%d-%H:%M:%S")))
    return fig


def image_dB(y):
    return 10 * np.log10(y)


def plot_coherence(
    array_dict, suptitle, aspect=None, origin="lower", *args, **kwargs
):

    colors = [
        "#003FFF",
        "#E8000B",
        "#03ED3A",
        "#1A1A1A",
    ]  # Blue, Red, Green, Black
    palette = sns.color_palette("hls", 8)  # sns.color_palette(colors, 4)

    n_images = len(array_dict.keys())

    fig_im, axs_im = plt.subplots(
        1, n_images, constrained_layout=True, figsize=(5 * n_images, 10)
    )
    fig_hist, axs_hist = plt.subplots(
        1, n_images, constrained_layout=True, figsize=(5 * n_images, 4)
    )

    for idx, key in enumerate(array_dict):
        title = key
        array = array_dict[key]
        if aspect is None:
            aspect = (array.shape[1] / array.shape[0]) / (1785 / 3140)
        elif aspect is None:
            aspect = (array.shape[1] / array.shape[0]) / aspect

        ax_im = axs_im[idx]
        if (np.max(array) - np.min(array)) < 1.1:
            ax_cc = ax_im.imshow(
                array, cmap="jet", vmin=0.0, vmax=1.0, aspect=aspect
            )
        elif (np.max(array) - np.min(array)) < 100 and (
            np.max(array) - np.min(array)
        ) > 10:
            ax_cc = ax_im.imshow(
                array, cmap="jet", vmin=0.0, vmax=90.0, aspect=aspect
            )
        else:
            ax_cc = ax_im.imshow(
                array,
                cmap="jet",
                vmin=np.min(array),
                vmax=np.max(array),
                aspect=aspect,
            )

        ax_im.axis("off")
        ax_im.set_title(title)
        # ax_im.clim(0, 10)
        ax_cc.set_clim(ax_cc.get_clim()[0], ax_cc.get_clim()[1])

        # fig.colorbar(ax_ang, ax=f_ax2, shrink=0.6)
        cbar = fig_im.colorbar(
            ax_cc,
            ax=ax_im,  # ticks=[-np.pi*.99, 0, np.pi*.99],
            shrink=1,
            location="bottom",
        )
        cbar.ax.set_xlabel("", rotation=0)

        ax_hist = axs_hist[idx]
        ax_hist.hist(
            (array.ravel()),
            color=palette[idx],
            bins=1000,
            histtype="step",
            label=title,
            density=False,
            weights=100 * np.ones(len(array.ravel())) / len(array.ravel()),
        )  # , linestyle=(':')
        ax_hist.set_title(title)
        ax_hist.set_xlim(ax_cc.get_clim())
        # Now we format the y-axis to display percentage
        ax_hist.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        # ax_hist.set_aspect(1/aspect)

    fig_im.suptitle(suptitle)
    fig_hist.suptitle(suptitle)

    # ax = (sns.kdeplot((array.flatten()), shade=False, ax=f_cc, legend=True, label=title))

    return fig_im, fig_hist


def plot_grid_hist(data_dic, gamma=0.3, set_lim=False, *args, **kwargs):
    import matplotlib.colors as mcolors
    import pandas as pd

    data = pd.DataFrame(data_dic, columns=data_dic.keys())

    g = sns.PairGrid(data, diag_sharey=False, corner=True)

    g.map_lower(sns.histplot, norm=mcolors.PowerNorm(gamma), *args, **kwargs)
    g.map_diag(sns.histplot, bins="auto")
    # g.map_upper(sns.kdeplot)
    # g.axes[:,:].set_xlim(0,)

    if set_lim:
        for index in range(g.axes[:, :].shape[0]):
            if "ratio" in g.axes[-1, index].get_xlabel().lower():
                g.axes[-1, index].set_xlim(0, 2.5)

            elif "alpha" in g.axes[-1, index].get_xlabel().lower():
                g.axes[-1, index].set_xlim(0, 90)
            else:
                g.axes[-1, index].set_xlim(0, 1)

            if "ratio" in g.axes[index, 0].get_ylabel().lower():
                g.axes[index, 0].set_ylim(0, 2.5)

            elif "alpha" in g.axes[index, 0].get_ylabel().lower():
                g.axes[index, 0].set_ylim(0, 90)

            else:
                g.axes[index, 0].set_ylim(0, 1)
    return g


def plot_hist2d_sb(x, y, gamma=0.3, *args, **kwargs):
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axScatter = plt.subplots(figsize=(10, 10))
    bins = 500
    # the scatter plot:
    # axScatter.hist2d(x.flatten(), y.flatten(),
    #          bins=bins, norm=mcolors.PowerNorm(gamma),  cmap='inferno') # range=np.array([(0, 1), (0, 90)])
    # axScatter.set_aspect(1.)
    sns.histplot(
        x=x.flatten(),
        y=y.flatten(),
        ax=axScatter,
        bins=bins,
        norm=mcolors.PowerNorm(gamma),
        *args,
        **kwargs,
    )

    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1.2, pad=0.0, sharex=axScatter)
    axHisty = divider.append_axes("right", 1.2, pad=0.0, sharey=axScatter)

    # make some labels invisible
    axHistx.xaxis.set_tick_params(labelbottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False)
    axHistx.axis("off")
    axHisty.axis("off")
    # axHistx.his+
    axHistx.hist(x.flatten(), bins=bins, histtype="step")
    axHisty.hist(
        y.flatten(), bins=bins, histtype="step", orientation="horizontal"
    )

    # sns.kdeplot(x=x.flatten(), ax=axHistx, color="r", shade=True)

    # sns.kdeplot(y=y.flatten(), ax=axHisty, color="b", shade=True)

    return fig


def plot_hist2d_sb2(x, y, gamma, *args, **kwargs):
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axScatter = plt.subplots(figsize=(10, 10))
    bins = 500
    # the scatter plot:
    # axScatter.hist2d(x.flatten(), y.flatten(),
    #          bins=bins, norm=mcolors.PowerNorm(gamma),  cmap='inferno') # range=np.array([(0, 1), (0, 90)])
    # axScatter.set_aspect(1.)
    sns.histplot(
        x=x.flatten(),
        y=y.flatten(),
        ax=axScatter,
        bins=bins,
        norm=mcolors.PowerNorm(gamma),
        *args,
        **kwargs,
    )

    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1.2, pad=0.0, sharex=axScatter)
    axHisty = divider.append_axes("right", 1.2, pad=0.0, sharey=axScatter)

    # make some labels invisible
    axHistx.xaxis.set_tick_params(labelbottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False)
    axHistx.axis("off")
    axHisty.axis("off")
    # axHistx.his+
    # axHistx.hist(x.flatten(), bins=bins, histtype="step")
    # axHisty.hist(y.flatten(), bins=bins, histtype="step", orientation='horizontal')

    sns.histplot(x=x.flatten(), ax=axHistx, color="r", shade=True)

    sns.histplot(y=y.flatten(), ax=axHisty, color="b", shade=True)

    return fig
