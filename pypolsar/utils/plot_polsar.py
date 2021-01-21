import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter

from .plot import image_dB

plt.rcParams["image.origin"] = "lower"

# Save a palette to a variable:\

colors = ["#003FFF", "#E8000B", "#03ED3A", "#1A1A1A"]  # B, R, G, Black
colors = ["#E8000B", "#03ED3A", "#003FFF", "#1A1A1A"]  # R, G, B,Black
colors = ["#003FFF", "#E8000B", "#03ED3A", "#1A1A1A"]  # Blue, Red, Green, Black

palette = sns.color_palette(colors, 4)


def plot_t_c_matrix(
    c_t_matrix,
    title="T",
    vmin=2,
    vmax=98,
    aspect=None,
    origin="lower",
    *args,
    **kwargs,
):
    C_T_11 = c_t_matrix[:, :, 0, 0]
    im_perc = np.percentile(np.abs(C_T_11), [vmin, vmax])
    if aspect is None:
        aspect = c_t_matrix.shape[1] / c_t_matrix.shape[0] * 2

    def plot_sections(c_t_matrix):

        fig, axs = plt.subplots(3, 4, constrained_layout=True)
        for idx, ax in zip(range(len(c_t_matrix)), axs[0, :3].flat):
            # print(idx)
            c_idx_abs = np.abs(c_t_matrix[:, :, idx, idx])
            pcm = ax.imshow(
                c_idx_abs,
                interpolation="hamming",
                norm=LogNorm(
                    vmin=im_perc[0], vmax=im_perc[1]
                ),  # vmin=.9*c11_db.min() , vmax= 1/3*c11_db.max(),
                cmap="gray",
                aspect=aspect,
                origin=origin,
                *args,
                **kwargs,
            )
            ax.axis("off")
            ax.axis("off")
            ax.set_title("${0}_{{{1}{1}}}$".format(title, idx + 1))

            # plot hisogram
            ax_hist = axs[0, -1]
            ax_hist.hist(
                image_dB(c_idx_abs.ravel()),
                bins="auto",
                color=palette[idx],
                histtype="step",
                label="${0}_{{{1}{1}}}$".format(title, idx + 1),
                density=True,
                linestyle=("solid"),
            )
            ax_hist.legend(loc="upper left")  # , bbox_to_anchor=(0.5, 1.05),
            #  ncol=3, fancybox=True, shadow=True)
            ax_hist.set_title("Histogram")

        cbar_1 = fig.colorbar(pcm, ax=axs[0, :3], shrink=0.6, location="left")
        cbar_1.set_label("$dB $ ", rotation=0)

        # plot of images
        il_lower = np.tril_indices(c_t_matrix.shape[3], -1)
        for idx, nx, ny, ax in zip(
            range(4), il_lower[0], il_lower[1], axs[1, :3].flat
        ):
            #
            # print(nx, ny)
            c_idx_abs = np.abs(c_t_matrix[:, :, nx, ny])
            c_idx_angle = np.angle(c_t_matrix[:, :, nx, ny])

            pcm = ax.imshow(
                c_idx_abs,
                interpolation="hamming",
                norm=LogNorm(
                    vmin=im_perc[0], vmax=im_perc[1]
                ),  # vmin=.9*c11_db.min() , vmax= 1/3*c11_db.max(),
                cmap="gray",
                aspect=aspect,
                origin=origin,
                *args,
                **kwargs,
            )
            ax.axis("off")
            ax.axis("off")
            ax.set_title("${0}_{{{1}{2}}}$".format(title, nx + 1, ny + 1))

            # Histogram
            ax_hist = axs[1, -1]
            ax_hist.hist(
                image_dB(c_idx_abs.ravel()),
                bins="auto",
                color=palette[idx],
                histtype="step",
                label="${0}_{{{1}{2}}}$".format(title, nx + 1, ny + 1),
                density=True,
                linestyle=("solid"),
            )

            ax_hist.legend(loc="upper left")  # , bbox_to_anchor=(0.5, 1.05),
            #  ncol=3, fancybox=True, shadow=True)
            ax_hist.set_title("Histogram")

            ax_phase = axs[2, idx]
            pcm_phase = ax_phase.imshow(
                c_idx_angle,
                interpolation="hamming",
                vmin=-np.pi,
                vmax=np.pi,  # vmin=.9*c11_db.min() , vmax= 1/3*c11_db.max(),
                cmap="jet",
                aspect=aspect,
                origin=origin,
                *args,
                **kwargs,
            )
            ax_phase.axis("off")
            ax_phase.axis("off")
            ax_phase.set_title(
                "$\phi_{{ {0}_{{{1}{2}}} }}$".format(title, nx + 1, ny + 1)
            )

            # Histogram
            ax_phase_hist = axs[2, -1]
            ax_phase_hist.hist(
                c_idx_angle.ravel(),
                bins="auto",
                color=palette[idx],
                histtype="step",
                label="$\phi_{{ {0}_{{{1}{2}}} }}$".format(
                    title, nx + 1, ny + 1
                ),
                density=True,
                linestyle=("solid"),
            )

            ax_phase_hist.legend(
                loc="upper left"
            )  # , bbox_to_anchor=(0.5, 1.05),
            #  ncol=3, fancybox=True, shadow=True)
            ax_phase_hist.set_title("Histogram")

        ax_phase_hist.set_xlim([-np.pi, np.pi])
        ax_phase_hist.set_xticks([-np.pi, 0, np.pi])
        ax_phase_hist.set_xticklabels(["$- \pi$", "0", "$\pi$"])

        cbar_2 = fig.colorbar(pcm, ax=axs[1, :3], shrink=0.6, location="left")
        cbar_2.set_label("$dB $ ", rotation=0)
        cbar_3 = fig.colorbar(
            pcm_phase,
            ax=axs[2, :3],
            shrink=0.6,
            location="left",
            ticks=[-np.pi, 0, np.pi],
        )
        cbar_3.ax.set_yticklabels(["$- \pi$", "0", "$\pi$"])
        cbar_3.set_label("$\phi$", rotation=0)
        return fig

    fig = plot_sections(c_t_matrix)
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
    # f_hist = {}
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
    fig = plt.figure(constrained_layout=False, figsize=(9 * n_images, 10))
    gs = fig.add_gridspec(nrows=2, ncols=n_images + 1)

    for idx, key in enumerate(array_dict):
        title = key
        array = array_dict[key]
        ax_im = fig.add_subplot(gs[0, idx])
        if (np.max(array) - np.min(array)) < 1.1:
            ax_cc = ax_im.imshow(array, cmap="jet", vmin=0.0, vmax=1.0)
        elif (np.max(array) - np.min(array)) < 100 and (
            np.max(array) - np.min(array)
        ) > 10:
            ax_cc = ax_im.imshow(array, cmap="jet", vmin=0.0, vmax=90.0)
        else:
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
