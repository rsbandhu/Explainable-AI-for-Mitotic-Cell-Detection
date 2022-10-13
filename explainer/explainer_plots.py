import warnings
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from matplotlib import cm, colors, pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.pyplot import axis, figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray

from captum.attr import visualization as viz

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

class VisualizeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4


def plot_image_attribution_heatmap(original_image, norm_attr, cmap="Blues"):
    
    fig = plt.figure(figsize=(6, 5))

    h = sns.heatmap(norm_attr, cmap= cmap, alpha=0.7, zorder=2)  # update
    #my_image = mpimg.imread("./image.png") # update
    # update
    h.imshow(original_image,
            aspect=h.get_aspect(),
            extent=h.get_xlim() + h.get_ylim(),
            zorder=1, cmap='gray')
    h.set(xticklabels=[])
    h.set(yticklabels=[])

    return fig, h


def plot_attributions(original_image, attributions, cmaps, titles):

    #fig = plt.figure(figsize=(6, 5))
    nrows = 2
    ncols =3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18,12))

    for i in range(nrows):
        for j in range(ncols-1):
            idx = i*nrows + j
            attr = attributions[idx]
            cmap = cmaps[idx]
            title = titles[idx]

            h = sns.heatmap(attr, ax=axes[i, j+1],
                            cmap=cmap, alpha=0.7, zorder=2)  # update
            #my_image = mpimg.imread("./image.png") # update
            # update
            h.imshow(original_image,
                    aspect=h.get_aspect(),
                    extent=h.get_xlim() + h.get_ylim(),
                    zorder=1, cmap='gray')
            h.set(xticklabels=[])
            h.set(yticklabels=[])
            axes[i, j+1].set_title(title)
    
        h = sns.heatmap(attr, ax=axes[i, 0],
                        cmap=cmap, alpha=0.0, zorder=2)  # update
        #my_image = mpimg.imread("./image.png") # update
        # update
        h.imshow(original_image,
                aspect=h.get_aspect(),
                extent=h.get_xlim() + h.get_ylim(),
                zorder=1, cmap='gray')
        h.set(xticklabels=[])
        h.set(yticklabels=[])
        axes[i, 0].set_title("Original Image")

    return fig

def cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def normalize_attr(attr: ndarray, sign: str, outlier_perc: Union[int, float] = 2,
    reduction_axis: Optional[int] = None):


    VisualizeSign = viz.VisualizeSign
    _cumulative_sum_threshold = viz._cumulative_sum_threshold

    attr_combined = attr
    if reduction_axis is not None:
        attr_combined = np.sum(attr, axis=reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign[sign] == VisualizeSign.all:
        threshold = _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.positive:
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(
            attr_combined, 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.negative:
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        attr_combined = np.abs(attr_combined)
        threshold = viz._cumulative_sum_threshold(
            attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return viz._normalize_scale(attr_combined, threshold)


def visualize_image_attr(
    attr: ndarray,
    original_image: Union[None, ndarray] = None,
    sign: str = "absolute_value",
    plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
    outlier_perc: Union[int, float] = 2,
    cmap: Union[None, str] = None,
    alpha_overlay: float = 0.5,
    show_colorbar: bool = False,
    title: Union[None, str] = None,
    fig_size: Tuple[int, int] = (6, 6),
    use_pyplot: bool = True,
):

    ImageVisualizationMethod = viz.ImageVisualizationMethod

    heat_map = None
    
    # Choose appropriate signed attributions and normalize.
    norm_attr = normalize_attr(attr, sign, outlier_perc, reduction_axis=2)

    # Set default colormap and bounds based on sign.
    if VisualizeSign[sign] == VisualizeSign.all:
        default_cmap = LinearSegmentedColormap.from_list(
            "RdWhGn", ["red", "white", "green"]
        )
        vmin, vmax = -1, 1
    elif VisualizeSign[sign] == VisualizeSign.positive:
        default_cmap = "Greens"
        vmin, vmax = 0, 1
    elif VisualizeSign[sign] == VisualizeSign.negative:
        default_cmap = "Reds"
        vmin, vmax = 0, 1
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        default_cmap = "Blues"
        vmin, vmax = 0, 1
    else:
        raise AssertionError("Visualize Sign type is not valid.")

    cmap = cmap if cmap is not None else default_cmap

    return original_image, norm_attr, cmap, vmin, vmax


def plot_heatmap(img1_orig, norm_attr,  cmap, vmin, vmax, method: str = "heat_map", alpha=0.7):

    ImageVisualizationMethod = viz.ImageVisualizationMethod
    heat_map = None
    
    plt_fig = plt.figure(figsize=(6, 6))
    plt_axis = plt_fig.subplots()

    original_image = viz._prepare_image(img1_orig * 255)
    "Original Image must be provided for any visualization other than heatmap."

    # Remove ticks and tick labels from plot.
    plt_axis.xaxis.set_ticks_position("none")
    plt_axis.yaxis.set_ticks_position("none")
    plt_axis.set_yticklabels([])
    plt_axis.set_xticklabels([])
    #plt_axis.grid(b=False)

    plt_axis.imshow(np.mean(original_image, axis=2), cmap="gray")
    plt_axis.imshow(norm_attr, cmap='Blues', vmin=vmin, vmax=vmax, alpha=0.7)

    # Show appropriate image visualization.
    if ImageVisualizationMethod[method] == ImageVisualizationMethod.heat_map:
        heat_map = plt_axis.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax)
    elif (
        ImageVisualizationMethod[method]
        == ImageVisualizationMethod.blended_heat_map):
        #plt_axis.imshow(np.mean(original_image, axis=2), cmap="gray")
        heat_map = plt_axis.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha
        )

    axis_separator = make_axes_locatable(plt_axis)
    colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
    plt_fig.colorbar(heat_map, orientation="horizontal", cax=colorbar_axis)

    return plt_fig


