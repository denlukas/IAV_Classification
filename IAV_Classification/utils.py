from pathlib import Path

import keras
import mlflow

import numpy as np
import pandas as pd
import seaborn as sns

from PIL import Image
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster

repo_dir = Path(__file__).resolve().parents[1]


def set_seeds(seed: int | None = None) -> int:
    if seed is None:
        seed = np.random.default_rng().integers(0, 2 ** 32).item()
        print(f'seed is {seed}')
    keras.utils.set_random_seed(seed)
    return seed


def to_intensities(_data: pd.DataFrame,
                   keep: list[str] | None = list(),
                   ) -> pd.DataFrame:
    # melt the OG table to that intensities can be plotted over time
    intensities_df = _data.reset_index().melt(
        id_vars=['label', 'video', 'trace'] + [
            k for k in keep if k in _data.columns],
        value_name='intensity', var_name='time', )
    # turn time into an integer
    intensities_df.time = [int(str(t).split('_')[-1]) for t in intensities_df.time]
    return intensities_df


def save(figure: plt.Figure | sns.FacetGrid | Image.Image,
         filename: str | Path,
         plots_dir: Path = repo_dir / 'plots',
         dpi: int = 300,
         ) -> None:
    """
    A helper function to save a matplotlib or seaborn figure.
    :param figure: the figure to save
    :param filename: the filename, including extension -> defining filetype
    :param plots_dir: The directory to save under, defaulting to @repo_dir / 'plots'
    :param dpi: resolution of the figure, defaults to 300
    :return:
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)
    if type(figure) == Image.Image:
        figure.save(plots_dir / filename)
    else:
        figure.savefig(plots_dir / filename, dpi=dpi, bbox_inches='tight')
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(plots_dir / filename), artifact_path='plots')



def get_clusterized_df_and_clusters(
        dataframe: pd.DataFrame,
        method: str = 'ward',
        metric: str = 'euclidean',
        optimal: bool = False,
        n_clusters: int = 2,
) -> tuple[pd.DataFrame, np.ndarray]:
    # compute the linkage matrix
    Z = linkage(dataframe.values,
                method=method, metric=metric,
                optimal_ordering=optimal,
                )

    # get the order of the leaves (i.e., reordered row indices)
    ll = leaves_list(Z)

    # get the reordered cluster allocations
    cluster_annotations = fcluster(Z, t=n_clusters, criterion='maxclust')[ll]

    # return the reordered DF and clusters
    return dataframe.iloc[ll], cluster_annotations



def make_palette_color_strip(
        values: np.ndarray,
        palette_name: str = 'mako',
        height: float = 5,
        shuffle: bool = False,
        nan_color: tuple = (0.85, 0.85, 0.85),  # Light gray in RGB
) -> np.ndarray:
    categories = np.unique(values)  # sort for consistent results
    palette = sns.color_palette(palette_name, n_colors=len(categories))

    # if using a nice continuous colourmap like mako,
    # enable shuffling it to avoid the impression of related-ness.
    if shuffle:
        rng = np.random.default_rng(42)  # be consistent about randomness
        palette = np.array(palette)[rng.permutation(len(categories))]

    # map categories to palette colors complicatedly
    color_map = dict(zip(categories, palette))
    rgb = np.array([color_map.get(val, nan_color) for val in values])
    rgb = (np.array(rgb) * 255).astype(np.uint8)

    # repeat vertically to form strip
    strip = np.repeat(rgb[np.newaxis, :, :], height, axis=0)
    return strip


def add_color_strip_to_axis(
        ax: plt.Axes,
        values: np.ndarray,
        palette_name: str = 'mako',
        height_frac: float = 0.03,
        location: str = 'bottom',
        shuffle: bool = False,
        nan_color: tuple = (0.85, 0.85, 0.85),
) -> plt.Axes:
    """
    Adds a color annotation strip to a given matplotlib axis.

    Parameters:
        ax           : Target matplotlib Axes.
        values       : Numpy Array of integers with categorical values.
        palette_name : Seaborn palette name.
        height_frac  : Fraction of the original axis height to use for the strip.
        location     : 'bottom' or 'top' of the original axis.
        shuffle      : Whether to shuffle color assignments.
        nan_color    : RGB tuple for missing values (in [0, 1] scale).

    Returns:
        strip_ax     : The inset Axes containing the color strip.
    """
    fig = ax.figure
    bbox = ax.get_position()

    # Position for the new inset axis
    if location == 'bottom':
        y0 = bbox.y0 - height_frac
    elif location == 'top':
        y0 = bbox.y1
    else:
        raise ValueError("location must be 'bottom' or 'top'")

    strip = make_palette_color_strip(
        values, palette_name, height=10, shuffle=shuffle, nan_color=nan_color)
    strip_ax = fig.add_axes([
        bbox.x0,
        y0,
        bbox.width,
        height_frac
    ])
    strip_ax.imshow(strip, aspect='auto')
    strip_ax.axis('off')

    return strip_ax


def draw_image(ar: np.ndarray, scale: float = 15) -> Image:
    # normalize to 0–255 and convert to uint8
    normalized = normalize_array(ar)
    # transpose to wide format
    img = Image.fromarray(normalized.T)
    width, height = img.size
    img = img.resize((width, height * scale), resample=Image.NEAREST)
    return img


def normalize_array(ar: np.ndarray) -> np.ndarray:
    return (255 * (ar - ar.min()) / (ar.max() - ar.min())).astype(np.uint8)


def grayscale_to_rgb_colormap(array: np.ndarray, cmap_name='flare') -> np.ndarray:
    """
    Convert a 2D grayscale array (values in [0,1]) to RGB using a matplotlib/seaborn colormap.

    Args:
        array (np.ndarray): 2D grayscale array with values in [0, 1].
        cmap_name (str): Name of a seaborn/matplotlib colormap (e.g., 'flare', 'crest').

    Returns:
        np.ndarray: 3D RGB image array with shape (H, W, 3), dtype=uint8.
    """
    cmap = plt.get_cmap(cmap_name)
    rgba_img = cmap(normalize_array(array))  # shape: (H, W, 4)
    rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)
    return rgb_img
