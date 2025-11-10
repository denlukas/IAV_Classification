import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
import typer

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from IAV_Classification.utils import repo_dir, set_seeds
from tsaug import TimeWarp, Drift, AddNoise
from typing import Optional, Union

app = typer.Typer()


@dataclass
class DataSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    weights: Optional[np.ndarray]  # If you don't want this attribute or any other below, remove as you want.
    train_idx: Optional[pd.Index]
    test_idx: Optional[pd.Index]
    features: Optional[list[str]]
    labels: Optional[list[str]]
    # groups: np.ndarray | None


@app.command()
def create_test_set(
        filename: str,
        seed: int | None = None,
        fold: int = 0,
        augment: bool = True,
        data_dir: Path = repo_dir / 'data',
) -> None:
    """
    Given a dataset as a TSV, this function creates a test split reproducibly.
    Then, the two partitions are saved as a TSV file each.
    """

    # Plot CV indices
    def plot_cv_indices(cv, X, y, groups, labels, videos, ax, n_splits, lw=10):
        """
        Visualizes how samples are assigned to each train/test split.
        """
        cmap_cv = plt.cm.coolwarm

        # Plot train/test splits per fold
        for ii, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            indices = np.full(len(X), np.nan)
            indices[test_idx] = 1
            indices[train_idx] = 0
            ax.scatter(range(len(X)), [ii + 0.5] * len(X),
                       c=indices, marker="_", lw=lw,
                       cmap=cmap_cv, vmin=-0.2, vmax=1.2)

        # Strain label colors
        strain_palette = {'PR8': 'tab:blue', 'X31': 'tab:orange'}
        label_colors = pd.Series(labels).map(lambda l: strain_palette.get(l, 'gray')).to_numpy()
        ax.scatter(range(len(X)), [ii + 1.5] * len(X),
                   c=label_colors, marker="_", lw=lw)

        # Video colors
        all_videos = np.unique(videos)
        combined_palette = sns.color_palette("tab20b", len(all_videos) // 2) + \
                           sns.color_palette("tab20c", len(all_videos) - len(all_videos) // 2)
        random.Random(4).shuffle(combined_palette)
        video_palette = dict(zip(all_videos, combined_palette))
        video_colors = pd.Series(videos).map(video_palette).to_numpy()
        ax.scatter(range(len(X)), [ii + 2.5] * len(X),
                   c=video_colors, marker="_", lw=lw)

        # Axis formatting
        yticklabels = list(range(n_splits)) + ['Strain', 'Video']
        ax.set(
            yticks=np.arange(n_splits + 2) + 0.5,
            yticklabels=yticklabels,
            xlabel='Sample index',
            ylabel='CV iteration',
            ylim=[n_splits + 2.2, -0.2],
            xlim=[0, len(X)],
        )
        ax.set_title(f'StratifiedGroupKFold - n_splits={n_splits}', fontsize=15)
        return ax

    # Load dataset
    data = pd.read_csv(data_dir / filename, sep='\t').set_index(['label', 'video', 'trace'])

    if seed is not None:
        set_seeds(seed)

    # Label & group setup
    labels = data.index.get_level_values('label')
    videos = data.index.get_level_values('video')
    group_labels = data.reset_index().apply(lambda s: f"{s.label}_{s.video}", axis=1)
    groups = LabelEncoder().fit_transform(group_labels)

    if 'PR8' in labels:
        y = np.array([int(label == 'PR8') for label in labels])
    else:
        y = LabelEncoder().fit_transform(labels)

    # Define CV
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    print(f"Number of folds available from StratifiedGroupKFold: {sgkf.get_n_splits(data.values, y, groups)}")
    print(sgkf)

    # Plot CV structure
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_cv_indices(sgkf, data.values, y, groups, labels, videos, ax=ax, n_splits=5)

    # Save figure
    output_dir = repo_dir / "IAV_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(filename).stem
    fig_path = output_dir / f"{base_name}_sgkf_fold{fold}.png"

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300)

    # Keep plot visible for interactive review
    plt.show()

    # Select specific fold
    for i, (train_idx, test_idx) in enumerate(sgkf.split(data.values, y, groups)):
        if i == fold:
            break

    train_and_val_set = data.iloc[train_idx]
    test_set = data.iloc[test_idx]

    # Data Augmentation
    if augment:
        augmenter = (
            TimeWarp(n_speed_change=1, max_speed_ratio=2) * 1 +
            Drift(max_drift=(0.01, 0.05)) +
            AddNoise(scale=0.005)
        )

        X_train = train_and_val_set.values
        train_augmented = augmenter.augment(X_train)

        # Build new index for augmented samples
        aug_index = pd.MultiIndex.from_tuples(
            [(label, video, f"{trace}_aug{i}")
             for i in range(train_augmented.shape[0] // X_train.shape[0])
             for (label, video, trace) in train_and_val_set.index],
            names=train_and_val_set.index.names
        )

        augmented_df = pd.DataFrame(train_augmented, columns=train_and_val_set.columns, index=aug_index)
        train_and_val_final = pd.concat([train_and_val_set, augmented_df])

        print(f"Augmented samples added: {len(augmented_df)}")
        print(f"Final train+val set size (with augmentation): {len(train_and_val_final)}")
    else:
        print("Skipping data augmentation...")
        train_and_val_final = train_and_val_set

    # Clean base name (remove .tsv if present)
    base_name = filename[:-4] if filename.endswith('.tsv') else filename

    # Save train/test TSVs with clean names
    train_and_val_final.to_csv(data_dir / f"{base_name}_train_val.tsv", sep='\t', header=True)
    test_set.to_csv(data_dir / f"{base_name}_test.tsv", sep='\t', header=True)

    # Summary print
    def print_summary(name: str, df: pd.DataFrame, augmented: bool = False, n_augmented: int = 0, n_original: int = 0):
        print(f"\n{name} set summary:")
        label_video = df.reset_index()[['label', 'video']].drop_duplicates()
        print(f"  Unique (label, video) pairs: {len(label_video)}")
        print(label_video.sort_values(['label', 'video']).to_string(index=False))

        trace_counts = df.reset_index().groupby('label').size()
        print("\n  Trace counts per label:")
        for label, count in trace_counts.items():
            print(f"    {label}: {count} traces")

        total_traces = len(df)
        if augmented:
            print(f"\n Total traces {name.lower()}: {n_original} + {n_augmented} augmented = {total_traces}")
        else:
            print(f"\n Total traces {name.lower()}: {total_traces}")
        print("-" * 50)

    # Print summaries
    if augment:
        print_summary(
            "Train+Val",
            train_and_val_final,
            augmented=True,
            n_augmented=len(augmented_df),
            n_original=len(train_and_val_set),
        )
    else:
        print_summary("Train+Val", train_and_val_final)

    print_summary("Test", test_set)


@app.command()
def load_dataset_split(
        filename: str,
        n_splits: int = 5,
        seed: Optional[int] = None,
        fold: int = 1,
        data_dir: Path = repo_dir / 'data',
) -> DataSplit:
    """
    Load a GFP blinking dataset TSV and return stratified, grouped train/test splits. The function
    constructs class-balanced stratified folds while preserving groupings based on "label_video",
    using StratifiedGroupKFold.
    To get only a single split instead of multiple folds, just use `load_dataset_splits(filename)[0]`.

    Notes:
        - Assumes the file has a multi-index with ['label', 'video', 'trace'].
        - Warns if the TSV has 61 timepoints, assuming the correction to drop "time=0".
        - Automatically calculates class weights, so they could be used in training.
        - Always treats 'PR8' as the positive class if it exists in the labels.

    Args:
        filename (str): Name of the TSV file to load.
        n_splits (int, optional): Number of cross-validation splits to generate. Defaults to 5.
        seed (int | None, optional): Set this for reproducible splits.
        fold (int, optional): Index of the fold to return. Defaults to 1, i.e. the first fold.
        data_dir (str | Path, optional): Directory containing the data file. Defaults to `repo_dir / 'data'`.

    Returns:
        DataSplit: A `DataSplit` object containing a train/test split
        with features, labels, weights, and metadata.

    Raises:
        RuntimeWarning: If the input file contains exactly 61 columns, indicating un-curated data.
    """

    data = pd.read_csv(data_dir / filename,
                       sep='\t').set_index(['label', 'video', 'trace'])

    x = data.values[:, :, np.newaxis]
    labels = data.index.get_level_values('label')
    if 'PR8' in labels:
        # ensure that PR8 is encoded as the positive class always
        y = np.array([int(d == 'PR8') for d in data.index.get_level_values('label')])
    else:
        # if this is something else, flexibly handle it
        y = LabelEncoder().fit_transform(labels)

    # calculate class weights
    f0, f1 = np.unique(y, return_counts=True)[1]
    weights = np.ones(y.shape[0])
    weights[y == 1] = (f0 + f1) / 2 / f1

    # build a label_video grouping, along which we will split
    group_labels = data.reset_index().apply(
        lambda s: f'{s.label}_{s.video}', axis=1)
    groups = LabelEncoder().fit_transform(group_labels)

    # get a train-val splitter
    sgkf = StratifiedGroupKFold(n_splits=n_splits,
                                shuffle=True,
                                random_state=seed)

    # get the indices for the correct fold
    train, test = next(split for i, split in enumerate(
        sgkf.split(x, y, groups=groups)) if i == fold)  # mind the groups here!

    # create a DataSplit object
    split = DataSplit(x[train], y[train], x[test], y[test],
                      weights[train],  # we do not need class weights for testing!
                      data.iloc[train].index,  # this and everything below is to re-construct the dataframe
                      data.iloc[test].index,
                      data.columns.tolist(), labels.unique().tolist())

    # Count and print how many X31 traces are in the validation set
    x31_count = sum(split.test_idx.get_level_values('label') == 'X31')
    print(f"Number of X31 traces in validation set: {x31_count}")
    pr8_count = sum(split.test_idx.get_level_values('label') == 'PR8')
    print(f"Number of PR8 traces in validation set: {pr8_count}")

    return split

# This class and the function below are meant to make it easier to load several datasets into memory
class DataSetChoice(str, Enum):
    # This thing is a lot like a dictionary, but instead of a key and value, each "entry" has a *name* and value.
    raw = 'all_traces'
    normalized = 'all_traces_normalized'
    # if you want, extend this with other curations / versions of the dataset!


def load_data(data_dir: str | Path = repo_dir / 'data') -> dict[str, pd.DataFrame]:
    """
    For all dataset aliases and filenames from the DataSetChoice Enum, load a TSV.
    Args:
        data_dir: The directory containing the TSV files. Defaults to `repo_dir / 'data'`.
    Returns:
        dict[str, pd.DataFrame]: A dictionary mapping dataset names to DataFrames.
    """
    datasets = dict()
    for choice in DataSetChoice:
        fh = Path(data_dir) / f'{choice.value}.tsv'
        if not fh.is_file():
            warnings.warn(f'File {fh} not found', RuntimeWarning)
        else:
            datasets[choice.name] = (pd.read_csv(fh, sep='\t')
                                     .set_index(['label', 'video', 'trace']))
    return datasets