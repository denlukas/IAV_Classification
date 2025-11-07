import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from itertools import chain
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from infa.utils import repo_dir, set_seeds

from tsaug import TimeWarp, Drift, AddNoise

import typer

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
    Given a dataset as a TSV, create a test split reproducibly. Then,
    save the two partitions as a TSV file each.
    """
    def plot_cv_indices(cv, X, y, groups, ax, n_splits, lw=10):
        cmap_cv = plt.cm.coolwarm
        for ii, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            indices = np.full(len(X), np.nan)
            indices[test_idx] = 1
            indices[train_idx] = 0
            ax.scatter(range(len(X)), [ii + 0.5] * len(X),
                       c=indices, marker="_", lw=lw, cmap=cmap_cv,
                       vmin=-0.2, vmax=1.2)
        class_colors = np.array(['blue', 'red'])[y]
        ax.scatter(range(len(X)), [ii + 1.5] * len(X), c=class_colors,
                   marker="_", lw=lw)
        tab20b = plt.cm.get_cmap('tab20b').colors
        tab20c = plt.cm.get_cmap('tab20c').colors
        combined_colors = list(chain(tab20b, tab20c))
        rng = np.random.default_rng(seed=42)
        rng.shuffle(combined_colors)
        unique_group_map = {v: i for i, v in enumerate(sorted(set(groups)))}
        group_color_indices = [unique_group_map[g] for g in groups]
        group_colors = [combined_colors[i % len(combined_colors)] for i in group_color_indices]
        ax.scatter(range(len(X)), [ii + 2.5] * len(X), c=group_colors,
                   marker="_", lw=lw)
        yticklabels = list(range(n_splits)) + ['class', 'group']
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
    if data.shape[1] == 61:
        warnings.warn(f'TSV at {filename} still has 61 timepoints.', RuntimeWarning)

    if seed is not None:
        set_seeds(seed)

    labels = data.index.get_level_values('label')
    group_labels = data.reset_index().apply(lambda s: f"{s.label}_{s.video}", axis=1)
    groups = LabelEncoder().fit_transform(group_labels)

    if 'PR8' in labels:
        y = np.array([int(label == 'PR8') for label in labels])
    else:
        y = LabelEncoder().fit_transform(labels)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    print(f"Number of folds available from StratifiedGroupKFold: {sgkf.get_n_splits(data.values, y, groups)}")
    print(sgkf)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_cv_indices(sgkf, data.values, y, groups, ax=ax, n_splits=5)
    plt.show()

    # Select fold
    for i, (train_idx, test_idx) in enumerate(sgkf.split(data.values, y, groups)):
        if i == fold:
            break

    train_and_val_set = data.iloc[train_idx]
    test_set = data.iloc[test_idx]

    # === Data Augmentation ===
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

    # Save TSV files
    train_and_val_final.to_csv(data_dir / (filename + '_train_val.tsv'), sep='\t', header=True)
    test_set.to_csv(data_dir / (filename + '_test.tsv'), sep='\t', header=True)

    # Summary
    def print_summary(name: str, df: pd.DataFrame):
        print(f"\n{name} set summary:")
        label_video = df.reset_index()[['label', 'video']].drop_duplicates()
        print(f"  Unique (label, video) pairs: {len(label_video)}")
        print(label_video.sort_values(['label', 'video']).to_string(index=False))
        trace_counts = df.reset_index().groupby('label').size()
        print("\n  Trace counts per label:")
        for label, count in trace_counts.items():
            print(f"    {label}: {count} traces")

    print_summary("Train+Val", train_and_val_set)
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
    if data.shape[1] == 61:
        warnings.warn(f'TSV at {filename} still has 61 timepoints.', RuntimeWarning)

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

@app.command()
def heatmaps(
        filename: str='all_traces.tsv',
        seed: int | None = None,
        fold: int = 0,
        augment: bool = True,
        data_dir: Path = repo_dir / 'data',
        method: str='average'
) -> None:
    def load_datasets(filename):
        # Load raw dataset
        data = pd.read_csv(data_dir/filename, sep='\t')  # add sep='\t' when dealing with a tsv file instead of a csv file
        #print(data.shape)  # quick overview of loaded dataset

        # ensure that re-occurring video names do not cause the two classes to mix
        data.video = data.apply(lambda s: f'{s.label}_{s.video}', axis=1)
        # set a multi-index
        data = data.set_index(['label', 'video', 'trace'])
        return data

    data=load_datasets(filename)

    def clusmapsns(df: pd.DataFrame, dateiname='plot.png', y_label='GFP intensity [a.u.]', cluster: bool = False):
        df = df.copy()
        # Plot raw dataset in heatmap
        # Extract levels from MultiIndex columns
        labels = df.T.columns.get_level_values('label')
        videos = df.T.columns.get_level_values('video')

        # Define label-level color palette
        label_palette = dict(zip(labels.unique(), ["Red", "Blue"]))
        label_colors = labels.map(label_palette)

        # Get all unique videos
        all_videos = videos.unique()

        # Create combined palette from tab20b and tab20c
        combined_palette = sns.color_palette("tab20b", len(all_videos) // 2) + \
                           sns.color_palette("tab20c", len(all_videos) - len(all_videos) // 2)

        # Randomize the combined color list
        random.Random(4).shuffle(combined_palette)

        # Assign each video a random color
        video_palette = dict(zip(all_videos, combined_palette))

        # Map video colors using the randomized palette
        video_colors = videos.map(video_palette)

        # Combine into col_colors DataFrame
        col_colors = pd.DataFrame({
            'Label': label_colors,
            'Video': video_colors
        }, index=df.T.columns)

        # Create clustermap
        g = sns.clustermap(df.T,
                           method=method,
                           metric='euclidean',
                           figsize=(20, 10),
                           row_cluster=False,
                           col_cluster=cluster,
                           dendrogram_ratio=(0.001, 0.001),
                           col_colors=col_colors,
                           cmap="Greys_r",
                           cbar_pos=(1, 0.08, 0.02, 0.6),
                           yticklabels=False,
                           xticklabels=False)

        # Adjust colorbar size and position to match heatmap height exactly
        heatmap_pos = g.ax_heatmap.get_position()
        cbar_width = 0.02
        cbar_x = heatmap_pos.x1 + 0.01

        g.cax.set_position([cbar_x, heatmap_pos.y0, cbar_width, heatmap_pos.height])

        # Add black border around colorbar
        for spine in g.cax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
            spine.set_visible(True)

        # Add colorbar label
        g.cax.set_ylabel(y_label, rotation=270, labelpad=15, fontsize=12)
        g.ax_heatmap.set_xlabel('')

        g.savefig(dateiname, dpi=300)

        return g

    # create directory
    Path('Data_Cleaning').mkdir(parents=True, exist_ok=True)

    # create heatmap for raw traces
    g = clusmapsns(df=data, dateiname='Data_Cleaning/Clustermap_raw_traces.png')

    #load norm dataset and create heatmap
    data_norm=load_datasets('all_traces_norm.tsv')
    g = clusmapsns(df=data_norm,
                   dateiname='Data_Cleaning/Clustermap_norm_traces.png',
                   y_label='MinMax Scale')


    def min_max(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        return (s - s.values.min()) / (s.values.max() - s.values.min())

    def clusmapscipy(df: pd.DataFrame, dateiname='plot.png', y_label='GFP intensity [a.u.]', cluster: bool = False,
                     n_clusters=9, hack=False):
        df = df.copy()

        # Extract metadata from MultiIndex columns (label, video)
        labels = df.T.columns.get_level_values('label')
        videos = df.T.columns.get_level_values('video')

        # Color palette for labels
        label_palette = dict(zip(labels.unique(), ["Red", "Blue"]))
        label_colors = labels.map(label_palette)

        # Color palette for videos (combined tab20b + tab20c)
        all_videos = videos.unique()
        combined_palette = sns.color_palette("tab20b", len(all_videos) // 2) + \
                           sns.color_palette("tab20c", len(all_videos) - len(all_videos) // 2)
        random.Random(4).shuffle(combined_palette)
        video_palette = dict(zip(all_videos, combined_palette))
        video_colors = videos.map(video_palette)

        # Prepare col_colors DataFrame with Label and Video colors
        col_colors = pd.DataFrame({
            'Label': label_colors,
            'Video': video_colors
        }, index=df.T.columns)

        # Initialize cluster_colors as None
        cluster_colors = None

        # Perform clustering if requested, and create cluster colorbar
        if cluster:
            # Hierarchical clustering on rows (samples) of df
            Z = linkage(df.values, method='ward', metric='euclidean')
            ll = leaves_list(Z)
            clusters = fcluster(Z, t=n_clusters, criterion='maxclust')[ll]
            df = df.iloc[ll]

            # Create cluster color palette
            clu_pal = sns.color_palette("tab20", n_clusters)
            random.Random(4).shuffle(clu_pal)
            cluster_palette = dict(zip(range(1, n_clusters + 1), clu_pal))
            cluster_colors = pd.Series(clusters, index=df.index).map(cluster_palette)

            # Now col_colors also needs to be reordered to match df rows after clustering
            # BUT col_colors index matches columns of df.T (== rows of df), so reorder col_colors by df.index
            col_colors = col_colors.loc[df.index]

            # Add Cluster colors to col_colors
            col_colors['Cluster'] = cluster_colors

        # Plot clustermap
        g = sns.clustermap(
            # df.T,  # transpose so columns = samples (rows of df), rows = features (columns of df)
            # df.apply(min_max, axis = 1).T,
            df.T if not hack else df.apply(min_max, axis=1).T,
            metric='euclidean',
            method=method,
            figsize=(20, 10),
            row_cluster=False,  # rows = features, do not cluster features
            col_cluster=False if cluster else True,  # if cluster done manually, no clustering on columns
            dendrogram_ratio=(0.001, 0.001),
            col_colors=col_colors,
            cmap="Greys_r",
            cbar_pos=(1, 0.08, 0.02, 0.6),
            yticklabels=False,
            xticklabels=False
        )

        # Adjust colorbar size and position to match heatmap height exactly
        heatmap_pos = g.ax_heatmap.get_position()
        cbar_width = 0.02
        cbar_x = heatmap_pos.x1 + 0.01

        g.cax.set_position([cbar_x, heatmap_pos.y0, cbar_width, heatmap_pos.height])

        # Add black border around colorbar
        for spine in g.cax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
            spine.set_visible(True)

        # Add colorbar label
        g.cax.set_ylabel(y_label, rotation=270, labelpad=15, fontsize=12)

        # Remove x-axis label
        g.ax_heatmap.set_xlabel('')

        g.savefig(dateiname, dpi=300)

        return g

    # Call your clusmap function with clustering enabled (this does clustering internally)
    g = clusmapscipy(df=data_norm,
                     dateiname='Data_Cleaning/Clustermap_norm_traces_add_colorbar.png',
                     y_label='MinMax Scale',
                     cluster=True,
                     n_clusters=9)


    def clusmapscipy_ticks(df: pd.DataFrame, dateiname='plot.png'):
        data_norm2 = df.copy().reset_index()

        # Step 2: Extract numeric timepoint data (all float64 columns)
        data_numeric = data_norm2.select_dtypes(include='float')

        # Step 3: Extract 'label' and 'video' from the metadata columns
        labels = data_norm2['label']
        videos = data_norm2['video']

        # Step 4: Define color palettes
        label_palette = dict(zip(labels.unique(), ["Red", "Blue"]))
        label_colors = labels.map(label_palette)

        # Define video palettes separately for each label
        pr8_videos = videos[labels == "PR8"].unique()
        x31_videos = videos[labels == "X31"].unique()

        pr8_palette = sns.color_palette("tab20b", len(pr8_videos))
        x31_palette = sns.color_palette("tab20c", len(x31_videos))

        video_palette = dict(zip(pr8_videos, pr8_palette))
        video_palette.update(dict(zip(x31_videos, x31_palette)))

        video_colors = videos.map(video_palette)

        # Step 5: Build col_colors DataFrame (index must match columns of transposed data)
        col_colors = pd.DataFrame({
            'Label': label_colors,
            'Video': video_colors
        }, index=data_numeric.index)

        # Step 6: Clustermap (transposing so columns = samples)
        g = sns.clustermap(
            data_numeric.T,
            metric='euclidean',
            method=method,
            figsize=(20, 10),
            row_cluster=False,
            dendrogram_ratio=(0.001, 0.001),
            col_colors=col_colors,
            cmap="Greys_r",
            # cbar_pos=None,
            yticklabels=False,
            xticklabels=True
        )

        # Step 7: Colorbar formatting
        heatmap_pos = g.ax_heatmap.get_position()
        cbar_width = 0.02
        cbar_x = heatmap_pos.x1 + 0.03

        g.cax.set_position([cbar_x, heatmap_pos.y0, cbar_width, heatmap_pos.height])

        for spine in g.cax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
            spine.set_visible(True)

        g.cax.set_ylabel('MinMax scale', rotation=270, labelpad=15, fontsize=12)
        g.ax_heatmap.set_xlabel('')

        # Step 7.5: Show only ~50 xticklabels
        num_ticks = 30
        xticks = g.ax_heatmap.get_xticks()
        xtick_labels = g.ax_heatmap.get_xticklabels()

        # Choose evenly spaced indices
        selected_indices = list(np.linspace(0, len(xticks) - 1, num=num_ticks, dtype=int))

        # Set new ticks and labels
        g.ax_heatmap.set_xticks([xticks[i] for i in selected_indices])
        g.ax_heatmap.set_xticklabels([xtick_labels[i].get_text() for i in selected_indices],
                                     rotation=90, fontsize=8)

        # Step 8: Save figure
        g.savefig(dateiname, dpi=300)

        # Step 9: Reorder full dataset based on clustering
        col_order = g.dendrogram_col.reordered_ind
        reordered_data = data_norm2.iloc[col_order, :]
        #reordered_data.to_csv('Data_Cleaning/reordered_clustermap_output.tsv', sep='\t', index=False)

        return reordered_data

    reordered_data = clusmapscipy_ticks(df=data_norm,
                        dateiname='Data_Cleaning/Clustermap_norm_traces_col_clusters_w_x_ticklabels.png',
                         #y_label='MinMax Scale',
                         #cluster=True,
                         #n_clusters=9
                            )

    def subplt(df: pd.DataFrame, dateiname='subplots.png'):
        df = df.copy().reset_index()

        # Step 2: Extract numeric timepoint data (all float64 columns)
        df = df.select_dtypes(include='float')

        # Remove "time_" prefix from column names
        df.columns = df.columns.str.replace('time_', '')
        # Convert the column names to integers (if needed)
        df.columns = df.columns.astype(int)
        # Optional: check result
        # print(df.columns)

        # --- Step 1: Extract displayed x-tick labels on the clustermap heatmap after filtering ---
        displayed_xticklabels = [tick.get_text() for tick in g.ax_heatmap.get_xticklabels() if tick.get_text() != '']

        # These labels are your sample names exactly as shown on the dendrogram x-axis.

        # --- Step 2: Prepare timepoints ---
        timepoints = df.columns.astype(float)

        # Select 7 ticks for x-axis labeling on plots
        num_xticks = 7
        xtick_indices = np.linspace(0, len(timepoints) - 1, num=num_xticks, dtype=int)
        xticks = timepoints[xtick_indices]
        xtick_labels = [str(int(t)) if t.is_integer() else f"{t:.1f}" for t in xticks]

        # --- Step 3: Create subplots and plot these exact samples ---

        nrows, ncols = 5, 6  # 30 subplots grid
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, 3 * nrows), sharex=True, sharey=True)
        axs = axs.flatten()

        for i, sample_name in enumerate(displayed_xticklabels):
            # Convert sample_name dtype to match index type, if needed
            if df.index.dtype == 'int64':
                sample_name = int(sample_name)
            elif df.index.dtype == 'float64':
                sample_name = float(sample_name)
            # else assume string, no conversion

            trace = df.loc[sample_name]
            label = labels.loc[sample_name]
            video = videos.loc[sample_name]

            axs[i].plot(timepoints, trace, color='black')
            axs[i].set_title(f"Label: {label} | Video: {video} | Sample: {sample_name}", fontsize=8)

            axs[i].set_xticks(xticks)
            axs[i].set_xticklabels(xtick_labels, fontsize=8)
            axs[i].tick_params(axis='y', labelsize=8)

        # Hide any unused axes (if any)
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        # Y-label only on leftmost plots
        for row in range(nrows):
            axs[row * ncols].set_ylabel('MinMax scale [0-1]', fontsize=10)

        # X-label only on bottom row plots
        for col in range(ncols):
            bottom_ax_index = (nrows - 1) * ncols + col
            if bottom_ax_index < len(axs):
                axs[bottom_ax_index].set_xlabel('Time [seconds]', fontsize=10)

        plt.tight_layout()
        plt.savefig(dateiname, dpi=300)
        return fig

    fig = subplt(df=data_norm, dateiname='Data_Cleaning/Subplots_from_reordered_Clustermap.png')


    def sctplt(df: pd.DataFrame, dateiname='scatterplot.png'):
        df = df.copy()
        # Load reordered data
        # reordered_data = pd.read_csv('reordered_clustermap_output.tsv', sep='\t')

        # Add unique sample index to match clustermap
        df['sample_id'] = df.index

        # Compute cumulative sum per sample
        data_numeric = df.select_dtypes(include='float')
        df['cum_sum'] = data_numeric.sum(axis=1)
        # df['cum_sum'] = data_numeric.std(axis=1)

        # Extract info for plotting
        labels = df['label']
        cum_sums = df['cum_sum']
        colors = ['red' if lbl == 'PR8' else 'blue' for lbl in labels]

        # Plot in the order they appear in the reordered file (already clustermap-ordered!)
        plt.figure(figsize=(14, 6))
        plt.scatter(range(len(cum_sums)), cum_sums, c=colors, s=10)
        plt.xlabel('Reordered Sample Index')
        plt.xlim(0, len(df))
        plt.ylabel('Cumulative Sum')
        plt.title('Cumulative Sum of Reordered Samples by Strain')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(dateiname, dpi=300)
        return plt.gcf()

    fig=sctplt(df=reordered_data, dateiname="Data_Cleaning/Scatterplot_from_reordered_Clustermap.png")

