app = typer.Typer()


@dataclass
class DataSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    weights: np.ndarray | None
    train_idx: pd.Index | None
    test_idx: pd.Index | None
    features: list[str] | None
    labels: list[str] | None

@app.command()
def create_test_set(
    filename: str,
    seed: int | None = None,
    augment: bool = True,
    test_size: float = 0.2,
    data_dir: Path = repo_dir / 'data',
) -> None:
    """
    Given a TSV dataset (columns: label, trace_ID, time_1, time_2, ...),
    create a reproducible train/test split and optionally augment training data.
    Augmented traces (with '_m' in trace_ID) are excluded from test/validation.
    """
    # --- Load dataset ---
    data = pd.read_csv(data_dir / filename, sep='\t').set_index(['label', 'trace_ID'])

    # --- Separate augmented vs non-augmented traces ---
    mask_aug = data.index.get_level_values("trace_ID").str.contains("_m")
    data_aug = data[mask_aug]
    data_clean = data[~mask_aug]

    # --- Prepare labels ---
    labels = data_clean.index.get_level_values('label')
    y = LabelEncoder().fit_transform(labels)

    # --- Compute class weights ---
    class_counts = np.bincount(y)
    weights = np.ones_like(y, dtype=float)
    for cls, count in enumerate(class_counts):
        weights[y == cls] = len(y) / (len(class_counts) * count)

    # --- Stratified train/test split (only on clean traces) ---
    X_train, X_test, y_train, y_test, idx_train, idx_test, w_train, w_test = train_test_split(
        data_clean.values, y, data_clean.index, weights,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )

    # --- Add augmented traces back into training set ---
    X_train = np.vstack([X_train, data_aug.values])
    idx_train = idx_train.append(data_aug.index)

    # Expand y_train and weights to match augmented traces
    y_aug = LabelEncoder().fit(labels).transform(data_aug.index.get_level_values("label"))
    y_train = np.concatenate([y_train, y_aug])

    w_aug = np.ones_like(y_aug, dtype=float)
    w_train = np.concatenate([w_train, w_aug])

    # Shuffle training set
    train_df = pd.DataFrame(X_train, columns=data.columns, index=idx_train)
    train_df['y'] = y_train  # temporary column for shuffle
    train_df['w'] = w_train  # temporary column for shuffle
    train_df = train_df.sample(frac=1, random_state=seed).reset_index()
    y_train = train_df.pop('y').to_numpy()
    w_train = train_df.pop('w').to_numpy()
    train_df.set_index(['label', 'trace_ID'], inplace=True)

    train_final = train_df
    test_set = pd.DataFrame(X_test, columns=data.columns, index=idx_test)

    # --- Optional augmentation ---
    if augment:
        from tsaug import TimeWarp, Drift, AddNoise

        augmenter = (
            TimeWarp(n_speed_change=1, max_speed_ratio=2) * 1 +
            Drift(max_drift=(0.01, 0.05)) +
            AddNoise(scale=0.005)
        )

        train_augmented = augmenter.augment(train_final.values)
        aug_index = pd.MultiIndex.from_tuples(
            [(label, f"{trace}_aug{i}")
             for i in range(train_augmented.shape[0] // train_final.shape[0])
             for (label, trace) in train_final.index],
            names=train_final.index.names
        )
        augmented_df = pd.DataFrame(train_augmented, columns=data.columns, index=aug_index)
        train_final = pd.concat([train_final, augmented_df])
        print(f"Augmented samples added: {len(augmented_df)}")
        print(f"Final train set size: {len(train_final)}")

    # --- Save TSV files ---
    train_final.to_csv(data_dir / (filename.replace('.tsv', '_train_val.tsv')), sep='\t', header=True)
    test_set.to_csv(data_dir / (filename.replace('.tsv', '_test.tsv')), sep='\t', header=True)

    # --- Print summary ---
    def print_summary(name: str, df: pd.DataFrame):
        print(f"\n{name} set summary:")
        print(f"  Total traces: {len(df)}")
        label_counts = df.reset_index().groupby('label').size()
        for label, count in label_counts.items():
            print(f"    {label}: {count} traces")

    print_summary("Train+Val", train_final)
    print_summary("Test", test_set)


@app.command()
def load_dataset_split(
    filename: str,
    test_size: float = 0.2,
    seed: int | None = None,
    data_dir: Path = repo_dir / 'data',
) -> DataSplit:
    """
    Load a TSV dataset (columns: label, trace_ID, time_1, ...),
    and return a simple stratified train/test split as a DataSplit object.
    Augmented traces (with '_m' in trace_ID) are excluded from test/validation.
    """

    # --- Load dataset ---
    data = pd.read_csv(data_dir / filename,
                       sep='\t').set_index(['label', 'trace_ID'])
    print(f"Dataset loaded: {data.shape[0]} traces, {data.shape[1]} timepoints")

    # --- Separate augmented vs non-augmented traces ---
    mask_aug = data.index.get_level_values("trace_ID").str.contains("_m")
    data_aug = data[mask_aug]
    data_clean = data[~mask_aug]

    # --- Prepare labels ---
    labels = data_clean.index.get_level_values('label')
    y = LabelEncoder().fit_transform(labels)
    X = data_clean.values

    # --- Compute class weights ---
    class_counts = np.bincount(y)
    weights = np.ones_like(y, dtype=float)
    for cls, count in enumerate(class_counts):
        weights[y == cls] = len(y) / (len(class_counts) * count)

    # --- Stratified train/test split (only on clean traces) ---
    X_train, X_test, y_train, y_test, idx_train, idx_test, w_train, w_test = train_test_split(
        X, y, data_clean.index, weights,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )

    # --- Add augmented traces back into training set ---
    X_train = np.vstack([X_train, data_aug.values])
    idx_train = idx_train.append(data_aug.index)

    # Expand y_train and weights for augmented traces
    y_aug = LabelEncoder().fit(labels).transform(data_aug.index.get_level_values("label"))
    y_train = np.concatenate([y_train, y_aug])
    w_aug = np.ones_like(y_aug, dtype=float)
    w_train = np.concatenate([w_train, w_aug])

    # Shuffle training set
    train_df = pd.DataFrame(X_train, columns=data.columns, index=idx_train)
    train_df['y'] = y_train
    train_df['w'] = w_train
    train_df = train_df.sample(frac=1, random_state=seed).reset_index()
    y_train = train_df.pop('y').to_numpy()
    w_train = train_df.pop('w').to_numpy()
    train_df.set_index(['label', 'trace_ID'], inplace=True)

    # --- Construct DataSplit object ---
    split = DataSplit(
        x_train=train_df.values,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        weights=w_train,
        train_idx=train_df.index,
        test_idx=idx_test,
        features=data.columns.tolist(),
        labels=list(np.unique(labels))
    )

    # --- Print summary ---
    def print_summary(name: str, idx: pd.Index):
        print(f"\n{name} set summary:")
        label_counts = pd.Series(idx.get_level_values('label')).value_counts()
        for label, count in label_counts.items():
            print(f"    {label}: {count} traces")

    print_summary("Train", split.train_idx)
    print_summary("Test", split.test_idx)

    return split
