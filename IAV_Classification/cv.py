import concurrent.futures
import multiprocessing as mp
from functools import partial
import numpy as np

import sys

import mlflow

from IAV_Classification.data import load_dataset_split
from IAV_Classification.ml import fit
from IAV_Classification.model import make_model
from IAV_Classification.utils import set_seeds

import typer

app = typer.Typer()


def train_fold(fold: int,
               parent_run_info: str,
               seed: int,
               dataset: str,
               epochs: int,
               batch_size: int,
               patience: int | None,
               n_folds: int,
               palette: str | None,
               verbose: int | None,
               **kwargs) -> int:
    """
    This function trains a model on a fold. It's most important purpose is to
    pick the correct data, and log to MLflow correctly.
    """

    set_seeds(seed)
    run_name = parent_run_info.run_name + f'_fold_{fold}'

    tags = {'cv': str(True), 'fold': str(fold)}
    if palette is not None:
        # get a colour for this fold
        import seaborn as sns
        color = sns.color_palette(palette, n_folds).as_hex()[fold]
        tags['mlflow.runColor'] = color

    with mlflow.start_run(run_name=run_name,
                          parent_run_id=parent_run_info.run_id,
                          nested=True, tags=tags,
                          ) as run:
        mlflow.log_params(dict(seed=seed, fold=fold, n_splits=n_folds,
                               parent_run_info=parent_run_info, run_name=run_name, dataset=dataset,
                               epochs=epochs, batch_size=batch_size, patience=patience, ))
        mlflow.set_tags(dict(cv=True, fold=fold, ))

        # get the splits for the i-th fold
        datasplit = load_dataset_split(dataset, n_splits=5, seed=2501, fold=0)
        #print(datasplit)
        num_x31_val = datasplit.test_idx.get_level_values("label").tolist().count("X31")
        print(f'Fold {fold} val split: {list(datasplit.test_idx.get_level_values("video").unique())}, '
              f'val size: {len(datasplit.y_test)} train size: {len(datasplit.y_train)}',
              f"Fold {fold}: Number of label traces for X31 in validation set: {num_x31_val}"
              )

        if num_x31_val == 0:
            print(f'Fold {fold} has only one class in the validation set! {num_x31_val}', file=sys.stderr)
            return fold

        # this is a hack to get the batch size to increase with the fold number
        batch_size = [2 ** i for i in range(3, 20)][fold]
        mlflow.log_param('fold_batch_size', batch_size)
        model = make_model(datasplit.x_train.shape[1:])

        # this is another hack to space out a parameter along the folds
        #mcd_values = np.arange(0.1, 1.0, 0.1)  # [0.1, 0.2, ..., 0.9]
        #mcd = float(mcd_values[fold])
        #mlflow.log_param('mcd', mcd)
        #model = make_model(datasplit.x_train.shape[1:], mcd=mcd)

        try:
            history = fit(model=model,
                          datasplit=datasplit,
                          run_name=run_name,
                          epochs=epochs,
                          batch_size=batch_size,
                          patience=patience,
                          verbose=verbose,
                          )
        except KeyboardInterrupt:
            pass

        from sklearn.metrics import confusion_matrix
        y_pred_bin = (model.predict(datasplit.x_test) > 0.5).astype(int)
        print(confusion_matrix(datasplit.y_test, y_pred_bin))

        # Infer the model signature
        signature = mlflow.models.infer_signature(datasplit.x_train[:4], model.predict(datasplit.x_train[:4]))

        # Log the model, which inherits the parameters and metric
        model_info = mlflow.keras.log_model(model, run_name, signature=signature)
    return fold


@app.command()
def cross_validate(
        dataset: str,
        run_name: str | None = None,
        n_folds: int = 10,
        n_jobs: int = mp.cpu_count(),
        epochs: int = 1000,
        batch_size: int = 32,
        seed: int | None = 42,
        patience: int | None = 200,
        palette: str | None = None,
        verbose: int | None = 0,
) -> None:
    """
    Perform k-fold cross-validation on a given dataset using multiprocessing.

    This function orchestrates the cross-validation process by splitting the dataset
    into `n_folds` and training each fold using a separate process, using at most
    `n_jobs` worker processes in parallel.
    Training parameters are passed through. The entire run is tracked using MLflow,
    with the parent run logging common parameters and each fold logged as a child.
    Raise:
        KeyboardInterrupt: Allows graceful shutdown of parallel processes on user interruption.
    """
    seed = set_seeds(seed)

    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.log_params(dict(seed=seed, dataset=dataset, run_name=run_name,
                               patience=patience, epochs=epochs,
                               batch_size=batch_size, n_folds=n_folds, ))
        mlflow.set_tags(dict(cv=True, n_folds=n_folds, ))
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for fold in executor.map(
                        partial(train_fold, parent_run_info=parent.info,
                                seed=seed, dataset=dataset,
                                epochs=epochs, batch_size=batch_size,
                                patience=patience, n_folds=n_folds, palette=palette, verbose=verbose, ),
                        range(n_folds)):
                    print(f'Fold {fold} finished')
        except KeyboardInterrupt:
            executor.shutdown(wait=True, cancel_futures=True)
