from perun import monitor
from pathlib import Path

import mlflow
import keras
import tensorflow as tf

import pandas as pd
import numpy as np

from infa.utils import repo_dir, set_seeds
from infa.mcc import MatthewsCorrelationCoefficient
from infa.data import DataSplit, load_dataset_split

from infa.model import make_model, monte_carlo_predict_samples

import typer

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay, classification_report, accuracy_score,
    confusion_matrix, matthews_corrcoef
)

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import wasserstein_distance

app = typer.Typer()


@monitor()
def fit(model: keras.Model,
        datasplit: DataSplit,
        run_name: str,
        epochs: int = 500,
        batch_size: int = 32,
        patience: int | None = 50,
        verbose: int | None = 1,
        ) -> keras.callbacks.History:
    chkpoint = repo_dir / 'mlmodels' / f'{run_name}.keras'
    chkpoint.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        mlflow.keras.MlflowCallback(),
        keras.callbacks.ModelCheckpoint(
            chkpoint, save_best_only=True, monitor='val_loss', mode='min',
        ),
    ]

    if patience is not None and patience > 0:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=patience, verbose=1))

    history = model.fit(
        datasplit.x_train,
        datasplit.y_train,
        sample_weight=datasplit.weights,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(datasplit.x_test, datasplit.y_test),
        shuffle=True,
        verbose=verbose,
    )
    return history


@app.command()
def train(
        dataset: str = 'normalized.tsv',
        run_name: str | None = None,
        epochs: int = 1000,
        batch_size: int = 32,
        seed: int | None = 42,
        fold: int = 1,
        patience: int | None = typer.Option(
            200, help='If > 0, use early stopping after this many epochs. Turn it off by passing "--patience 0"'),
        verbose: int = 2,
) -> None:
    seed = set_seeds(seed)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(dict(seed=seed, dataset=dataset,
                               run_name=run_name,
                               patience=patience, epochs=epochs,
                               batch_size=batch_size, ))

        datasplit = load_dataset_split(dataset, n_splits=5, seed=seed, fold=fold)
        model = make_model(datasplit.x_train.shape[1:])

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

        y_pred_prob = model.predict(datasplit.x_test).astype('float32')
        y_pred_bin = (y_pred_prob > 0.5).astype(int)
        print(confusion_matrix(datasplit.y_test, y_pred_bin))
        acc = accuracy_score(datasplit.y_test, y_pred_bin)
        print(f"\nAccuracy: {acc:.4f}")

        signature = mlflow.models.infer_signature(datasplit.x_train[:4], model.predict(datasplit.x_train[:4]))
        mlflow.keras.log_model(model, run_name, signature=signature)


@app.command()
def evaluate(model_uri: str,
             dataset: str = 'normalized.tsv',
             seed: int | None = 42,
             fold: int = 1,
             include_uncertainty: bool = True,
             use_post_analysis: bool = True,
             n_mc_samples: int = 100,
             ct_threshold: float = 0.7
             ) -> None:
    model = mlflow.keras.load_model(model_uri)
    datasplit = load_dataset_split(dataset, n_splits=5, seed=seed, fold=fold)

    # -------------------
    # Monte Carlo predictions
    # -------------------
    if include_uncertainty:
        print("Running Monte Carlo prediction...")
        y_pred_prob, probs_try = monte_carlo_predict_samples(
            model, datasplit.x_test, n_mc_samples=n_mc_samples
        )

        # Fix for binary classification
        if probs_try.shape[2] == 1:
            probs_try = np.concatenate([1 - probs_try, probs_try], axis=2)

    else:
        y_pred_prob = model.predict(datasplit.x_test).astype("float32")
        probs_try = None

    y_pred_prob = y_pred_prob.flatten()
    y_pred_bin = (y_pred_prob > 0.5).astype(int)

    # -------------------
    # Standard evaluation metrics
    # -------------------
    acc = accuracy_score(datasplit.y_test, y_pred_bin)
    mcc = matthews_corrcoef(datasplit.y_test, y_pred_bin)

    target_names = ["X31", "PR8"]
    class_report = classification_report(datasplit.y_test, y_pred_bin,
                                         target_names=target_names, digits=3)
    auc_score = roc_auc_score(datasplit.y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(datasplit.y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(datasplit.y_test, y_pred_prob)
    ap_score = average_precision_score(datasplit.y_test, y_pred_prob)
    baseline_precision = np.mean(datasplit.y_test)

    total_traces = len(datasplit.y_test)
    pr8_traces = np.sum(datasplit.y_test == 1)
    x31_traces = np.sum(datasplit.y_test == 0)

    # -------------------
    # Evaluation plots
    # -------------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    disp = ConfusionMatrixDisplay.from_predictions(
        datasplit.y_test, y_pred_bin, ax=ax[0, 0], cmap="viridis",
        colorbar=True, normalize='true', display_labels=target_names
    )
    im = ax[0, 0].images[-1]
    im.set_norm(Normalize(vmin=0, vmax=1))
    cbar = disp.ax_.images[-1].colorbar
    if cbar:
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[0, 0].set_title("Confusion Matrix", fontsize=14, pad=15)

    ax[0, 1].plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
    ax[0, 1].set_xlabel("False Positive Rate")
    ax[0, 1].set_ylabel("True Positive Rate")
    ax[0, 1].legend(loc="lower right")

    # --- Unfiltered Precision-Recall Curve ---
    baseline_precision = np.mean(datasplit.y_test)
    ax[1, 0].plot(recall, precision, label=f"AP = {ap_score:.4f}")
    ax[1, 0].hlines(
        y=baseline_precision, xmin=0, xmax=1,
        colors="gray", linestyles="--", label=f"Baseline = {baseline_precision:.2f}"
    )
    ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, pad=15)
    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].legend(loc="lower left")

    metrics_text = (
        f"Total number of traces: {total_traces}\n"
        f"PR8: {pr8_traces}\n"
        f"X31: {x31_traces}\n\n"
        f"Accuracy: {acc:.4f}\n"
        f"MCC: {mcc:.4f}\n\n"
        f"Classification Report:\n{class_report}"
    )
    ax[1, 1].axis("off")
    ax[1, 1].text(0, 1, metrics_text, ha="left", va="top",
                  family="monospace", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle("Model Evaluation Report", fontsize=20, weight="bold")
    plt.show()

    print("\n================ Evaluation Report ================\n")
    print(f"Accuracy: {acc:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print("\nClassification Report:")
    print(class_report)
    print(f"ROC-AUC: {auc_score:.4f}")
    print(f"Average Precision (PR-AUC): {ap_score:.4f}")

    # -------------------
    # Post-analysis with CT (Wasserstein distance)
    # -------------------
    if use_post_analysis and include_uncertainty:
        print("\n================ Post-analysis ================\n")

        probs_mean = probs_try.mean(axis=0)
        predicted_label = np.argmax(probs_mean, axis=1)
        prediction_probability = probs_mean.max(axis=1)
        prediction_stds = probs_try.std(axis=0).mean(axis=1)

        list_min_w_distances = []
        for i in range(probs_try.shape[1]):
            sample_probs = probs_try[:, i, :]
            mean_probs = sample_probs.mean(axis=0)
            top_class = np.argmax(mean_probs)
            other_classes = [c for c in range(sample_probs.shape[1]) if c != top_class]

            min_w = min(
                (wasserstein_distance(sample_probs[:, top_class], sample_probs[:, c])
                 for c in other_classes),
                default=0.0
            )
            list_min_w_distances.append(min_w)

        post_df = pd.DataFrame({
            "y_true": datasplit.y_test,
            "predicted_label": predicted_label,
            "prediction_probability": prediction_probability,
            "prediction_std": prediction_stds,
            "min_wasserstein": list_min_w_distances
        })

        # -------------------
        # Plot min_wasserstein distribution with viridis gradient
        # -------------------
        plt.figure(figsize=(6, 4))

        # Compute histogram
        counts, bins = np.histogram(post_df['min_wasserstein'], bins=50)

        # Normalize counts to map to colormap
        norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
        cmap = plt.cm.viridis
        colors = cmap(norm(counts))

        # Plot each bin with corresponding color
        for i in range(len(bins) - 1):
            plt.bar(bins[i], counts[i], width=bins[i + 1] - bins[i], color=colors[i], align='edge')

        plt.xlabel("Minimal Wasserstein distance")
        plt.ylabel("Number of samples")
        plt.title("Distribution of minimal Wasserstein distance")
        plt.show()

        # -------------------
        # Second Evaluation Report (Filtered by CT threshold)
        # -------------------
        mask_ct = post_df["min_wasserstein"] > ct_threshold
        filtered_df = post_df[mask_ct]

        if len(filtered_df) > 0:
            y_true_filt = filtered_df["y_true"].values
            y_pred_filt = filtered_df["predicted_label"].values
            y_prob_filt = filtered_df["prediction_probability"].values

            acc_filt = accuracy_score(y_true_filt, y_pred_filt)
            mcc_filt = matthews_corrcoef(y_true_filt, y_pred_filt)
            auc_filt = roc_auc_score(y_true_filt, y_prob_filt)
            precision_filt, recall_filt, _ = precision_recall_curve(y_true_filt, y_prob_filt)
            ap_filt = average_precision_score(y_true_filt, y_prob_filt)
            fpr_filt, tpr_filt, _ = roc_curve(y_true_filt, y_prob_filt)
            class_report_filt = classification_report(y_true_filt, y_pred_filt,
                                                      target_names=target_names, digits=3)

            total_filtered = len(y_true_filt)
            pr8_filtered = np.sum(y_true_filt == 1)
            x31_filtered = np.sum(y_true_filt == 0)

            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            disp_filt = ConfusionMatrixDisplay.from_predictions(
                y_true_filt, y_pred_filt, ax=ax[0, 0], cmap="viridis",
                colorbar=True, normalize='true', display_labels=target_names
            )
            im_filt = ax[0, 0].images[-1]
            im_filt.set_norm(Normalize(vmin=0, vmax=1))
            cbar_filt = disp_filt.ax_.images[-1].colorbar
            if cbar_filt:
                cbar_filt.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax[0, 0].set_title(f"Confusion Matrix (CT>{ct_threshold})", fontsize=14, pad=15)

            ax[0, 1].plot(fpr_filt, tpr_filt, label=f"AUC = {auc_filt:.4f}")
            ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
            ax[0, 1].legend(loc="lower right")

            ax[1, 0].plot(recall_filt, precision_filt, label=f"AP = {ap_filt:.4f}")
            baseline_precision_filt = np.mean(y_true_filt)
            ax[1, 0].hlines(
                y=baseline_precision_filt, xmin=0, xmax=1,
                colors="gray", linestyles="--", label=f"Baseline = {baseline_precision_filt:.2f}"
            )
            ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, pad=15)
            ax[1, 0].set_ylim(0, 1)
            ax[1, 0].legend(loc="lower left")

            metrics_text = (
                f"Traces remaining: {total_filtered}\n"
                f"PR8: {pr8_filtered}\n"
                f"X31: {x31_filtered}\n\n"
                f"Accuracy: {acc_filt:.4f}\n"
                f"MCC: {mcc_filt:.4f}\n\n"
                f"Classification Report:\n{class_report_filt}"
            )
            ax[1, 1].axis("off")
            ax[1, 1].text(0, 1, metrics_text, ha="left", va="top",
                          family="monospace", fontsize=11)

            plt.suptitle("Filtered Evaluation Report", fontsize=20, weight="bold")
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.show()

            print("\n================ Filtered Evaluation Report ================\n")
            print(f"Traces remaining: {total_filtered}")
            print(f"PR8: {pr8_filtered}")
            print(f"X31: {x31_filtered}")
            print(f"Accuracy (CT>{ct_threshold}): {acc_filt:.4f}")
            print(f"MCC (CT>{ct_threshold}): {mcc_filt:.4f}")
            print(class_report_filt)
            print(f"ROC-AUC (CT>{ct_threshold}): {auc_filt:.4f}")
            print(f"Average Precision (PR-AUC): {ap_filt:.4f}")

        # -------------------
        # ECDF Plot
        # -------------------
        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.get_cmap("tab10")
        colors = {label: cmap(i % cmap.N) for i, label in enumerate(target_names)}

        for label, idx in zip(target_names, [0, 1]):
            values = post_df.loc[post_df["y_true"] == idx, "min_wasserstein"].values
            if len(values) == 0:
                continue
            x = np.sort(values)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where="post", color=colors[label], alpha=0.8, label=f"{label} (n={len(values)})")

        ax.set_xlabel("min_wasserstein")
        ax.set_ylabel("ECDF")
        ax.set_title(f"ECDF of min_wasserstein by class (CT>{ct_threshold})")
        ax.legend(title="Class")
        plt.tight_layout()
        plt.show()

        # -------------------
        # Bar plot for fractions across CT thresholds
        # -------------------
        max_val = post_df["min_wasserstein"].max()
        ct_thresholds = np.linspace(0, max_val, num=5)  # 5 evenly spaced thresholds
        ct_thresholds = np.round(ct_thresholds, 2)  # round for cleaner labels

        fractions_per_threshold = []

        for ct in ct_thresholds:
            mask_ct = post_df['min_wasserstein'] >= ct  # use >= instead of >
            y_true_filtered = post_df["y_true"].values[mask_ct] if mask_ct.sum() > 0 else np.array([])

            if len(y_true_filtered) > 0:
                filtered_fractions = [
                    np.sum(y_true_filtered == i) / len(y_true_filtered)
                    for i in [0, 1]
                ]
            else:
                filtered_fractions = [0, 0]

            fractions_per_threshold.append(filtered_fractions)

        fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
        bar_width = 0.7
        x_positions = np.arange(len(ct_thresholds))
        bottoms = np.zeros(len(ct_thresholds))

        for i, label in enumerate(target_names):
            class_heights = [fractions_per_threshold[j][i] for j in range(len(ct_thresholds))]
            ax_bar.bar(x_positions, class_heights, width=bar_width, bottom=bottoms, label=label)
            bottoms += class_heights

        ax_bar.set_xticks(x_positions)
        ax_bar.set_xticklabels([str(ct) for ct in ct_thresholds])
        ax_bar.set_ylabel("Fraction of traces per class")
        ax_bar.set_xlabel("CT threshold")
        ax_bar.set_title("Class fractions across CT thresholds")
        ax_bar.legend()

        plt.tight_layout()
        plt.show()


@app.command()
def test(model_uri: str,
         test_tsv: Path,
         ) -> None:
    pass

@app.command()
def predict(model_uri: str,
            test_tsv: Path,
            output_tsv: Path,
            ) -> None:
    pass
