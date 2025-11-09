from pathlib import Path

import mlflow
import keras
import tensorflow as tf

import pandas as pd
import numpy as np

from IAV_Classification.utils import repo_dir, set_seeds
from IAV_Classification.mcc import MatthewsCorrelationCoefficient
from IAV_Classification.data import DataSplit, load_dataset_split

from IAV_Classification.model import make_model, monte_carlo_predict_samples

import typer

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay, classification_report, accuracy_score,
    confusion_matrix, matthews_corrcoef
)

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.colors import Normalize
from scipy.stats import wasserstein_distance
import warnings

app = typer.Typer()


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

    # where ModelCheckpoint in fit() saves the model
    chkpoint = repo_dir / 'mlmodels' / f'{run_name}.keras'

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(dict(
            seed=seed, dataset=dataset, run_name=run_name, fold=fold,
            patience=patience, epochs=epochs, batch_size=batch_size
        ))

        datasplit = load_dataset_split(dataset, n_splits=5, seed=seed, fold=fold)
        model = make_model(datasplit.x_train.shape[1:])

        history = None
        try:
            history = fit(
                model=model,
                datasplit=datasplit,
                run_name=run_name,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                verbose=verbose,
            )
        except KeyboardInterrupt:
            print("\n Training interrupted by user (KeyboardInterrupt).")
            print("Continuing with evaluation using the current/best available weights...")

        # If we have a saved best model, prefer it for evaluation
        if chkpoint.exists():
            try:
                model = keras.models.load_model(chkpoint)
                print(f"Loaded best checkpoint: {chkpoint}")
            except Exception as e:
                print(f"Could not load checkpoint at {chkpoint}: {e}")

        # -------------------
        # Plot Training History (only if we have it)
        # -------------------
        if history is not None:
            plt.figure(figsize=(12, 5))

            # Loss
            plt.subplot(1, 2, 1)
            plt.plot(history.history.get('loss', []), label='Training Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

            # Accuracy (handle both 'acc' and 'accuracy' naming)
            acc_key = 'accuracy' if 'accuracy' in history.history else ('acc' if 'acc' in history.history else None)
            val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else ('val_acc' if 'val_acc' in history.history else None)

            if acc_key or val_acc_key:
                plt.subplot(1, 2, 2)
                if acc_key:
                    plt.plot(history.history[acc_key], label='Training Accuracy')
                if val_acc_key:
                    plt.plot(history.history[val_acc_key], label='Validation Accuracy')
                plt.title('Model Accuracy')
                plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

            plt.suptitle('Training History', fontsize=16, weight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        # -------------------
        # Evaluation (always run)
        # -------------------
        y_pred_prob = model.predict(datasplit.x_test).astype('float32').flatten()
        y_pred_bin = (y_pred_prob > 0.5).astype(int)

        cm = confusion_matrix(datasplit.y_test, y_pred_bin)
        print("\nConfusion Matrix:")
        print(cm)

        acc = accuracy_score(datasplit.y_test, y_pred_bin)
        mcc = matthews_corrcoef(datasplit.y_test, y_pred_bin)

        print(f"\nAccuracy: {acc:.4f}")
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

        # Log what we safely can
        metrics_to_log = {'accuracy': acc, 'mcc': mcc}
        if history is not None and 'val_loss' in history.history:
            try:
                metrics_to_log['val_loss_best'] = float(np.min(history.history['val_loss']))
            except Exception:
                pass
        mlflow.log_metrics(metrics_to_log)

        # Save model to MLflow (use the checkpointed/best model if loaded)
        signature = mlflow.models.infer_signature(
            datasplit.x_train[:4],
            model.predict(datasplit.x_train[:4])
        )
        mlflow.keras.log_model(model, run_name, signature=signature)


@app.command()
def evaluate(model_uri: str,
             dataset: str = 'normalized.tsv',
             seed: int | None = 42,
             fold: int = 1,
             include_uncertainty: bool = True,
             use_post_analysis: bool = True,
             n_mc_samples: int = 100,
             ct_threshold: float = 0,
             spaced_threshold: bool = True
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

    # --- ROC Curve ---
    ax[0, 1].plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
    ax[0, 1].set_xlabel("False Positive Rate")
    ax[0, 1].set_ylabel("True Positive Rate")
    ax[0, 1].set_xlim(0, 1)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].legend(loc="lower right")

    # --- Precision-Recall Curve ---
    baseline_precision = np.mean(datasplit.y_test)
    ax[1, 0].plot(recall, precision, label=f"AP = {ap_score:.4f}")
    ax[1, 0].hlines(
        y=baseline_precision, xmin=0, xmax=1,
        colors="gray", linestyles="--", label=f"Baseline = {baseline_precision:.2f}"
    )
    ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, pad=15)
    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")
    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].legend(loc="lower left")

    # --- Metrics summary box ---
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

            # --- Filtered ROC Curve ---
            ax[0, 1].plot(fpr_filt, tpr_filt, label=f"AUC = {auc_filt:.4f}")
            ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
            ax[0, 1].set_xlabel("False Positive Rate")
            ax[0, 1].set_ylabel("True Positive Rate")
            ax[0, 1].set_xlim(0, 1)
            ax[0, 1].set_ylim(0, 1)
            ax[0, 1].legend(loc="lower right")

            # --- Filtered Precision-Recall Curve ---
            ax[1, 0].plot(recall_filt, precision_filt, label=f"AP = {ap_filt:.4f}")
            baseline_precision_filt = np.mean(y_true_filt)
            ax[1, 0].hlines(
                y=baseline_precision_filt, xmin=0, xmax=1,
                colors="gray", linestyles="--", label=f"Baseline = {baseline_precision_filt:.2f}"
            )
            ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, pad=15)
            ax[1, 0].set_xlabel("Recall")
            ax[1, 0].set_ylabel("Precision")
            ax[1, 0].set_xlim(0, 1)
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

        # --- Strain label colors (consistent across all plots) ---
        strain_palette = {'PR8': 'tab:blue', 'X31': 'tab:orange'}

        # -------------------
        # ECDF Plot
        # -------------------
        fig, ax = plt.subplots(figsize=(6, 5))

        # Get min/max for x-axis limits
        x_min, x_max = post_df["min_wasserstein"].min(), post_df["min_wasserstein"].max()

        for label, idx in zip(target_names, [0, 1]):
            values = post_df.loc[post_df["y_true"] == idx, "min_wasserstein"].values
            if len(values) == 0:
                continue
            x = np.sort(values)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(
                x, y,
                where="post",
                color=strain_palette.get(label, "gray"),
                alpha=0.8,
                label=f"{label} (n={len(values)})"
            )

        # --- Remove whitespace ---
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)

        # --- Labels and title ---
        ax.set_xlabel("min_wasserstein")
        ax.set_ylabel("ECDF")
        ax.set_title(f"ECDF of min_wasserstein by class (CT>{ct_threshold})")
        ax.legend(title="Class")

        plt.tight_layout()
        plt.show()

        # -------------------
        # Bar plot for fractions across CT thresholds
        # -------------------
        #min_w = post_df["min_wasserstein"].min()
        max_w = post_df["min_wasserstein"].max()

        # Use same spaced-threshold logic for consistency
        if spaced_threshold:
            ct_thresholds = np.linspace(0, max_w, num=6)[:-1]  # → 5 thresholds (6−1)
        else:
            ct_thresholds = [ct_threshold]

        ct_thresholds = np.round(ct_thresholds, 3)

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
            ax_bar.bar(
                x_positions,
                class_heights,
                width=bar_width,
                bottom=bottoms,
                color=strain_palette.get(label, "gray"),
                label=label
            )
            bottoms += class_heights

        ax_bar.set_xticks(x_positions)
        ax_bar.set_xticklabels([str(ct) for ct in ct_thresholds])
        ax_bar.set_ylabel("Fraction of traces per class")
        ax_bar.set_xlabel("CT threshold")
        ax_bar.set_title("Class fractions across CT thresholds")
        ax_bar.legend()

        plt.tight_layout()
        plt.show()

        # -------------------
        # Second Evaluation Report (Filtered by CT threshold)
        # -------------------
        if spaced_threshold:
            min_w = post_df["min_wasserstein"].min()
            max_w = post_df["min_wasserstein"].max()
            ct_thresholds = np.linspace(min_w, max_w, num=6)[1:-1]  # 4 evenly spaced internal thresholds
            ct_thresholds = np.round(ct_thresholds, 3)
        else:
            ct_thresholds = [ct_threshold]

        for ct_threshold in ct_thresholds:
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
                class_report_filt = classification_report(
                    y_true_filt, y_pred_filt, target_names=target_names, digits=3
                )

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

                # --- Filtered ROC Curve ---
                ax[0, 1].plot(fpr_filt, tpr_filt, label=f"AUC = {auc_filt:.4f}")
                ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
                ax[0, 1].set_xlabel("False Positive Rate")
                ax[0, 1].set_ylabel("True Positive Rate")
                ax[0, 1].set_xlim(0, 1)
                ax[0, 1].set_ylim(0, 1)
                ax[0, 1].legend(loc="lower right")

                # --- Filtered Precision-Recall Curve ---
                ax[1, 0].plot(recall_filt, precision_filt, label=f"AP = {ap_filt:.4f}")
                baseline_precision_filt = np.mean(y_true_filt)
                ax[1, 0].hlines(
                    y=baseline_precision_filt, xmin=0, xmax=1,
                    colors="gray", linestyles="--", label=f"Baseline = {baseline_precision_filt:.2f}"
                )
                ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, pad=15)
                ax[1, 0].set_xlabel("Recall")
                ax[1, 0].set_ylabel("Precision")
                ax[1, 0].set_xlim(0, 1)
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

                plt.suptitle(f"Filtered Evaluation Report (CT>{ct_threshold})", fontsize=20, weight="bold")
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                plt.show()

                print("\n================ Filtered Evaluation Report ================\n")
                print(f"CT threshold: {ct_threshold}")
                print(f"Traces remaining: {total_filtered}")
                print(f"PR8: {pr8_filtered}")
                print(f"X31: {x31_filtered}")
                print(f"Accuracy (CT>{ct_threshold}): {acc_filt:.4f}")
                print(f"MCC (CT>{ct_threshold}): {mcc_filt:.4f}")
                print(class_report_filt)
                print(f"ROC-AUC (CT>{ct_threshold}): {auc_filt:.4f}")
                print(f"Average Precision (PR-AUC): {ap_filt:.4f}")

@app.command()
def test(model_uri: str,
         dataset: str = 'test.tsv',
         include_uncertainty: bool = True,
         use_post_analysis: bool = True,
         n_mc_samples: int = 100,
         ct_threshold: float = 0.7,
         spaced_threshold: bool = True,
         ) -> None:
    """
    Evaluate a trained model on a held-out test dataset (xxx_test.tsv).

    Args:
        model_uri (str): Path or URI to the trained model (MLflow or local .keras file)
        dataset (Path): Path to the *_test.tsv file
        include_uncertainty (bool): If True, perform Monte Carlo dropout predictions
        use_post_analysis (bool): If True, perform Wasserstein CT filtering analysis
        n_mc_samples (int): Number of MC samples for uncertainty estimation
        ct_threshold (float): Threshold for filtering by min Wasserstein distance
    """

    # -------------------
    # Load model
    # -------------------
    model = mlflow.keras.load_model(model_uri)

    # -------------------
    # Load test dataset
    # -------------------
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        # Try to find it in your default data directory
        candidate = repo_dir / 'data' / dataset_path.name
        if candidate.exists():
            dataset_path = candidate
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path, sep='\t').set_index(['label', 'video', 'trace'])

    x_test = data.values[:, :, np.newaxis]
    labels = data.index.get_level_values('label')
    if 'PR8' in labels:
        y_test = np.array([int(label == 'PR8') for label in labels])
    else:
        y_test = LabelEncoder().fit_transform(labels)

    # -------------------
    # Monte Carlo predictions (optional)
    # -------------------
    if include_uncertainty:
        print("Running Monte Carlo prediction...")
        y_pred_prob, probs_try = monte_carlo_predict_samples(
            model, x_test, n_mc_samples=n_mc_samples
        )

        # Fix for binary classification
        if probs_try.shape[2] == 1:
            probs_try = np.concatenate([1 - probs_try, probs_try], axis=2)

    else:
        y_pred_prob = model.predict(x_test).astype("float32")
        probs_try = None

    y_pred_prob = y_pred_prob.flatten()
    y_pred_bin = (y_pred_prob > 0.5).astype(int)

    # -------------------
    # Standard evaluation metrics
    # -------------------
    acc = accuracy_score(y_test, y_pred_bin)
    mcc = matthews_corrcoef(y_test, y_pred_bin)

    target_names = ["X31", "PR8"]
    class_report = classification_report(y_test, y_pred_bin,
                                         target_names=target_names, digits=3)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    ap_score = average_precision_score(y_test, y_pred_prob)
    baseline_precision = np.mean(y_test)

    total_traces = len(y_test)
    pr8_traces = np.sum(y_test == 1)
    x31_traces = np.sum(y_test == 0)

    # -------------------
    # Evaluation plots
    # -------------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_bin, ax=ax[0, 0], cmap="viridis",
        colorbar=True, normalize='true', display_labels=target_names
    )
    im = ax[0, 0].images[-1]
    im.set_norm(Normalize(vmin=0, vmax=1))
    cbar = disp.ax_.images[-1].colorbar
    if cbar:
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[0, 0].set_title("Confusion Matrix", fontsize=14, pad=15)
    # --- ROC Curve ---
    ax[0, 1].plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
    ax[0, 1].set_xlabel("False Positive Rate")
    ax[0, 1].set_ylabel("True Positive Rate")
    ax[0, 1].set_xlim(0, 1)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].legend(loc="lower right")

    # --- Precision-Recall Curve ---
    ax[1, 0].plot(recall, precision, label=f"AP = {ap_score:.4f}")
    ax[1, 0].hlines(
        y=baseline_precision, xmin=0, xmax=1,
        colors="gray", linestyles="--", label=f"Baseline = {baseline_precision:.2f}"
    )
    ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, pad=15)
    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")
    ax[1, 0].set_xlim(0, 1)
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
    plt.suptitle("Model Test Report", fontsize=20, weight="bold")
    plt.show()

    print("\n================ Test Evaluation Report ================\n")
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
            "y_true": y_test,
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

            # --- Filtered ROC Curve ---
            ax[0, 1].plot(fpr_filt, tpr_filt, label=f"AUC = {auc_filt:.4f}")
            ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
            ax[0, 1].set_xlabel("False Positive Rate")
            ax[0, 1].set_ylabel("True Positive Rate")
            ax[0, 1].set_xlim(0, 1)
            ax[0, 1].set_ylim(0, 1)
            ax[0, 1].legend(loc="lower right")

            # --- Filtered Precision-Recall Curve ---
            ax[1, 0].plot(recall_filt, precision_filt, label=f"AP = {ap_filt:.4f}")
            baseline_precision_filt = np.mean(y_true_filt)
            ax[1, 0].hlines(
                y=baseline_precision_filt, xmin=0, xmax=1,
                colors="gray", linestyles="--", label=f"Baseline = {baseline_precision_filt:.2f}"
            )
            ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, pad=15)
            ax[1, 0].set_xlabel("Recall")
            ax[1, 0].set_ylabel("Precision")
            ax[1, 0].set_xlim(0, 1)
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

        # --- Strain label colors (consistent across all plots) ---
        strain_palette = {'PR8': 'tab:blue', 'X31': 'tab:orange'}

        # -------------------
        # ECDF Plot
        # -------------------
        fig, ax = plt.subplots(figsize=(6, 5))

        x_min, x_max = post_df["min_wasserstein"].min(), post_df["min_wasserstein"].max()

        for label, idx in zip(target_names, [0, 1]):
            values = post_df.loc[post_df["y_true"] == idx, "min_wasserstein"].values
            if len(values) == 0:
                continue
            x = np.sort(values)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(
                x,
                y,
                where="post",
                color=strain_palette.get(label, "gray"),  # consistent with strain colors
                alpha=0.8,
                label=f"{label} (n={len(values)})"
            )

        # --- Remove whitespace around axes ---
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)

        # --- Axis labels and styling ---
        ax.set_xlabel("min_wasserstein")
        ax.set_ylabel("ECDF")
        ax.set_title(f"ECDF of min_wasserstein by class (CT>{ct_threshold})")
        ax.legend(title="Class")

        plt.tight_layout()
        plt.show()

        # -------------------
        # Bar plot for fractions across CT thresholds
        # -------------------
        #min_w = post_df["min_wasserstein"].min()
        max_w = post_df["min_wasserstein"].max()

        # Use same spaced-threshold logic for consistency
        if spaced_threshold:
            ct_thresholds = np.linspace(0, max_w, num=6)[:-1]  # → 5 thresholds (6−1)
        else:
            ct_thresholds = [ct_threshold]

        ct_thresholds = np.round(ct_thresholds, 3)

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
            ax_bar.bar(
                x_positions,
                class_heights,
                width=bar_width,
                bottom=bottoms,
                color=strain_palette.get(label, "gray"),  # consistent with ECDF and scatter plots
                label=label
            )
            bottoms += class_heights

        ax_bar.set_xticks(x_positions)
        ax_bar.set_xticklabels([str(ct) for ct in ct_thresholds])
        ax_bar.set_ylabel("Fraction of traces per class")
        ax_bar.set_xlabel("CT threshold")
        ax_bar.set_title("Class fractions across CT thresholds")
        ax_bar.legend()

        plt.tight_layout()
        plt.show()

        # -------------------
        # Second Evaluation Report (Filtered by CT threshold)
        # -------------------
        if spaced_threshold:
            min_w = post_df["min_wasserstein"].min()
            max_w = post_df["min_wasserstein"].max()
            ct_thresholds = np.linspace(min_w, max_w, num=6)[1:-1]  # 4 evenly spaced internal thresholds
            ct_thresholds = np.round(ct_thresholds, 3)
        else:
            ct_thresholds = [ct_threshold]

        for ct_threshold in ct_thresholds:
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
                class_report_filt = classification_report(
                    y_true_filt, y_pred_filt, target_names=target_names, digits=3
                )

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

                # --- Filtered ROC Curve ---
                ax[0, 1].plot(fpr_filt, tpr_filt, label=f"AUC = {auc_filt:.4f}")
                ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
                ax[0, 1].set_xlabel("False Positive Rate")
                ax[0, 1].set_ylabel("True Positive Rate")
                ax[0, 1].set_xlim(0, 1)
                ax[0, 1].set_ylim(0, 1)
                ax[0, 1].legend(loc="lower right")

                # --- Filtered Precision-Recall Curve ---
                ax[1, 0].plot(recall_filt, precision_filt, label=f"AP = {ap_filt:.4f}")
                baseline_precision_filt = np.mean(y_true_filt)
                ax[1, 0].hlines(
                    y=baseline_precision_filt, xmin=0, xmax=1,
                    colors="gray", linestyles="--", label=f"Baseline = {baseline_precision_filt:.2f}"
                )
                ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, pad=15)
                ax[1, 0].set_xlabel("Recall")
                ax[1, 0].set_ylabel("Precision")
                ax[1, 0].set_xlim(0, 1)
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

                plt.suptitle(f"Filtered Test Report (CT>{ct_threshold})", fontsize=20, weight="bold")
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                plt.show()

                print("\n================ Filtered Test Report ================\n")
                print(f"CT threshold: {ct_threshold}")
                print(f"Traces remaining: {total_filtered}")
                print(f"PR8: {pr8_filtered}")
                print(f"X31: {x31_filtered}")
                print(f"Accuracy (CT>{ct_threshold}): {acc_filt:.4f}")
                print(f"MCC (CT>{ct_threshold}): {mcc_filt:.4f}")
                print(class_report_filt)
                print(f"ROC-AUC (CT>{ct_threshold}): {auc_filt:.4f}")
                print(f"Average Precision (PR-AUC): {ap_filt:.4f}")

@app.command()
def predict_labeled(model_uri: str,
                    dataset: Path,
                    output_tsv: Path,
                    threshold: float = 0.5) -> None:
    """
    Predict on a labeled dataset formatted as:
    label, video, trace, time_0, time_10, ..., time_600

    Args:
        model_uri (str): Path or URI to the trained model (MLflow or .keras)
        dataset (Path): Path to labeled dataset (TSV)
        output_tsv (Path): Path to save predictions
        threshold (float): Probability cutoff for binary classification (default=0.5)
    """

    # -------------------
    # Load model
    # -------------------
    print(f"Loading model from: {model_uri}")
    model = mlflow.keras.load_model(model_uri)

    # -------------------
    # Load data
    # -------------------
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        candidate = repo_dir / 'data' / dataset_path.name
        if candidate.exists():
            dataset_path = candidate
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Don’t set index here
    data = pd.read_csv(dataset_path, sep='\t')

    # Expect first 3 columns: label, video, trace
    expected_prefix = ["label", "video", "trace"]
    if not all(c in data.columns[:3].tolist() for c in expected_prefix):
        raise ValueError(
            f"The first three columns must be 'label', 'video', 'trace'. Got: {data.columns[:3].tolist()}"
        )

    # Extract parts
    labels = data.iloc[:, 0].astype(str)
    videos = data.iloc[:, 1].astype(str)
    traces = data.iloc[:, 2].astype(str)

    # Numeric features (from 4th column onward)
    x_data = data.iloc[:, 3:].values[:, :, np.newaxis]

    # -------------------
    # Encode labels (binary classification)
    # -------------------
    if set(labels.unique()) == {"X31", "PR8"}:
        y_true = np.array([1 if l == "PR8" else 0 for l in labels])
        target_names = ["X31", "PR8"]
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_true = le.fit_transform(labels)
        target_names = le.classes_.tolist()

    # -------------------
    # Run predictions
    # -------------------
    print("Running predictions...")
    y_pred_prob = model.predict(x_data).astype("float32").flatten()
    y_pred_bin = (y_pred_prob > threshold).astype(int)

    # -------------------
    # Compute metrics
    # -------------------
    acc = accuracy_score(y_true, y_pred_bin)
    mcc = matthews_corrcoef(y_true, y_pred_bin)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    ap_score = average_precision_score(y_true, y_pred_prob)
    class_report = classification_report(y_true, y_pred_bin, target_names=target_names)
    cm = confusion_matrix(y_true, y_pred_bin)

    # -------------------
    # Print results
    # -------------------
    print("\n================ Prediction Evaluation ================\n")
    print(f"Accuracy: {acc:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"ROC-AUC: {auc_score:.4f}")
    print(f"Average Precision (PR-AUC): {ap_score:.4f}")
    print("\nClassification Report:\n")
    print(class_report)
    print("Confusion Matrix:\n", cm)

    # -------------------
    # Save predictions
    # -------------------
    pred_df = pd.DataFrame({
        "label_true": labels,
        "video": videos,
        "trace": traces,
        "y_pred_prob": y_pred_prob,
        "y_pred_bin": y_pred_bin,
        "predicted_label": [target_names[i] for i in y_pred_bin],
    })

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_tsv, sep="\t", index=False)
    print(f"\nPredictions saved to: {output_tsv}")

    # -------------------
    # Visualization (2×2 with metrics panel)
    # -------------------
    # counts per class for the summary text
    total_traces = len(y_true)
    counts_per_class = "\n".join(
        f"{name}: {int(np.sum(y_true == i))}" for i, name in enumerate(target_names)
    )

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred_bin, ax=ax[0, 0], cmap="viridis",
        colorbar=True, display_labels=target_names, normalize="true"
    )
    ax[0, 0].set_title("Confusion Matrix", fontsize=14, pad=15)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    ax[0, 1].plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
    ax[0, 1].set_xlabel("False Positive Rate")
    ax[0, 1].set_ylabel("True Positive Rate")
    ax[0, 1].set_xlim(0, 1)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].legend(loc="lower right")

    # Precision–Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    baseline = np.mean(y_true)
    ax[1, 0].plot(recall, precision, label=f"AP = {ap_score:.4f}")
    ax[1, 0].hlines(baseline, xmin=0, xmax=1, color="gray", linestyle="--",
                    label=f"Baseline = {baseline:.2f}")
    ax[1, 0].set_title("Precision–Recall Curve", fontsize=14, pad=15)
    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")
    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].legend(loc="lower left")

    # Metrics summary box (Accuracy, MCC, Classification Report)
    metrics_text = (
        f"Total traces: {total_traces}\n"
        f"{counts_per_class}\n\n"
        f"Accuracy: {acc:.4f}\n"
        f"MCC: {mcc:.4f}\n\n"
        f"Classification Report:\n{class_report}"
    )
    ax[1, 1].axis("off")
    ax[1, 1].text(0, 1, metrics_text, ha="left", va="top",
                  family="monospace", fontsize=11)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle("Prediction Evaluation Report", fontsize=20, weight="bold")
    plt.show()


@app.command()
def predict_unlabeled(model_uri: str,
                      data_tsv: Path,
                      output_tsv: Path,
                      threshold: float = 0.5) -> None:
    """
    Predict on an unlabeled dataset formatted as:
    trace, time_0, time_10, ..., time_600

    Loads data the same way as predict_labeled (path fallback to repo_dir/data).
    Plots histogram of predicted probabilities with viridis colormap.
    """

    # -------------------
    # Load model
    # -------------------
    print(f"Loading model from: {model_uri}")
    model = mlflow.keras.load_model(model_uri)

    # -------------------
    # Load data (same style as predict_labeled)
    # -------------------
    dataset_path = Path(data_tsv)
    if not dataset_path.exists():
        candidate = repo_dir / 'data' / dataset_path.name
        if candidate.exists():
            dataset_path = candidate
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading unlabeled dataset from: {dataset_path}")
    data = pd.read_csv(dataset_path, sep="\t")

    # Expect first column: trace
    if "trace" not in data.columns[:1].tolist():
        raise ValueError(
            "Input TSV must contain a 'trace' column as the first column."
        )

    # Extract trace identifiers
    traces = data["trace"].astype(str)

    # Extract numeric features (all columns after 'trace')
    feature_cols = data.columns[1:]
    if len(feature_cols) == 0:
        raise ValueError("No timepoint columns found after 'trace'.")

    x_data = data.loc[:, feature_cols].values[:, :, np.newaxis]

    # -------------------
    # Run predictions
    # -------------------
    print("Running model predictions...")
    y_pred_prob = model.predict(x_data).astype("float32").flatten()
    y_pred_bin = (y_pred_prob > threshold).astype(int)

    # Map binary outputs to class names
    target_names = ["X31", "PR8"]
    predicted_labels = [target_names[i] for i in y_pred_bin]

    # -------------------
    # Save results
    # -------------------
    pred_df = pd.DataFrame({
        "trace": traces,
        "y_pred_prob": y_pred_prob,
        "y_pred_bin": y_pred_bin,
        "predicted_label": predicted_labels
    })

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_tsv, sep="\t", index=False)
    print(f"\nPredictions saved to: {output_tsv}")

    # -------------------
    # Probability histogram (viridis colormap)
    # -------------------
    plt.figure(figsize=(6, 4))
    counts, bins, patches = plt.hist(y_pred_prob, bins=30)
    # Color each bar by its height using viridis
    norm = mcolors.Normalize(vmin=counts.min() if counts.size else 0,
                             vmax=counts.max() if counts.size else 1)
    cmap = cm.get_cmap("viridis")
    for c, p in zip(counts, patches):
        p.set_facecolor(cmap(norm(c)))
    # Threshold line
    plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold}")
    plt.title("Distribution of Predicted Probabilities")
    plt.xlabel("Predicted Probability (PR8 class)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nPrediction complete. (Unlabeled data — no metrics computed.)")