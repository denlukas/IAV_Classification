from pathlib import Path
import tensorflow as tf
import mlflow
import keras
import pandas as pd
import numpy as np
import typer

from IAV_Classification.utils import repo_dir, set_seeds
from IAV_Classification.data import DataSplit, load_dataset_split
from IAV_Classification.model import make_model, monte_carlo_predict_samples

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

    # --- saving helper (only addition) ---
    output_dir = repo_dir / "IAV_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tag = f"{Path(model_uri).stem}_{Path(dataset).stem}_fold{fold}_seed{seed}"

    def savefig(fig, label: str):
        path = output_dir / f"{base_tag}_{label}.png"
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[saved] {path}")

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

    # ROC
    ax[0, 1].plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
    ax[0, 1].set_xlabel("False Positive Rate")
    ax[0, 1].set_ylabel("True Positive Rate")
    ax[0, 1].set_xlim(0, 1)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].legend(loc="lower right")

    # PR
    baseline_precision = np.mean(datasplit.y_test)
    ax[1, 0].plot(recall, precision, label=f"AP = {ap_score:.4f}")
    ax[1, 0].hlines(y=baseline_precision, xmin=0, xmax=1,
                    colors="gray", linestyles="--",
                    label=f"Baseline = {baseline_precision:.2f}")
    ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, pad=15)
    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")
    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].legend(loc="lower left")

    # metrics box
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
    savefig(fig, "eval_main")
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
            min_w = min((wasserstein_distance(sample_probs[:, top_class], sample_probs[:, c])
                         for c in other_classes), default=0.0)
            list_min_w_distances.append(min_w)

        # --- class mapping ---
        idx_to_name = {0: "X31", 1: "PR8"}
        target_names = ["X31", "PR8"]

        # Build a friendly table
        post_df = pd.DataFrame({
            "y_true": datasplit.y_test.astype(int),  # 0/1
            "predicted_label": predicted_label.astype(int),  # 0/1
            "y_true_name": [idx_to_name[i] for i in datasplit.y_test],  # "X31"/"PR8"
            "predicted_name": [idx_to_name[i] for i in predicted_label],  # "X31"/"PR8"
            "prediction_probability": prediction_probability,
            "prediction_std": prediction_stds,
            "min_wasserstein": list_min_w_distances
        })

        # save the post-analysis table as TSV
        post_tsv = output_dir / f"{base_tag}_post_analysis.tsv"
        post_df.to_csv(post_tsv, sep="\t", index=False)
        print(f"[saved] {post_tsv}")

        # Histogram (viridis gradient)
        fig_hist = plt.figure(figsize=(6, 4))
        counts, bins = np.histogram(post_df['min_wasserstein'], bins=50)
        norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
        cmap = plt.cm.viridis
        colors = cmap(norm(counts))
        for i in range(len(bins) - 1):
            plt.bar(bins[i], counts[i], width=bins[i + 1] - bins[i],
                    color=colors[i], align='edge')
        plt.xlabel("Minimal Wasserstein distance")
        plt.ylabel("Number of samples")
        plt.title("Distribution of minimal Wasserstein distance")
        savefig(fig_hist, "hist_min_wasserstein")
        plt.show()

        # Single-cut filtered eval (CT > ct_threshold)
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

            fig_filt, ax = plt.subplots(2, 2, figsize=(10, 8))
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
            ax[0, 1].set_xlabel("False Positive Rate")
            ax[0, 1].set_ylabel("True Positive Rate")
            ax[0, 1].set_xlim(0, 1)
            ax[0, 1].set_ylim(0, 1)
            ax[0, 1].legend(loc="lower right")

            ax[1, 0].plot(recall_filt, precision_filt, label=f"AP = {ap_filt:.4f}")
            baseline_precision_filt = np.mean(y_true_filt)
            ax[1, 0].hlines(y=baseline_precision_filt, xmin=0, xmax=1,
                            colors="gray", linestyles="--",
                            label=f"Baseline = {baseline_precision_filt:.2f}")
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
            savefig(fig_filt, f"eval_filtered_CTgt_{ct_threshold:g}")
            plt.show()

        # ECDF
        fig_ecdf, ax = plt.subplots(figsize=(6, 5))
        x_min, x_max = post_df["min_wasserstein"].min(), post_df["min_wasserstein"].max()
        strain_palette = {'PR8': 'tab:blue', 'X31': 'tab:orange'}
        for label, idx in zip(target_names, [0, 1]):
            values = post_df.loc[post_df["y_true"] == idx, "min_wasserstein"].values
            if len(values) == 0:
                continue
            x = np.sort(values)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where="post", color=strain_palette.get(label, "gray"),
                    alpha=0.8, label=f"{label} (n={len(values)})")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel("min_wasserstein")
        ax.set_ylabel("ECDF")
        ax.set_title(f"ECDF of min_wasserstein by class (CT>{ct_threshold})")
        ax.legend(title="Class")
        plt.tight_layout()
        savefig(fig_ecdf, "ecdf_min_w")
        plt.show()

        # Bar plot for fractions across CT thresholds
        max_w = post_df["min_wasserstein"].max()
        if spaced_threshold:
            ct_thresholds = np.linspace(0, max_w, num=6)[:-1]
        else:
            ct_thresholds = [ct_threshold]
        ct_thresholds = np.round(ct_thresholds, 3)

        fractions_per_threshold = []
        for ct in ct_thresholds:
            mask_ct = post_df['min_wasserstein'] >= ct
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
            ax_bar.bar(x_positions, class_heights, width=bar_width, bottom=bottoms,
                       color=strain_palette.get(label, "gray"), label=label)
            bottoms += class_heights
        ax_bar.set_xticks(x_positions)
        ax_bar.set_xticklabels([str(ct) for ct in ct_thresholds])
        ax_bar.set_ylabel("Fraction of traces per class")
        ax_bar.set_xlabel("CT threshold")
        ax_bar.set_title("Class fractions across CT thresholds")
        ax_bar.legend()
        plt.tight_layout()
        savefig(fig_bar, "ct_fractions")
        plt.show()

        # Loop over spaced thresholds (second evaluation report)
        if spaced_threshold:
            min_w = post_df["min_wasserstein"].min()
            max_w = post_df["min_wasserstein"].max()
            ct_thresholds = np.linspace(min_w, max_w, num=6)[1:-1]
            ct_thresholds = np.round(ct_thresholds, 3)
        else:
            ct_thresholds = [ct_threshold]

        for ct_th in ct_thresholds:
            mask_ct = post_df["min_wasserstein"] > ct_th
            filtered_df = post_df[mask_ct]
            if len(filtered_df) == 0:
                continue

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

            fig_loop, ax = plt.subplots(2, 2, figsize=(10, 8))
            disp_filt = ConfusionMatrixDisplay.from_predictions(
                y_true_filt, y_pred_filt, ax=ax[0, 0], cmap="viridis",
                colorbar=True, normalize='true', display_labels=target_names
            )
            im_filt = ax[0, 0].images[-1]
            im_filt.set_norm(Normalize(vmin=0, vmax=1))
            cbar_filt = disp_filt.ax_.images[-1].colorbar
            if cbar_filt:
                cbar_filt.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax[0, 0].set_title(f"Confusion Matrix (CT>{ct_th})", fontsize=14, pad=15)

            ax[0, 1].plot(fpr_filt, tpr_filt, label=f"AUC = {auc_filt:.4f}")
            ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
            ax[0, 1].set_xlabel("False Positive Rate")
            ax[0, 1].set_ylabel("True Positive Rate")
            ax[0, 1].set_xlim(0, 1)
            ax[0, 1].set_ylim(0, 1)
            ax[0, 1].legend(loc="lower right")

            ax[1, 0].plot(recall_filt, precision_filt, label=f"AP = {ap_filt:.4f}")
            baseline_precision_filt = np.mean(y_true_filt)
            ax[1, 0].hlines(y=baseline_precision_filt, xmin=0, xmax=1,
                            colors="gray", linestyles="--",
                            label=f"Baseline = {baseline_precision_filt:.2f}")
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

            plt.suptitle(f"Filtered Evaluation Report (CT>{ct_th})", fontsize=20, weight="bold")
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            savefig(fig_loop, f"eval_filtered_CTgt_{ct_th:g}")
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

    # --- saving helper (same pattern as evaluate) ---
    output_dir = repo_dir / "IAV_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tag = f"{Path(model_uri).stem}_{dataset_path.stem}"

    def savefig(fig, label: str):
        path = output_dir / f"{base_tag}_{label}.png"
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[saved] {path}")

    # -------------------
    # Monte Carlo predictions (optional)
    # -------------------
    if include_uncertainty:
        print("Running Monte Carlo prediction...")
        y_pred_prob, probs_try = monte_carlo_predict_samples(
            model, x_test, n_mc_samples=n_mc_samples
        )
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

    # ROC
    ax[0, 1].plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
    ax[0, 1].set_xlabel("False Positive Rate")
    ax[0, 1].set_ylabel("True Positive Rate")
    ax[0, 1].set_xlim(0, 1)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].legend(loc="lower right")

    # PR
    ax[1, 0].plot(recall, precision, label=f"AP = {ap_score:.4f}")
    ax[1, 0].hlines(y=baseline_precision, xmin=0, xmax=1,
                    colors="gray", linestyles="--",
                    label=f"Baseline = {baseline_precision:.2f}")
    ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, pad=15)
    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")
    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].legend(loc="lower left")

    # metrics box
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
    savefig(fig, "test_main")
    plt.show()
    plt.close(fig)

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

        # --- Post-analysis table ---
        idx_to_name = {0: "X31", 1: "PR8"}
        target_names = ["X31", "PR8"]  # keep for plots/metrics order

        post_df = pd.DataFrame({
            "y_true": y_test.astype(int),  # 0/1
            "predicted_label": predicted_label.astype(int),  # 0/1
            "y_true_name": [idx_to_name[i] for i in y_test],  # "X31"/"PR8"
            "predicted_name": [idx_to_name[i] for i in predicted_label],  # "X31"/"PR8"
            "prediction_probability": prediction_probability,
            "prediction_std": prediction_stds,
            "min_wasserstein": list_min_w_distances
        })

        # save the post-analysis table as TSV
        post_tsv = output_dir / f"{base_tag}_post_analysis.tsv"
        post_df.to_csv(post_tsv, sep="\t", index=False)
        print(f"[saved] {post_tsv}")

        # Histogram (viridis gradient)
        fig_hist = plt.figure(figsize=(6, 4))
        counts, bins = np.histogram(post_df['min_wasserstein'], bins=50)
        norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
        cmap = plt.cm.viridis
        colors = cmap(norm(counts))
        for i in range(len(bins) - 1):
            plt.bar(bins[i], counts[i], width=bins[i + 1] - bins[i],
                    color=colors[i], align='edge')
        plt.xlabel("Minimal Wasserstein distance")
        plt.ylabel("Number of samples")
        plt.title("Distribution of minimal Wasserstein distance")
        savefig(fig_hist, "test_hist_min_wasserstein")
        plt.show()
        plt.close(fig_hist)

        # Single-cut filtered eval (CT > ct_threshold)
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

            fig_filt, ax = plt.subplots(2, 2, figsize=(10, 8))
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
            ax[0, 1].set_xlabel("False Positive Rate")
            ax[0, 1].set_ylabel("True Positive Rate")
            ax[0, 1].set_xlim(0, 1)
            ax[0, 1].set_ylim(0, 1)
            ax[0, 1].legend(loc="lower right")

            ax[1, 0].plot(recall_filt, precision_filt, label=f"AP = {ap_filt:.4f}")
            baseline_precision_filt = np.mean(y_true_filt)
            ax[1, 0].hlines(y=baseline_precision_filt, xmin=0, xmax=1,
                            colors="gray", linestyles="--",
                            label=f"Baseline = {baseline_precision_filt:.2f}")
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

            plt.suptitle("Filtered Test Report", fontsize=20, weight="bold")
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            savefig(fig_filt, f"test_filtered_CTgt_{ct_threshold:g}")
            plt.show()
            plt.close(fig_filt)

        # ECDF
        fig_ecdf, ax = plt.subplots(figsize=(6, 5))
        x_min, x_max = post_df["min_wasserstein"].min(), post_df["min_wasserstein"].max()
        strain_palette = {'PR8': 'tab:blue', 'X31': 'tab:orange'}
        for label, idx in zip(target_names, [0, 1]):
            values = post_df.loc[post_df["y_true"] == idx, "min_wasserstein"].values
            if len(values) == 0:
                continue
            x = np.sort(values)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where="post", color=strain_palette.get(label, "gray"),
                    alpha=0.8, label=f"{label} (n={len(values)})")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel("min_wasserstein")
        ax.set_ylabel("ECDF")
        ax.set_title(f"ECDF of min_wasserstein by class (CT>{ct_threshold})")
        ax.legend(title="Class")
        plt.tight_layout()
        savefig(fig_ecdf, "test_ecdf_min_w")
        plt.show()
        plt.close(fig_ecdf)

        # Bar plot for fractions across CT thresholds
        max_w = post_df["min_wasserstein"].max()
        if spaced_threshold:
            ct_thresholds = np.linspace(0, max_w, num=6)[:-1]
        else:
            ct_thresholds = [ct_threshold]
        ct_thresholds = np.round(ct_thresholds, 3)

        fractions_per_threshold = []
        for ct in ct_thresholds:
            mask_ct = post_df['min_wasserstein'] >= ct
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
            ax_bar.bar(x_positions, class_heights, width=bar_width, bottom=bottoms,
                       color=strain_palette.get(label, "gray"), label=label)
            bottoms += class_heights
        ax_bar.set_xticks(x_positions)
        ax_bar.set_xticklabels([str(ct) for ct in ct_thresholds])
        ax_bar.set_ylabel("Fraction of traces per class")
        ax_bar.set_xlabel("CT threshold")
        ax_bar.set_title("Class fractions across CT thresholds")
        ax_bar.legend()
        plt.tight_layout()
        savefig(fig_bar, "test_ct_fractions")
        plt.show()
        plt.close(fig_bar)

        # Loop over spaced thresholds (second evaluation report)
        if spaced_threshold:
            min_w = post_df["min_wasserstein"].min()
            max_w = post_df["min_wasserstein"].max()
            ct_thresholds = np.linspace(min_w, max_w, num=6)[1:-1]
            ct_thresholds = np.round(ct_thresholds, 3)
        else:
            ct_thresholds = [ct_threshold]

        for ct_th in ct_thresholds:
            mask_ct = post_df["min_wasserstein"] > ct_th
            filtered_df = post_df[mask_ct]
            if len(filtered_df) == 0:
                continue

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

            fig_loop, ax = plt.subplots(2, 2, figsize=(10, 8))
            disp_filt = ConfusionMatrixDisplay.from_predictions(
                y_true_filt, y_pred_filt, ax=ax[0, 0], cmap="viridis",
                colorbar=True, normalize='true', display_labels=target_names
            )
            im_filt = ax[0, 0].images[-1]
            im_filt.set_norm(Normalize(vmin=0, vmax=1))
            cbar_filt = disp_filt.ax_.images[-1].colorbar
            if cbar_filt:
                cbar_filt.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax[0, 0].set_title(f"Confusion Matrix (CT>{ct_th})", fontsize=14, pad=15)

            ax[0, 1].plot(fpr_filt, tpr_filt, label=f"AUC = {auc_filt:.4f}")
            ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
            ax[0, 1].set_xlabel("False Positive Rate")
            ax[0, 1].set_ylabel("True Positive Rate")
            ax[0, 1].set_xlim(0, 1)
            ax[0, 1].set_ylim(0, 1)
            ax[0, 1].legend(loc="lower right")

            ax[1, 0].plot(recall_filt, precision_filt, label=f"AP = {ap_filt:.4f}")
            baseline_precision_filt = np.mean(y_true_filt)
            ax[1, 0].hlines(y=baseline_precision_filt, xmin=0, xmax=1,
                            colors="gray", linestyles="--",
                            label=f"Baseline = {baseline_precision_filt:.2f}")
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

            plt.suptitle(f"Filtered Test Report (CT>{ct_th})", fontsize=20, weight="bold")
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            savefig(fig_loop, f"test_filtered_CTgt_{ct_th:g}")
            plt.show()
            plt.close(fig_loop)

            print("\n================ Filtered Test Report ================\n")
            print(f"CT threshold: {ct_th}")
            print(f"Traces remaining: {total_filtered}")
            print(f"PR8: {pr8_filtered}")
            print(f"X31: {x31_filtered}")
            print(f"Accuracy (CT>{ct_th}): {acc_filt:.4f}")
            print(f"MCC (CT>{ct_th}): {mcc_filt:.4f}")
            print(class_report_filt)
            print(f"ROC-AUC (CT>{ct_th}): {auc_filt:.4f}")
            print(f"Average Precision (PR-AUC): {ap_filt:.4f}")


@app.command()
def predict_labeled(model_uri: str,
                    dataset: Path,
                    threshold: float = 0.5) -> None:
    """
    Predict on a labeled dataset formatted as:
    label, video, trace, time_0, time_10, ..., time_600

    Saves a standardized TSV where class mapping is enforced:
      y_true=1 -> PR8, y_true=0 -> X31
      predicted_label=1 -> PR8, predicted_label=0 -> X31
    Also saves a 2x2 evaluation plot.

    Args:
        model_uri (str): Path or URI to the trained model (MLflow or .keras)
        dataset (Path): Path to labeled dataset (TSV)
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
    # Enforce PR8/X31 schema (1=PR8, 0=X31)
    # -------------------
    unique_labels = set(labels.unique())
    if not unique_labels.issubset({"PR8", "X31"}):
        raise ValueError(
            f"This function writes a standardized TSV with class mapping 1=PR8, 0=X31, "
            f"but found other labels: {sorted(unique_labels - {'PR8','X31'})}. "
            f"Please filter the dataset to only PR8/X31."
        )

    # y_true: 1 for PR8, 0 for X31
    y_true = np.array([1 if l == "PR8" else 0 for l in labels], dtype=int)
    target_names = ["X31", "PR8"]  # order used for plots and display
    idx_to_name = {0: "X31", 1: "PR8"}

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
    # Prepare output paths (TSV + Plot)
    # -------------------
    output_dir = repo_dir / "IAV_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tag = f"{Path(model_uri).stem}_{dataset_path.stem}_thr{threshold:g}"

    preds_path = output_dir / f"{base_tag}_predictions.tsv"
    plot_path  = output_dir / f"{base_tag}_prediction_report.png"

    # -------------------
    # Save predictions TSV (standardized schema)
    # -------------------
    pred_df = pd.DataFrame({
        # original identifiers
        "label_true_name": labels,                 # original label text from file
        "video": videos,
        "trace": traces,
        # standardized numeric + readable columns
        "y_pred_prob": y_pred_prob,
        "y_true": y_true,                          # 0/1 where 1=PR8, 0=X31
        "predicted_label": y_pred_bin,  # 0/1 where 1=PR8, 0=X31
        "y_true_classname": [idx_to_name[i] for i in y_true],  # "PR8"/"X31"
        "predicted_classname": [idx_to_name[i] for i in y_pred_bin]  # "PR8"/"X31"
    })

    pred_df.to_csv(preds_path, sep="\t", index=False)
    print(f"\nPredictions saved to: {preds_path}")

    # -------------------
    # Visualization (2×2 with metrics panel)
    # -------------------
    total_traces = len(y_true)
    counts_per_class = "\n".join(
        f"{name}: {int(np.sum(y_true == i))}" for i, name in enumerate(target_names)
    )

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # Confusion Matrix (normalized) with viridis
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred_bin, ax=ax[0, 0], cmap="viridis",
        colorbar=True, display_labels=target_names, normalize="true"
    )
    try:
        disp.ax_.images[-1].set_clim(0, 1)
        cbar = disp.ax_.images[-1].colorbar
        if cbar:
            cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    except Exception:
        pass
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

    # Metrics summary box
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

    plt.suptitle("Prediction Evaluation Report", fontsize=20, weight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save PNG and also show
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
    plt.show()
    plt.close(fig)


@app.command()
def predict_unlabeled(model_uri: str,
                      data_tsv: Path,
                      threshold: float = 0.5) -> None:
    """
    Predict on an unlabeled dataset formatted as:
    trace, time_0, time_10, ..., time_600

    - Saves predictions TSV and histogram PNG to IAV_output/
    - Histogram uses viridis colormap with no borders
    - Pred label mapping: 1=PR8, 0=X31
    """

    # -------------------
    # Load model
    # -------------------
    print(f"Loading model from: {model_uri}")
    model = mlflow.keras.load_model(model_uri)

    # -------------------
    # Load data
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
        raise ValueError("Input TSV must contain a 'trace' column as the first column.")

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

    # Map 0/1 to class names (1=PR8, 0=X31)
    idx_to_name = {0: "X31", 1: "PR8"}
    predicted_classname = np.vectorize(idx_to_name.get)(y_pred_bin)

    # -------------------
    # Output paths
    # -------------------
    output_dir = repo_dir / "IAV_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tag = f"{Path(model_uri).stem}_{dataset_path.stem}_thr{threshold:g}"

    preds_path = output_dir / f"{base_tag}_unlabeled_predictions.tsv"
    plot_path  = output_dir / f"{base_tag}_unlabeled_hist.png"

    # -------------------
    # Save predictions TSV (unlabeled → no true labels/videos)
    # -------------------
    pred_df = pd.DataFrame({
        "trace": traces,
        "y_pred_prob": y_pred_prob,
        "y_pred_bin": y_pred_bin,                 # 0/1 where 1=PR8, 0=X31
        "predicted_classname": predicted_classname  # "PR8"/"X31"
    })
    pred_df.to_csv(preds_path, sep="\t", index=False)
    print(f"\nPredictions saved to: {preds_path}")
    print("Class mapping in TSV: y_pred_bin → 1=PR8, 0=X31")

    # Probability histogram (viridis gradient, no borders)
    fig = plt.figure(figsize=(6, 4))
    counts, bins, patches = plt.hist(y_pred_prob, bins=30, edgecolor='none')  # no borders
    norm = mcolors.Normalize(vmin=counts.min() if counts.size else 0,
                             vmax=counts.max() if counts.size else 1)
    cmap = cm.get_cmap("viridis")
    for c, p in zip(counts, patches):
        p.set_facecolor(cmap(norm(c)))
        p.set_linewidth(0)

    # Threshold line
    plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold}")

    plt.title("Distribution of Predicted Probabilities")
    plt.xlabel("Predicted Probability (PR8 class)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    # Save and show
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[saved] {plot_path}")
    plt.show()
    plt.close(fig)


@app.command()
def predict_labeled_model(
    model_path: Path,
    dataset: Path,
    threshold: float = 0.5
) -> None:
    """
    Predict on a labeled dataset formatted as:
    label, video, trace, time_0, time_10, ..., time_600

    Saves a standardized TSV where class mapping is enforced:
      y_true=1 -> PR8, y_true=0 -> X31
      predicted_label=1 -> PR8, predicted_label=0 -> X31
    Also saves a 2x2 evaluation plot.

    Args:
        model_path (Path): Path to the trained Keras model (.keras file or SavedModel folder)
        dataset (Path): Path to labeled dataset (TSV)
        threshold (float): Probability cutoff for binary classification (default=0.5)
    """

    # -------------------
    # Load model
    # -------------------
    mp = Path(model_path)
    if not mp.exists():
        candidate = repo_dir / "models" / mp.name
        if candidate.exists():
            mp = candidate
        else:
            raise FileNotFoundError(f"Model not found: {model_path} (also checked {candidate})")

    print(f"Loading model from local path: {mp}")
    try:
        model = tf.keras.models.load_model(mp)
    except Exception as e:
        raise RuntimeError(f"Failed to load Keras model from {mp}: {e}")

    # -------------------
    # Load data
    # -------------------
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        candidate = repo_dir / "data" / dataset_path.name
        if candidate.exists():
            dataset_path = candidate
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

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
    x_raw = data.iloc[:, 3:].to_numpy(dtype="float32")

    # -------------------
    # Adjust for expected input length
    # -------------------
    expected_len = model.input_shape[1] if len(model.input_shape) > 1 else x_raw.shape[1]
    current_len = x_raw.shape[1]
    if current_len > expected_len:
        print(f"[warn] Truncating from {current_len} to {expected_len} timepoints to match model.")
        x_raw = x_raw[:, :expected_len]
    elif current_len < expected_len:
        print(f"[warn] Padding from {current_len} to {expected_len} timepoints to match model.")
        pad = expected_len - current_len
        last = x_raw[:, [-1]]
        x_raw = np.concatenate([x_raw, np.repeat(last, pad, axis=1)], axis=1)

    x_data = x_raw[:, :, np.newaxis]

    # -------------------
    # Enforce PR8/X31 schema
    # -------------------
    unique_labels = set(labels.unique())
    if not unique_labels.issubset({"PR8", "X31"}):
        raise ValueError(
            f"This function writes a standardized TSV with class mapping 1=PR8, 0=X31, "
            f"but found other labels: {sorted(unique_labels - {'PR8','X31'})}. "
            f"Please filter the dataset to only PR8/X31."
        )

    y_true = np.array([1 if l == "PR8" else 0 for l in labels], dtype=int)
    target_names = ["X31", "PR8"]
    idx_to_name = {0: "X31", 1: "PR8"}

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
    # Prepare output paths
    # -------------------
    output_dir = repo_dir / "IAV_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tag = f"{mp.stem}_{dataset_path.stem}_thr{threshold:g}"

    preds_path = output_dir / f"{base_tag}_predictions_model.tsv"
    plot_path  = output_dir / f"{base_tag}_prediction_report_model.png"

    # -------------------
    # Save predictions TSV
    # -------------------
    pred_df = pd.DataFrame({
        "label_true_name": labels,
        "video": videos,
        "trace": traces,
        "y_pred_prob": y_pred_prob,
        "y_true": y_true,
        "predicted_label": y_pred_bin,
        "y_true_classname": [idx_to_name[i] for i in y_true],
        "predicted_classname": [idx_to_name[i] for i in y_pred_bin]
    })
    pred_df.to_csv(preds_path, sep="\t", index=False)
    print(f"\nPredictions saved to: {preds_path}")

    # -------------------
    # Visualization
    # -------------------
    total_traces = len(y_true)
    counts_per_class = "\n".join(
        f"{name}: {int(np.sum(y_true == i))}" for i, name in enumerate(target_names)
    )

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # Confusion Matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred_bin, ax=ax[0, 0], cmap="viridis",
        colorbar=True, display_labels=target_names, normalize="true"
    )
    try:
        disp.ax_.images[-1].set_clim(0, 1)
        cbar = disp.ax_.images[-1].colorbar
        if cbar:
            cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    except Exception:
        pass
    ax[0, 0].set_title("Confusion Matrix", fontsize=14, pad=15)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    ax[0, 1].plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax[0, 1].set_title("ROC Curve", fontsize=14)
    ax[0, 1].legend(loc="lower right")

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    baseline = np.mean(y_true)
    ax[1, 0].plot(recall, precision, label=f"AP = {ap_score:.4f}")
    ax[1, 0].hlines(baseline, xmin=0, xmax=1, color="gray", linestyle="--",
                    label=f"Baseline = {baseline:.2f}")
    ax[1, 0].set_title("Precision–Recall Curve", fontsize=14)
    ax[1, 0].legend(loc="lower left")

    # Metrics text
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

    plt.suptitle("Prediction Evaluation Report", fontsize=20, weight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
    plt.show()
    plt.close(fig)


@app.command()
def predict_unlabeled_model(
        model_path: Path,
        data_tsv: Path,
        threshold: float = 0.5
) -> None:
    """
    Predict on an unlabeled dataset formatted as:
    trace, time_0, time_10, ..., time_600

    - Saves predictions TSV and histogram PNG to IAV_output/
    - Histogram uses viridis colormap with no borders
    - Pred label mapping: 1=PR8, 0=X31
    """

    # -------------------
    # Resolve model path
    # -------------------
    mp = Path(model_path)
    if not mp.exists():
        # Try resolving relative to repo_dir/models
        candidate = repo_dir / "models" / mp.name
        if candidate.exists():
            mp = candidate
        else:
            raise FileNotFoundError(f"Model not found: {model_path} (also checked {candidate})")

    print(f"Loading model from local path: {mp}")
    try:
        # Works for .keras files and SavedModel directories
        model = tf.keras.models.load_model(mp)
    except Exception as e:
        raise RuntimeError(f"Failed to load Keras model from {mp}: {e}")

    # -------------------
    # Load data
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
        raise ValueError("Input TSV must contain a 'trace' column as the first column.")

    # Extract trace identifiers
    traces = data["trace"].astype(str)

    # Extract numeric features (all columns after 'trace')
    feature_cols = data.columns[1:]
    if len(feature_cols) == 0:
        raise ValueError("No timepoint columns found after 'trace'.")

    # Shape: (N, T, 1) — add channel dim for 1D convs/RNNs expecting features_last
    x_data = data.loc[:, feature_cols].values[:, :, np.newaxis].astype("float32")

    # -------------------
    # Run predictions
    # -------------------
    print("Running model predictions...")
    y_pred_prob = model.predict(x_data).astype("float32").flatten()
    y_pred_bin = (y_pred_prob > threshold).astype(int)

    # Map 0/1 to class names (1=PR8, 0=X31)
    idx_to_name = {0: "X31", 1: "PR8"}
    predicted_classname = np.vectorize(idx_to_name.get)(y_pred_bin)

    # -------------------
    # Output paths
    # -------------------
    output_dir = repo_dir / "IAV_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tag = f"{mp.stem}_{dataset_path.stem}_thr{threshold:g}"

    preds_path = output_dir / f"{base_tag}_unlabeled_predictions_model.tsv"
    plot_path  = output_dir / f"{base_tag}_unlabeled_hist_model.png"

    # -------------------
    # Save predictions TSV (unlabeled → no true labels/videos)
    # -------------------
    pred_df = pd.DataFrame({
        "trace": traces,
        "y_pred_prob": y_pred_prob,
        "y_pred_bin": y_pred_bin,                 # 0/1 where 1=PR8, 0=X31
        "predicted_classname": predicted_classname  # "PR8"/"X31"
    })
    pred_df.to_csv(preds_path, sep="\t", index=False)
    print(f"\nPredictions saved to: {preds_path}")

    # Probability histogram (viridis gradient, no borders)
    fig = plt.figure(figsize=(6, 4))
    counts, bins, patches = plt.hist(y_pred_prob, bins=30, edgecolor='none')  # no borders
    norm = mcolors.Normalize(vmin=counts.min() if counts.size else 0,
                             vmax=counts.max() if counts.size else 1)
    cmap = cm.get_cmap("viridis")
    for c, p in zip(counts, patches):
        p.set_facecolor(cmap(norm(c)))
        p.set_linewidth(0)

    # Threshold line
    plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold}")

    plt.title("Distribution of Predicted Probabilities")
    plt.xlabel("Predicted Probability (PR8 class)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[saved] {plot_path}")
    plt.show()
    plt.close(fig)
