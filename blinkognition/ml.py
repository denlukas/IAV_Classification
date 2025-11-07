from pathlib import Path

import mlflow
import keras

import pandas as pd
import numpy as np
from matplotlib.colors import Normalize

from blinkognition.utils import repo_dir, set_seeds
from blinkognition.mcc import MatthewsCorrelationCoefficient
from blinkognition.data import DataSplit, load_dataset_split
from blinkognition.model import make_model, monte_carlo_predict_samples

import typer

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay, classification_report, accuracy_score,
    confusion_matrix, matthews_corrcoef
)

import matplotlib.pyplot as plt
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
        patience: int | None = typer.Option(
            200, help='Early stopping patience. Set 0 to disable'),
        verbose: int = 2,
) -> None:
    seed = set_seeds(seed)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(dict(seed=seed, dataset=dataset,
                               run_name=run_name,
                               patience=patience, epochs=epochs,
                               batch_size=batch_size))

        datasplit = load_dataset_split(dataset, seed=seed)

        datasplit.x_train = datasplit.x_train[..., np.newaxis]
        datasplit.x_test  = datasplit.x_test[..., np.newaxis]

        model = make_model(datasplit.x_train.shape[1:])

        try:
            history = fit(model=model,
                          datasplit=datasplit,
                          run_name=run_name,
                          epochs=epochs,
                          batch_size=batch_size,
                          patience=patience,
                          verbose=verbose)
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
             include_uncertainty: bool = True,
             use_post_analysis: bool = True,
             n_mc_samples: int = 1,
             ct_threshold: float = 0,
             spaced_threshold: bool = True
             ) -> None:

    # Map numeric classes to E1/E2 labels
    class_labels = {0: "E1", 1: "E2"}

    model = mlflow.keras.load_model(model_uri)
    datasplit = load_dataset_split(dataset, seed=seed)
    datasplit.x_test = datasplit.x_test[..., np.newaxis]

    # Monte Carlo predictions if needed
    if include_uncertainty:
        print("Running Monte Carlo prediction...")
        y_pred_prob, probs_try = monte_carlo_predict_samples(
            model, datasplit.x_test, n_mc_samples=n_mc_samples
        )
        if probs_try.shape[2] == 1:
            probs_try = np.concatenate([1 - probs_try, probs_try], axis=2)
    else:
        y_pred_prob = model.predict(datasplit.x_test).astype("float32")
        probs_try = None

    y_pred_prob = y_pred_prob.flatten()
    y_pred_bin = (y_pred_prob > 0.5).astype(int)

    # --- map ground truth and predictions to E1/E2 ---
    y_true_str = pd.Series(datasplit.y_test).map(class_labels).values
    y_pred_str = pd.Series(y_pred_bin).map(class_labels).values

    # Basic metrics
    acc = accuracy_score(y_true_str, y_pred_str)
    mcc = matthews_corrcoef(y_true_str, y_pred_str)
    class_report = classification_report(y_true_str, y_pred_str, digits=3, labels=["E1", "E2"])
    auc_score = roc_auc_score(datasplit.y_test, y_pred_prob)  # keep numeric for ROC
    fpr, tpr, _ = roc_curve(datasplit.y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(datasplit.y_test, y_pred_prob)
    ap_score = average_precision_score(datasplit.y_test, y_pred_prob)

    classes = ["E1", "E2"]
    total_traces = len(y_true_str)

    # Plot evaluation metrics
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true_str,
        y_pred_str,
        display_labels=classes,
        ax=ax[0, 0],
        cmap="viridis",
        colorbar=True,
        normalize='true',
        values_format=".2f"
    )
    im = ax[0, 0].images[-1]
    im.set_norm(Normalize(vmin=0, vmax=1))

    ax[0, 0].set_title("Confusion Matrix", fontsize=14, pad=15)
    # --- ROC curve ---
    ax[0, 1].plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
    ax[0, 1].set_xlabel("False Positive Rate")
    ax[0, 1].set_ylabel("True Positive Rate")
    ax[0, 1].set_xlim(0, 1)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].margins(x=0, y=0)
    ax[0, 1].legend(loc="lower right")

    # --- Precision-Recall curve ---
    baseline_precision = np.mean(datasplit.y_test)  # fraction of positive class
    ax[1, 0].plot(recall, precision, label=f"AP = {ap_score:.4f}")
    ax[1, 0].hlines(
        y=baseline_precision, xmin=0, xmax=1,
        colors="gray", linestyles="--",
        label=f"Baseline = {baseline_precision:.2f}"
    )
    ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, pad=15)
    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")
    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].margins(x=0, y=0)
    ax[1, 0].legend(loc="lower left")

    counts = {c: np.sum(y_true_str == c) for c in classes}
    metrics_text = (
        f"Total traces: {total_traces}\n"
        + "\n".join([f"{c}: {counts[c]}" for c in classes])
        + f"\n\nAccuracy: {acc:.4f}\nMCC: {mcc:.4f}\n\nClassification Report:\n{class_report}"
    )

    ax[1, 1].axis("off")
    ax[1, 1].text(0, 1, metrics_text, ha="left", va="top", family="monospace", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Model Evaluation Report", fontsize=20, weight="bold")
    plt.show()

    print("\n================ Evaluation Report ================\n")
    print(f"Accuracy: {acc:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print("\nClassification Report:")
    print(class_report)
    print(f"\nROC-AUC: {auc_score:.4f}")
    print(f"Average Precision (PR-AUC): {ap_score:.4f}")

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

        post_df = pd.DataFrame({
            "y_true": pd.Series(datasplit.y_test).map(class_labels),
            "predicted_label": pd.Series(predicted_label).map(class_labels),
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

        plt.xlabel('Minimal Wasserstein distance')
        plt.ylabel('Number of samples')
        plt.title('Distribution of minimal Wasserstein distance')
        plt.show()

        output_csv = repo_dir / "post_analysis_report.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        post_df.to_csv(output_csv, index=False)
        print(f"Post-analysis report saved to {output_csv}")

        mask_ct = post_df['min_wasserstein'] > ct_threshold
        print(f"CT threshold {ct_threshold}: {mask_ct.sum()} samples remaining out of {len(post_df)}")

        y_true_filtered = post_df["y_true"].values[mask_ct]
        y_pred_filtered = post_df["predicted_label"].values[mask_ct]
        y_prob_filtered = probs_mean[mask_ct, 1] if mask_ct.sum() > 0 else np.array([])

        if len(y_true_filtered) > 1 and len(np.unique(y_true_filtered)) > 1:
            acc_f = accuracy_score(y_true_filtered, y_pred_filtered)
            mcc_f = matthews_corrcoef(y_true_filtered, y_pred_filtered)
            auc_f = roc_auc_score((y_true_filtered == "E2").astype(int), y_prob_filtered)
            class_report_f = classification_report(y_true_filtered, y_pred_filtered, digits=3, labels=["E1", "E2"])

            fpr_f, tpr_f, _ = roc_curve((y_true_filtered == "E2").astype(int), y_prob_filtered)
            precision_f, recall_f, _ = precision_recall_curve((y_true_filtered == "E2").astype(int), y_prob_filtered)
            ap_f = average_precision_score((y_true_filtered == "E2").astype(int), y_prob_filtered)

            fig_f, ax_f = plt.subplots(2, 2, figsize=(10, 8))
            ConfusionMatrixDisplay.from_predictions(
                y_true_filtered,
                y_pred_filtered,
                display_labels=classes,
                ax=ax_f[0, 0],
                cmap="viridis",
                colorbar=True,
                normalize='true',
                values_format=".2f"
            )
            im_f = ax_f[0, 0].images[-1]
            im_f.set_norm(Normalize(vmin=0, vmax=1))

            ax_f[0, 0].set_title("Confusion Matrix (Filtered CT)", fontsize=14, pad=15)
            # --- Filtered ROC curve ---
            ax_f[0, 1].plot(fpr_f, tpr_f, label=f"AUC = {auc_f:.4f}")
            ax_f[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax_f[0, 1].set_title("ROC Curve (Filtered CT)", fontsize=14, pad=15)
            ax_f[0, 1].set_xlabel("False Positive Rate")
            ax_f[0, 1].set_ylabel("True Positive Rate")
            ax_f[0, 1].set_xlim(0, 1)
            ax_f[0, 1].set_ylim(0, 1)
            ax_f[0, 1].margins(x=0, y=0)  # remove whitespace
            ax_f[0, 1].legend(loc="lower right")

            # --- Filtered Precision-Recall curve ---
            baseline_precision_f = np.mean(y_true_filtered == "E2")

            ax_f[1, 0].plot(recall_f, precision_f, label=f"AP = {ap_f:.4f}")
            ax_f[1, 0].hlines(
                y=baseline_precision_f, xmin=0, xmax=1,
                colors="gray", linestyles="--",
                label=f"Baseline = {baseline_precision_f:.2f}"
            )

            ax_f[1, 0].set_title("Precision-Recall Curve (Filtered CT)", fontsize=14, pad=15)
            ax_f[1, 0].set_xlabel("Recall")
            ax_f[1, 0].set_ylabel("Precision")
            ax_f[1, 0].set_xlim(0, 1)
            ax_f[1, 0].set_ylim(0, 1)
            ax_f[1, 0].margins(x=0, y=0)  # remove whitespace
            ax_f[1, 0].legend(loc="lower left")

            counts_f = {c: np.sum(y_true_filtered == c) for c in classes}
            metrics_text_f = (
                f"Total traces: {len(y_true_filtered)} / {len(post_df)} remaining\n"
                + "\n".join([f"{c}: {counts_f[c]}" for c in classes])
                + f"\nCT-threshold used: {ct_threshold}\n\n"
                + f"Accuracy: {acc_f:.4f}\n"
                + f"MCC: {mcc_f:.4f}\n\n"
                + f"Classification Report:\n{class_report_f}"
            )

            ax_f[1, 1].axis("off")
            ax_f[1, 1].text(0, 1, metrics_text_f, ha="left", va="top", family="monospace", fontsize=11)
            fig_f.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.suptitle("Post-analysis Filtered Evaluation (CT)", fontsize=20, weight="bold")
            plt.show()

        print("\n====== Post-analysis Filtered Evaluation ======")
        print(f"Traces remaining: {len(y_true_filtered)} / {len(datasplit.y_test)}")
        print(f"Accuracy: {acc_f:.4f}")
        print(f"MCC: {mcc_f:.4f}")
        print("\nClassification Report:")
        print(class_report_f)
        print(f"ROC-AUC: {auc_f:.4f}")
        print(f"Average Precision (PR-AUC): {ap_f:.4f}")

        # ============================================================
        # ECDF and Barplot
        # ============================================================
        if use_post_analysis and include_uncertainty:
            # --- ECDF ---
            if len(post_df) > 0:
                fig, ax = plt.subplots(figsize=(6, 5))
                cmap = plt.get_cmap("tab10")
                colors = {label: cmap(i % cmap.N) for i, label in enumerate(classes)}

                for label in classes:
                    values = post_df.loc[post_df["y_true"] == label, "min_wasserstein"].values
                    if len(values) == 0:
                        continue
                    x = np.sort(values)
                    y = np.arange(1, len(x) + 1) / len(x)
                    ax.step(
                        x, y,
                        where="post",
                        color=colors[label],
                        alpha=0.8,
                        label=f"{label} (n={len(values)})"
                    )

                # --- Remove whitespace ---
                x_min, x_max = post_df["min_wasserstein"].min(), post_df["min_wasserstein"].max()
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(0, 1)
                ax.margins(x=0, y=0)

                # --- Labels and title ---
                ax.set_xlabel("Minimal Wasserstein distance")
                ax.set_ylabel("ECDF")
                ax.set_title("ECDF of minimal Wasserstein distance")
                ax.legend(title="Class")

                plt.tight_layout()
                plt.show()
            else:
                print(f"No samples left after applying CT>{ct_threshold}, skipping ECDF plot.")

            # Barplot
            ct_thresholds = [0, 0.2, 0.4, 0.6, 0.8]
            fractions_per_threshold = []

            for ct in ct_thresholds:
                mask_ct = post_df['min_wasserstein'] > ct
                y_true_filtered = post_df["y_true"].values[mask_ct] if mask_ct.sum() > 0 else np.array([])

                if len(y_true_filtered) > 0:
                    filtered_fractions = [
                        np.sum(y_true_filtered == c) / len(y_true_filtered)
                        for c in classes
                    ]
                else:
                    filtered_fractions = [0 for _ in classes]

                fractions_per_threshold.append(filtered_fractions)

            fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
            bar_width = 0.7
            x_positions = np.arange(len(ct_thresholds))
            bottoms = np.zeros(len(ct_thresholds))

            for i, c in enumerate(classes):
                class_heights = [fractions_per_threshold[j][i] for j in range(len(ct_thresholds))]
                ax_bar.bar(x_positions, class_heights, width=bar_width, bottom=bottoms, label=c)
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
            # Post-analysis evaluation with spaced CT thresholds
            # -------------------
            if spaced_threshold:
                min_w = post_df["min_wasserstein"].min()
                max_w = post_df["min_wasserstein"].max()
                ct_thresholds = np.linspace(min_w, max_w, num=6)[1:-1]  # 4 evenly spaced thresholds
                ct_thresholds = np.round(ct_thresholds, 3)
            else:
                ct_thresholds = [ct_threshold]

            for ct in ct_thresholds:
                mask_ct = post_df['min_wasserstein'] > ct
                print(f"\nCT threshold {ct}: {mask_ct.sum()} samples remaining out of {len(post_df)}")

                y_true_filtered = post_df["y_true"].values[mask_ct]
                y_pred_filtered = post_df["predicted_label"].values[mask_ct]
                y_prob_filtered = probs_mean[mask_ct, 1] if mask_ct.sum() > 0 else np.array([])

                if len(y_true_filtered) > 1 and len(np.unique(y_true_filtered)) > 1:
                    acc_f = accuracy_score(y_true_filtered, y_pred_filtered)
                    mcc_f = matthews_corrcoef(y_true_filtered, y_pred_filtered)
                    auc_f = roc_auc_score((y_true_filtered == "E2").astype(int), y_prob_filtered)
                    class_report_f = classification_report(y_true_filtered, y_pred_filtered,
                                                           digits=3, labels=["E1", "E2"])

                    fpr_f, tpr_f, _ = roc_curve((y_true_filtered == "E2").astype(int), y_prob_filtered)
                    precision_f, recall_f, _ = precision_recall_curve((y_true_filtered == "E2").astype(int),
                                                                      y_prob_filtered)
                    ap_f = average_precision_score((y_true_filtered == "E2").astype(int), y_prob_filtered)

                    fig_f, ax_f = plt.subplots(2, 2, figsize=(10, 8))
                    ConfusionMatrixDisplay.from_predictions(
                        y_true_filtered,
                        y_pred_filtered,
                        display_labels=classes,
                        ax=ax_f[0, 0],
                        cmap="viridis",
                        colorbar=True,
                        normalize='true',
                        values_format=".2f"
                    )
                    im_f = ax_f[0, 0].images[-1]
                    im_f.set_norm(Normalize(vmin=0, vmax=1))

                    ax_f[0, 0].set_title(f"Confusion Matrix (CT>{ct})", fontsize=14, pad=15)
                    # --- ROC curve ---
                    ax_f[0, 1].plot(fpr_f, tpr_f, label=f"AUC = {auc_f:.4f}")
                    ax_f[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
                    ax_f[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
                    ax_f[0, 1].set_xlabel("False Positive Rate")
                    ax_f[0, 1].set_ylabel("True Positive Rate")
                    ax_f[0, 1].set_xlim(0, 1)
                    ax_f[0, 1].set_ylim(0, 1)
                    ax_f[0, 1].margins(x=0, y=0)  # remove whitespace
                    ax_f[0, 1].legend(loc="lower right")

                    # --- Precision-Recall curve ---
                    baseline_precision_f = np.mean(y_true_filtered == "E2")

                    ax_f[1, 0].plot(recall_f, precision_f, label=f"AP = {ap_f:.4f}")
                    ax_f[1, 0].hlines(
                        y=baseline_precision_f, xmin=0, xmax=1,
                        colors="gray", linestyles="--",
                        label=f"Baseline = {baseline_precision_f:.2f}"
                    )

                    ax_f[1, 0].set_title("Precision-Recall Curve (Filtered CT)", fontsize=14, pad=15)
                    ax_f[1, 0].set_xlabel("Recall")
                    ax_f[1, 0].set_ylabel("Precision")
                    ax_f[1, 0].set_xlim(0, 1)
                    ax_f[1, 0].set_ylim(0, 1)
                    ax_f[1, 0].margins(x=0, y=0)  # remove whitespace
                    ax_f[1, 0].legend(loc="lower left")

                    counts_f = {c: np.sum(y_true_filtered == c) for c in classes}
                    metrics_text_f = (
                            f"Total traces: {len(y_true_filtered)} / {len(post_df)} remaining\n"
                            + "\n".join([f"{c}: {counts_f[c]}" for c in classes])
                            + f"\nCT threshold used: {ct}\n\n"
                            + f"Accuracy: {acc_f:.4f}\n"
                            + f"MCC: {mcc_f:.4f}\n\n"
                            + f"Classification Report:\n{class_report_f}"
                    )

                    ax_f[1, 1].axis("off")
                    ax_f[1, 1].text(0, 1, metrics_text_f, ha="left", va="top",
                                    family="monospace", fontsize=11)
                    fig_f.tight_layout(rect=[0, 0.05, 1, 0.95])
                    plt.suptitle(f"Post-analysis Filtered Evaluation (CT>{ct})", fontsize=20, weight="bold")
                    plt.show()

                    print("\n====== Post-analysis Filtered Evaluation ======")
                    print(f"CT threshold: {ct}")
                    print(f"Traces remaining: {len(y_true_filtered)} / {len(datasplit.y_test)}")
                    print(f"Accuracy: {acc_f:.4f}")
                    print(f"MCC: {mcc_f:.4f}")
                    print("\nClassification Report:")
                    print(class_report_f)
                    print(f"ROC-AUC: {auc_f:.4f}")
                    print(f"Average Precision (PR-AUC): {ap_f:.4f}")


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
