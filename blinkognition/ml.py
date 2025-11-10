import mlflow
import keras
import pandas as pd
import numpy as np
import typer
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from pathlib import Path
from blinkognition.utils import repo_dir, set_seeds
from blinkognition.data import DataSplit, load_dataset_split
from blinkognition.model import make_model, monte_carlo_predict_samples
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay, classification_report, accuracy_score,
    confusion_matrix, matthews_corrcoef
)
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

    # saving helper for output files
    output_dir = repo_dir / "blinkognition_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tag = f"{Path(model_uri).stem}_{Path(dataset).stem}_seed{seed}"

    def savefig(fig, label: str):
        path = output_dir / f"{base_tag}_{label}.png"
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[saved] {path}")

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

    # map ground truth and predictions to E1/E2
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

    # Evaluation report
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true_str, y_pred_str, display_labels=classes, ax=ax[0, 0],
        cmap="viridis", colorbar=True, normalize='true', values_format=".2f"
    )
    im = ax[0, 0].images[-1]
    im.set_norm(Normalize(vmin=0, vmax=1))
    ax[0, 0].set_title("Confusion Matrix", fontsize=14, pad=15)

    # ROC curve
    ax[0, 1].plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
    ax[0, 1].set_xlabel("False Positive Rate"); ax[0, 1].set_ylabel("True Positive Rate")
    ax[0, 1].set_xlim(0, 1); ax[0, 1].set_ylim(0, 1); ax[0, 1].margins(x=0, y=0)
    ax[0, 1].legend(loc="lower right")

    # PR curve
    baseline_precision = np.mean(datasplit.y_test)
    ax[1, 0].plot(recall, precision, label=f"AP = {ap_score:.4f}")
    ax[1, 0].hlines(y=baseline_precision, xmin=0, xmax=1, colors="gray", linestyles="--",
                    label=f"Baseline = {baseline_precision:.2f}")
    ax[1, 0].set_title("Precision-Recall Curve", fontsize=14, pad=15)
    ax[1, 0].set_xlabel("Recall"); ax[1, 0].set_ylabel("Precision")
    ax[1, 0].set_xlim(0, 1); ax[1, 0].set_ylim(0, 1); ax[1, 0].margins(x=0, y=0)
    ax[1, 0].legend(loc="lower left")

    # metrics box
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
    savefig(fig, "eval_main")
    plt.show()
    plt.close(fig)

    # Print results
    print("\n================ Evaluation Report ================\n")
    print(f"Accuracy: {acc:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print("\nClassification Report:")
    print(class_report)
    print(f"\nROC-AUC: {auc_score:.4f}")
    print(f"Average Precision (PR-AUC): {ap_score:.4f}")

    # Post-analysis with CT (Wasserstein distance)
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

        # save TSV with full post-analysis table
        post_csv = output_dir / f"{base_tag}_post_analysis.tsv"
        post_df.to_csv(post_csv, sep="\t", index=False)
        print(f"[saved] {post_csv}")

        # Histogram
        fig_hist = plt.figure(figsize=(6, 4))
        counts, bins = np.histogram(post_df['min_wasserstein'], bins=50)
        norm = plt.Normalize(vmin=counts.min() if counts.size else 0,
                             vmax=counts.max() if counts.size else 1)
        cmap = plt.cm.viridis
        cols = cmap(norm(counts if counts.size else np.array([0])))
        for i in range(len(bins) - 1):
            h = counts[i] if i < len(counts) else 0
            plt.bar(bins[i], h, width=bins[i + 1] - bins[i],
                    color=cols[i if i < len(cols) else 0], align='edge')
        plt.xlabel('Minimal Wasserstein distance')
        plt.ylabel('Number of samples')
        plt.title('Distribution of minimal Wasserstein distance')
        savefig(fig_hist, "hist_min_wasserstein")
        plt.show()
        plt.close(fig_hist)

        # Single-cut filtered eval (CT > ct_threshold)
        mask_ct = post_df['min_wasserstein'] > ct_threshold
        filtered_df = post_df[mask_ct]
        if len(filtered_df) > 0:
            y_true_filt = filtered_df["y_true"].values
            y_pred_filt = filtered_df["predicted_label"].values
            y_prob_filt = probs_mean[mask_ct, 1]

            acc_f = accuracy_score(y_true_filt, y_pred_filt)
            mcc_f = matthews_corrcoef(y_true_filt, y_pred_filt)
            auc_f = roc_auc_score((y_true_filt == "E2").astype(int), y_prob_filt)
            precision_f, recall_f, _ = precision_recall_curve((y_true_filt == "E2").astype(int), y_prob_filt)
            ap_f = average_precision_score((y_true_filt == "E2").astype(int), y_prob_filt)
            fpr_f, tpr_f, _ = roc_curve((y_true_filt == "E2").astype(int), y_prob_filt)
            class_report_f = classification_report(y_true_filt, y_pred_filt, digits=3, labels=["E1", "E2"])

            fig_filt, ax = plt.subplots(2, 2, figsize=(10, 8))
            ConfusionMatrixDisplay.from_predictions(
                y_true_filt, y_pred_filt, display_labels=classes, ax=ax[0, 0],
                cmap="viridis", colorbar=True, normalize='true', values_format=".2f"
            )
            im_f = ax[0, 0].images[-1]; im_f.set_norm(Normalize(vmin=0, vmax=1))
            ax[0, 0].set_title(f"Confusion Matrix (CT>{ct_threshold})", fontsize=14, pad=15)

            # ROC curve
            ax[0, 1].plot(fpr_f, tpr_f, label=f"AUC = {auc_f:.4f}")
            ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax[0, 1].set_title("ROC Curve (Filtered CT)", fontsize=14, pad=15)
            ax[0, 1].set_xlabel("False Positive Rate"); ax[0, 1].set_ylabel("True Positive Rate")
            ax[0, 1].set_xlim(0, 1); ax[0, 1].set_ylim(0, 1); ax[0, 1].margins(x=0, y=0)
            ax[0, 1].legend(loc="lower right")

            # PR curve
            baseline_precision_f = np.mean(y_true_filt == "E2")
            ax[1, 0].plot(recall_f, precision_f, label=f"AP = {ap_f:.4f}")
            ax[1, 0].hlines(y=baseline_precision_f, xmin=0, xmax=1, colors="gray", linestyles="--",
                            label=f"Baseline = {baseline_precision_f:.2f}")
            ax[1, 0].set_title("Precision-Recall Curve (Filtered CT)", fontsize=14, pad=15)
            ax[1, 0].set_xlabel("Recall"); ax[1, 0].set_ylabel("Precision")
            ax[1, 0].set_xlim(0, 1); ax[1, 0].set_ylim(0, 1); ax[1, 0].margins(x=0, y=0)
            ax[1, 0].legend(loc="lower left")

            # metrics box
            counts_f = {c: np.sum(y_true_filt == c) for c in classes}
            metrics_text_f = (
                f"Traces remaining: {len(y_true_filt)} / {len(post_df)}\n"
                + "\n".join([f"{c}: {counts_f[c]}" for c in classes])
                + f"\nCT-threshold used: {ct_threshold}\n\n"
                + f"Accuracy: {acc_f:.4f}\nMCC: {mcc_f:.4f}\n\n"
                + f"Classification Report:\n{class_report_f}"
            )
            ax[1, 1].axis("off")
            ax[1, 1].text(0, 1, metrics_text_f, ha="left", va="top", family="monospace", fontsize=11)

            plt.suptitle("Filtered Evaluation Report", fontsize=20, weight="bold")
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            savefig(fig_filt, f"eval_filtered_CTgt_{ct_threshold:g}")
            plt.show()
            plt.close(fig_filt)

        # ECDF
        fig_ecdf, ax = plt.subplots(figsize=(6, 5))
        x_min, x_max = post_df["min_wasserstein"].min(), post_df["min_wasserstein"].max()
        cmap = plt.get_cmap("tab10")
        colors = {"E1": cmap(0), "E2": cmap(1)}
        for label in classes:
            vals = post_df.loc[post_df["y_true"] == label, "min_wasserstein"].values
            if vals.size == 0: continue
            x = np.sort(vals); y = np.arange(1, len(x)+1)/len(x)
            ax.step(x, y, where="post", color=colors[label], alpha=0.8, label=f"{label} (n={len(vals)})")
        ax.set_xlim(x_min, x_max); ax.set_ylim(0, 1); ax.margins(x=0, y=0)
        ax.set_xlabel("Minimal Wasserstein distance"); ax.set_ylabel("ECDF")
        ax.set_title("ECDF of minimal Wasserstein distance"); ax.legend(title="Class")
        savefig(fig_ecdf, "ecdf_min_w")
        plt.tight_layout()
        plt.show()
        plt.close(fig_ecdf)

        # Class fractions vs CT thresholds
        max_w = post_df["min_wasserstein"].max()
        ct_thresholds_all = (np.linspace(0, max_w, num=6)[:-1] if spaced_threshold else [ct_threshold])
        ct_thresholds_all = np.round(ct_thresholds_all, 3)

        fractions_per_threshold = []
        for ct in ct_thresholds_all:
            mask = post_df['min_wasserstein'] >= ct
            y_tf = post_df["y_true"].values[mask] if mask.sum() > 0 else np.array([])
            if y_tf.size > 0:
                fracs = [(y_tf == c).mean() for c in classes]
            else:
                fracs = [0.0, 0.0]
            fractions_per_threshold.append(fracs)

        fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
        x_pos = np.arange(len(ct_thresholds_all)); bottoms = np.zeros(len(ct_thresholds_all))
        for i, c in enumerate(classes):
            heights = [fractions_per_threshold[j][i] for j in range(len(ct_thresholds_all))]
            ax_bar.bar(x_pos, heights, width=0.7, bottom=bottoms, label=c)
            bottoms += heights
        ax_bar.set_xticks(x_pos); ax_bar.set_xticklabels([str(ct) for ct in ct_thresholds_all])
        ax_bar.set_ylabel("Fraction of traces per class"); ax_bar.set_xlabel("CT threshold")
        ax_bar.set_title("Class fractions across CT thresholds"); ax_bar.legend()
        savefig(fig_bar, "ct_fractions")
        plt.tight_layout()
        plt.show()
        plt.close(fig_bar)

        # Filtered reports for three spaced thresholds
        if spaced_threshold:
            internal = np.linspace(post_df["min_wasserstein"].min(),
                                   post_df["min_wasserstein"].max(), num=6)[1:-1]
            ct_loop = np.round(internal, 3)[:3]  # take first three to cap total at 8 plots
        else:
            ct_loop = [ct_threshold]

        for ct_th in ct_loop:
            mask_ct = post_df["min_wasserstein"] > ct_th
            filtered_df = post_df[mask_ct]
            if len(filtered_df) == 0:
                continue

            y_true_filt = filtered_df["y_true"].values
            y_pred_filt = filtered_df["predicted_label"].values
            y_prob_filt = probs_mean[mask_ct, 1]

            acc_f = accuracy_score(y_true_filt, y_pred_filt)
            mcc_f = matthews_corrcoef(y_true_filt, y_pred_filt)
            auc_f = roc_auc_score((y_true_filt == "E2").astype(int), y_prob_filt)
            precision_f, recall_f, _ = precision_recall_curve((y_true_filt == "E2").astype(int), y_prob_filt)
            ap_f = average_precision_score((y_true_filt == "E2").astype(int), y_prob_filt)
            fpr_f, tpr_f, _ = roc_curve((y_true_filt == "E2").astype(int), y_prob_filt)
            class_report_f = classification_report(y_true_filt, y_pred_filt, digits=3, labels=["E1", "E2"])

            fig_loop, ax = plt.subplots(2, 2, figsize=(10, 8))
            ConfusionMatrixDisplay.from_predictions(
                y_true_filt, y_pred_filt, display_labels=classes, ax=ax[0, 0],
                cmap="viridis", colorbar=True, normalize='true', values_format=".2f"
            )
            im_f = ax[0, 0].images[-1]; im_f.set_norm(Normalize(vmin=0, vmax=1))
            ax[0, 0].set_title(f"Confusion Matrix (CT>{ct_th})", fontsize=14, pad=15)

            # ROC curve
            ax[0, 1].plot(fpr_f, tpr_f, label=f"AUC = {auc_f:.4f}")
            ax[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax[0, 1].set_title("ROC Curve", fontsize=14, pad=15)
            ax[0, 1].set_xlabel("False Positive Rate"); ax[0, 1].set_ylabel("True Positive Rate")
            ax[0, 1].set_xlim(0, 1); ax[0, 1].set_ylim(0, 1); ax[0, 1].legend(loc="lower right")

            # PR curve
            baseline_precision_f = np.mean(y_true_filt == "E2")
            ax[1, 0].plot(recall_f, precision_f, label=f"AP = {ap_f:.4f}")
            ax[1, 0].hlines(y=baseline_precision_f, xmin=0, xmax=1, colors="gray", linestyles="--",
                            label=f"Baseline = {baseline_precision_f:.2f}")
            ax[1, 0].set_title("Precision-Recall Curve (Filtered CT)", fontsize=14, pad=15)
            ax[1, 0].set_xlabel("Recall"); ax[1, 0].set_ylabel("Precision")
            ax[1, 0].set_xlim(0, 1); ax[1, 0].set_ylim(0, 1); ax[1, 0].legend(loc="lower left")

            # metrics box
            counts_f = {c: np.sum(y_true_filt == c) for c in classes}
            metrics_text_f = (
                f"Traces remaining: {len(y_true_filt)} / {len(post_df)}\n"
                + "\n".join([f"{c}: {counts_f[c]}" for c in classes])
                + f"\nCT-threshold used: {ct_th}\n\n"
                + f"Accuracy: {acc_f:.4f}\nMCC: {mcc_f:.4f}\n\n"
                + f"Classification Report:\n{class_report_f}"
            )
            ax[1, 1].axis("off")
            ax[1, 1].text(0, 1, metrics_text_f, ha="left", va="top", family="monospace", fontsize=11)

            plt.suptitle(f"Filtered Evaluation Report (CT>{ct_th})", fontsize=20, weight="bold")
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            savefig(fig_loop, f"eval_filtered_CTgt_{ct_th:g}")
            plt.show()
            plt.close(fig_loop)

            # Print results
            print("\n====== Post-analysis Filtered Evaluation ======")
            print(f"CT threshold: {ct_th}")
            print(f"Traces remaining: {len(y_true_filt)} / {len(datasplit.y_test)}")
            print(f"Accuracy: {acc_f:.4f}")
            print(f"MCC: {mcc_f:.4f}")
            print("\nClassification Report:")
            print(class_report_f)
            print(f"ROC-AUC: {auc_f:.4f}")
            print(f"Average Precision (PR-AUC): {ap_f:.4f}")
