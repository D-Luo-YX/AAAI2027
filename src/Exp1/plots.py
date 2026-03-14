from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np



def plot_learning_curves(curves_df, results_dir: Path, active_fed_methods):
    fed_curves = curves_df[curves_df["method"].isin(active_fed_methods)].copy()
    if len(fed_curves) == 0:
        return

    grouped = (
        fed_curves.groupby(["dataset", "model", "method", "round"])["test_acc"]
        .mean()
        .reset_index()
    )

    for dataset_name in grouped["dataset"].unique():
        for model_name in grouped["model"].unique():
            part = grouped[
                (grouped["dataset"] == dataset_name)
                & (grouped["model"] == model_name)
            ]

            plt.figure(figsize=(6, 4))
            for method_name in part["method"].unique():
                sub = part[part["method"] == method_name]
                plt.plot(sub["round"], sub["test_acc"], label=method_name)

            plt.xlabel("Global round")
            plt.ylabel("Test accuracy")
            plt.title(f"{dataset_name} - {model_name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / f"curve_{dataset_name}_{model_name}.png", dpi=200)
            plt.close()



def plot_final_bars(summary_df, results_dir: Path):
    grouped = (
        summary_df.groupby(["dataset", "model", "method"])["test_acc"]
        .agg(["mean", "std"])
        .reset_index()
    )

    for model_name in grouped["model"].unique():
        part = grouped[grouped["model"] == model_name].copy()
        datasets = ["Cora", "CiteSeer", "PubMed"]
        methods = list(part["method"].unique())

        x = np.arange(len(datasets))
        width = 0.18

        plt.figure(figsize=(8, 4))
        for i, method_name in enumerate(methods):
            vals = []
            errs = []
            for dataset_name in datasets:
                row = part[
                    (part["dataset"] == dataset_name)
                    & (part["method"] == method_name)
                ]
                vals.append(row["mean"].iloc[0] if len(row) > 0 else 0.0)
                errs.append(row["std"].iloc[0] if len(row) > 0 else 0.0)

            plt.bar(x + i * width, vals, width=width, yerr=errs, label=method_name)

        plt.xticks(x + width * (len(methods) - 1) / 2, datasets)
        plt.ylabel("Test accuracy")
        plt.title(f"Final accuracy - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / f"bar_{model_name}.png", dpi=200)
        plt.close()



def plot_partition_stats(partition_df, results_dir: Path):
    grouped = (
        partition_df.groupby(["dataset", "client_id"])["num_nodes"]
        .mean()
        .reset_index()
    )

    for dataset_name in grouped["dataset"].unique():
        part = grouped[grouped["dataset"] == dataset_name]
        plt.figure(figsize=(7, 4))
        plt.bar(part["client_id"], part["num_nodes"])
        plt.xlabel("Client ID")
        plt.ylabel("Average number of nodes")
        plt.title(f"Client size distribution - {dataset_name}")
        plt.tight_layout()
        plt.savefig(results_dir / f"client_size_{dataset_name}.png", dpi=200)
        plt.close()
