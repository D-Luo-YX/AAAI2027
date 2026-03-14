import pandas as pd

from config import Exp1Config
from data import load_planetoid, louvain_partition_to_clients
from methods import build_method_registry
from plots import plot_final_bars, plot_learning_curves, plot_partition_stats
from reporting import save_latex_table
from runners import run_centralized, run_federated, run_local_only
from utils import set_seed



def main():
    cfg = Exp1Config()
    method_registry = build_method_registry(cfg)

    summary_rows = []
    curve_dfs = []
    partition_dfs = []

    for dataset_name in cfg.datasets:
        for seed in cfg.seeds:
            set_seed(seed)

            dataset, data = load_planetoid(dataset_name, seed, cfg)
            clients, part_df = louvain_partition_to_clients(
                data=data,
                num_clients=cfg.num_clients,
                resolution=cfg.resolution,
                seed=seed,
            )
            part_df["dataset"] = dataset_name
            part_df["seed"] = seed
            partition_dfs.append(part_df)

            in_dim = dataset.num_features
            out_dim = dataset.num_classes

            for model_name in cfg.models:
                centralized_summary, centralized_curve = run_centralized(
                    data=data,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    seed=seed,
                    cfg=cfg,
                )
                summary_rows.append(centralized_summary)
                curve_dfs.append(centralized_curve)

                local_summary, local_curve = run_local_only(
                    clients=clients,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    seed=seed,
                    cfg=cfg,
                )
                summary_rows.append(local_summary)
                curve_dfs.append(local_curve)

                for method_name in cfg.federated_methods_to_run:
                    method = method_registry[method_name]
                    fed_summary, fed_curve = run_federated(
                        clients=clients,
                        in_dim=in_dim,
                        out_dim=out_dim,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        seed=seed,
                        method=method,
                        cfg=cfg,
                    )
                    summary_rows.append(fed_summary)
                    curve_dfs.append(fed_curve)

                print(f"Done | dataset={dataset_name} | seed={seed} | model={model_name}")

    summary_df = pd.DataFrame(summary_rows)
    curves_df = pd.concat(curve_dfs, ignore_index=True)
    partition_df = pd.concat(partition_dfs, ignore_index=True)

    summary_df.to_csv(cfg.results_dir / "summary.csv", index=False)
    curves_df.to_csv(cfg.results_dir / "curves.csv", index=False)
    partition_df.to_csv(cfg.results_dir / "partition_stats.csv", index=False)

    plot_learning_curves(curves_df, cfg.results_dir, cfg.federated_methods_to_run)
    plot_final_bars(summary_df, cfg.results_dir)
    plot_partition_stats(partition_df, cfg.results_dir)
    save_latex_table(summary_df, cfg.results_dir / "table_main.tex")

    print(summary_df.groupby(["dataset", "model", "method"])["test_acc"].mean())
    print("\nReserved method slots:")
    for name in cfg.reserved_method_slots:
        print(f"- {name}")


if __name__ == "__main__":
    main()
