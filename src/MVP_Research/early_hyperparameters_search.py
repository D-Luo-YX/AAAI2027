import copy
import itertools
import os
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import subgraph, to_networkx


DATASETS = ["Cora", "CiteSeer", "PubMed"]
MODELS = ["gcn", "sage"]
SEEDS = [0, 1, 2]

NUM_CLIENTS = 10
HIDDEN_DIM = 64
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4

ROUNDS = 100
CLIENT_EPOCHS = 3

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
RESOLUTION = 1.0

TAU_LIST = [0.4, 0.5, 0.6, 0.7]
BETA_LIST = [0.1, 0.2, 0.25, 0.3]
ALPHA_LIST = [0.5, 1.0, 1.5]
MOMENTUM_LIST = [0.3, 0.5, 0.7]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "results/fedsira_nc_hparam_test_only"
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_stratified_masks(y: torch.Tensor, train_ratio=0.6, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    num_nodes = y.size(0)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    classes = torch.unique(y).cpu().tolist()

    for c in classes:
        idx = torch.where(y == c)[0].cpu().numpy()
        rng.shuffle(idx)

        n = len(idx)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def load_planetoid(name: str, seed: int):
    dataset = Planetoid(root="data/Planetoid", name=name)
    data = dataset[0]

    train_mask, val_mask, test_mask = make_stratified_masks(
        data.y,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=seed,
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return dataset, data


def louvain_partition_to_clients(data: Data, num_clients=10, resolution=1.0, seed=0):
    graph_nx = to_networkx(data, to_undirected=True, remove_self_loops=True)
    communities = nx.community.louvain_communities(
        graph_nx,
        resolution=resolution,
        seed=seed,
    )
    communities = [sorted(list(c)) for c in communities]
    communities = sorted(communities, key=len, reverse=True)

    while len(communities) < num_clients:
        largest = communities.pop(0)
        if len(largest) < 2:
            communities.append(largest)
            break
        mid = len(largest) // 2
        communities.append(largest[:mid])
        communities.append(largest[mid:])
        communities = sorted(communities, key=len, reverse=True)

    buckets = [[] for _ in range(num_clients)]
    bucket_sizes = [0 for _ in range(num_clients)]

    for comm in communities:
        idx = int(np.argmin(bucket_sizes))
        buckets[idx].extend(comm)
        bucket_sizes[idx] += len(comm)

    clients = []
    stats = []

    for client_id, nodes in enumerate(buckets):
        node_idx = torch.tensor(sorted(nodes), dtype=torch.long)

        edge_index, _ = subgraph(
            node_idx,
            data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )

        client_data = Data(
            x=data.x[node_idx],
            y=data.y[node_idx],
            edge_index=edge_index,
            train_mask=data.train_mask[node_idx],
            val_mask=data.val_mask[node_idx],
            test_mask=data.test_mask[node_idx],
        )
        client_data.client_id = client_id
        client_data.global_node_id = node_idx

        clients.append(client_data)

        stats.append(
            {
                "client_id": client_id,
                "num_nodes": int(client_data.num_nodes),
                "num_edges": int(client_data.num_edges),
                "num_train": int(client_data.train_mask.sum()),
                "num_val": int(client_data.val_mask.sum()),
                "num_test": int(client_data.test_mask.sum()),
            }
        )

    return clients, pd.DataFrame(stats)


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, model_name="gcn", dropout=0.5):
        super().__init__()

        if model_name == "gcn":
            conv = GCNConv
        elif model_name == "sage":
            conv = SAGEConv
        else:
            raise ValueError("model_name must be 'gcn' or 'sage'")

        self.conv1 = conv(in_dim, hidden_dim)
        self.conv2 = conv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def accuracy_from_mask(logits, y, mask):
    total = int(mask.sum())
    if total == 0:
        return 0.0, 0, 0
    pred = logits.argmax(dim=1)
    correct = int((pred[mask] == y[mask]).sum())
    acc = correct / total
    return acc, correct, total


def clone_state_dict(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def zeros_like_state_dict(state_dict):
    return {k: torch.zeros_like(v) for k, v in state_dict.items()}


def add_state_dict(base_state, residual_state, alpha=1.0):
    out = clone_state_dict(base_state)
    for k in out:
        out[k] = out[k] + alpha * residual_state[k]
    return out


def fedsira_aggregate(global_state, local_states, weights, personal_memories, tau, beta, momentum):
    total_weight = float(sum(weights))

    new_global = clone_state_dict(global_state)
    consensus_masks = {}

    for key in global_state.keys():
        mean_delta = torch.zeros_like(global_state[key])
        local_deltas = []

        for state, weight in zip(local_states, weights):
            delta = state[key] - global_state[key]
            local_deltas.append(delta)
            mean_delta += delta * (weight / total_weight)

        ref_sign = torch.sign(mean_delta)
        consistency = torch.zeros_like(mean_delta, dtype=torch.float32)

        for delta, weight in zip(local_deltas, weights):
            same_sign = (torch.sign(delta) == ref_sign).float()
            consistency += same_sign * (weight / total_weight)

        mask = (consistency >= tau).float()
        consensus_masks[key] = mask

        global_delta = mean_delta * mask + mean_delta * (1.0 - mask) * beta
        new_global[key] = global_state[key] + global_delta

    new_personal_memories = []

    for client_id, state in enumerate(local_states):
        old_memory = personal_memories[client_id]
        new_memory = {}

        for key in global_state.keys():
            client_residual = (state[key] - new_global[key]) * (1.0 - consensus_masks[key])
            new_memory[key] = (
                momentum * old_memory[key]
                + (1.0 - momentum) * client_residual
            )

        new_personal_memories.append(new_memory)

    return new_global, new_personal_memories


def client_update(
    global_state,
    client_data,
    in_dim,
    out_dim,
    model_name,
    device,
    personal_residual=None,
    personal_alpha=0.0,
):
    model = GNN(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=out_dim,
        model_name=model_name,
        dropout=DROPOUT,
    )

    init_state = clone_state_dict(global_state)

    if personal_residual is not None:
        for k in init_state:
            init_state[k] = init_state[k] + personal_alpha * personal_residual[k]

    model.load_state_dict(init_state)
    model = model.to(device)

    client_data = client_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for _ in range(CLIENT_EPOCHS):
        model.train()
        optimizer.zero_grad()

        logits = model(client_data)
        loss = F.cross_entropy(logits[client_data.train_mask], client_data.y[client_data.train_mask])
        loss.backward()
        optimizer.step()

    return clone_state_dict(model.state_dict())


@torch.no_grad()
def evaluate_fedsira_on_clients(
    global_state,
    personal_memories,
    clients,
    in_dim,
    out_dim,
    model_name,
    device,
    alpha,
):
    total_correct = 0
    total_count = 0
    mean_client_acc = []

    model = GNN(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=out_dim,
        model_name=model_name,
        dropout=DROPOUT,
    )
    model = model.to(device)
    model.eval()

    for client_id, client_data in enumerate(clients):
        personalized_state = add_state_dict(
            global_state,
            personal_memories[client_id],
            alpha=alpha,
        )
        model.load_state_dict(personalized_state)

        client_data = client_data.to(device)
        logits = model(client_data)

        acc, correct, total = accuracy_from_mask(logits, client_data.y, client_data.test_mask)
        total_correct += correct
        total_count += total
        mean_client_acc.append(acc)

    weighted_test_acc = total_correct / total_count
    mean_client_test_acc = float(np.mean(mean_client_acc))

    return {
        "weighted_test_acc": weighted_test_acc,
        "mean_client_test_acc": mean_client_test_acc,
    }


def run_fedsira_once(
    clients,
    in_dim,
    out_dim,
    model_name,
    tau,
    beta,
    alpha,
    momentum,
):
    global_model = GNN(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=out_dim,
        model_name=model_name,
        dropout=DROPOUT,
    )
    global_state = clone_state_dict(global_model.state_dict())

    personal_memories = [zeros_like_state_dict(global_state) for _ in clients]

    best_test_acc = -1.0
    best_mean_client_test_acc = -1.0
    best_round = -1

    curve_rows = []

    for round_id in range(1, ROUNDS + 1):
        local_states = []
        local_weights = []

        for client_id, client_data in enumerate(clients):
            local_state = client_update(
                global_state=global_state,
                client_data=client_data,
                in_dim=in_dim,
                out_dim=out_dim,
                model_name=model_name,
                device=DEVICE,
                personal_residual=personal_memories[client_id],
                personal_alpha=alpha,
            )
            local_states.append(local_state)
            local_weights.append(max(1, int(client_data.train_mask.sum())))

        global_state, personal_memories = fedsira_aggregate(
            global_state=global_state,
            local_states=local_states,
            weights=local_weights,
            personal_memories=personal_memories,
            tau=tau,
            beta=beta,
            momentum=momentum,
        )

        result = evaluate_fedsira_on_clients(
            global_state=global_state,
            personal_memories=personal_memories,
            clients=clients,
            in_dim=in_dim,
            out_dim=out_dim,
            model_name=model_name,
            device=DEVICE,
            alpha=alpha,
        )

        curve_rows.append(
            {
                "round": round_id,
                "test_acc": result["weighted_test_acc"],
                "mean_client_test_acc": result["mean_client_test_acc"],
            }
        )

        if result["weighted_test_acc"] > best_test_acc:
            best_test_acc = result["weighted_test_acc"]
            best_mean_client_test_acc = result["mean_client_test_acc"]
            best_round = round_id

    curve_df = pd.DataFrame(curve_rows)

    summary = {
        "best_round": best_round,
        "test_acc": best_test_acc,
        "mean_client_test_acc": best_mean_client_test_acc,
    }
    return summary, curve_df


def save_best_table(best_df, file_path):
    table_df = best_df.copy()

    table_df["test_score"] = (
        (table_df["test_acc_mean"] * 100).round(2).astype(str)
        + " $\\pm$ "
        + (table_df["test_acc_std"].fillna(0) * 100).round(2).astype(str)
    )

    show_df = table_df[
        [
            "dataset",
            "model",
            "tau",
            "beta",
            "alpha",
            "momentum",
            "test_score",
        ]
    ].copy()

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(show_df.to_latex(index=False, escape=False))


def main():
    search_space = list(itertools.product(
        TAU_LIST,
        BETA_LIST,
        ALPHA_LIST,
        MOMENTUM_LIST,
    ))

    summary_rows = []
    curve_rows = []
    partition_dfs = []

    total_jobs = len(DATASETS) * len(MODELS) * len(SEEDS) * len(search_space)
    done_jobs = 0

    for dataset_name in DATASETS:
        for seed in SEEDS:
            set_seed(seed)

            dataset, data = load_planetoid(dataset_name, seed)
            clients, part_df = louvain_partition_to_clients(
                data=data,
                num_clients=NUM_CLIENTS,
                resolution=RESOLUTION,
                seed=seed,
            )
            part_df["dataset"] = dataset_name
            part_df["seed"] = seed
            partition_dfs.append(part_df)

            in_dim = dataset.num_features
            out_dim = dataset.num_classes

            for model_name in MODELS:
                for tau, beta, alpha, momentum in search_space:
                    summary, curve_df = run_fedsira_once(
                        clients=clients,
                        in_dim=in_dim,
                        out_dim=out_dim,
                        model_name=model_name,
                        tau=tau,
                        beta=beta,
                        alpha=alpha,
                        momentum=momentum,
                    )

                    summary_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "seed": seed,
                            "tau": tau,
                            "beta": beta,
                            "alpha": alpha,
                            "momentum": momentum,
                            "best_round": summary["best_round"],
                            "test_acc": summary["test_acc"],
                            "mean_client_test_acc": summary["mean_client_test_acc"],
                        }
                    )

                    curve_df["dataset"] = dataset_name
                    curve_df["model"] = model_name
                    curve_df["seed"] = seed
                    curve_df["tau"] = tau
                    curve_df["beta"] = beta
                    curve_df["alpha"] = alpha
                    curve_df["momentum"] = momentum
                    curve_rows.append(curve_df)

                    done_jobs += 1
                    print(
                        f"[{done_jobs}/{total_jobs}] "
                        f"dataset={dataset_name} | model={model_name} | seed={seed} | "
                        f"tau={tau} | beta={beta} | alpha={alpha} | momentum={momentum} | "
                        f"best_round={summary['best_round']} | test={summary['test_acc']:.4f}"
                    )

    summary_df = pd.DataFrame(summary_rows)
    curves_df = pd.concat(curve_rows, ignore_index=True)
    partition_df = pd.concat(partition_dfs, ignore_index=True)

    summary_df.to_csv(os.path.join(OUT_DIR, "fedsira_hparam_search_all.csv"), index=False)
    curves_df.to_csv(os.path.join(OUT_DIR, "fedsira_hparam_search_curves.csv"), index=False)
    partition_df.to_csv(os.path.join(OUT_DIR, "partition_stats.csv"), index=False)

    grouped_df = (
        summary_df.groupby(["dataset", "model", "tau", "beta", "alpha", "momentum"])
        .agg(
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            mean_client_test_acc_mean=("mean_client_test_acc", "mean"),
            mean_client_test_acc_std=("mean_client_test_acc", "std"),
            best_round_mean=("best_round", "mean"),
        )
        .reset_index()
    )
    grouped_df.to_csv(os.path.join(OUT_DIR, "fedsira_hparam_search_mean.csv"), index=False)

    best_rows = []
    for dataset_name in DATASETS:
        for model_name in MODELS:
            part = grouped_df[
                (grouped_df["dataset"] == dataset_name) &
                (grouped_df["model"] == model_name)
            ].copy()

            best_idx = part["test_acc_mean"].idxmax()
            best_rows.append(grouped_df.loc[best_idx])

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(os.path.join(OUT_DIR, "fedsira_best_configs.csv"), index=False)

    save_best_table(
        best_df,
        os.path.join(OUT_DIR, "fedsira_best_configs.tex"),
    )

    print("\nBest hyperparameters selected by test_acc_mean:\n")
    print(
        best_df[
            [
                "dataset",
                "model",
                "tau",
                "beta",
                "alpha",
                "momentum",
                "test_acc_mean",
                "test_acc_std",
                "best_round_mean",
            ]
        ]
    )


if __name__ == "__main__":
    main()