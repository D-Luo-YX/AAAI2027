import copy
import os
import random

import matplotlib.pyplot as plt
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

CENTRAL_EPOCHS = 200
LOCAL_EPOCHS = 200
ROUNDS = 100
CLIENT_EPOCHS = 3

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
RESOLUTION = 1.0

SIRA_CONSENSUS_TAU = 0.6
SIRA_RESIDUAL_BETA = 0.25
SIRA_PERSONAL_ALPHA = 1.0
SIRA_MEMORY_MOMENTUM = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "results/fedsira_nc"
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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

@torch.no_grad()
def evaluate_model(model: nn.Module, data: Data, device):
    model.eval()
    data = data.to(device)
    logits = model(data)

    train_acc, train_correct, train_total = accuracy_from_mask(logits, data.y, data.train_mask)
    val_acc, val_correct, val_total = accuracy_from_mask(logits, data.y, data.val_mask)
    test_acc, test_correct, test_total = accuracy_from_mask(logits, data.y, data.test_mask)

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "train_correct": train_correct,
        "val_correct": val_correct,
        "test_correct": test_correct,
        "train_total": train_total,
        "val_total": val_total,
        "test_total": test_total,
    }


def train_full_batch(model, data, epochs, lr, weight_decay, device):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    curve = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        metrics = evaluate_model(model, data, device)
        curve.append(
            {
                "epoch": epoch,
                "train_acc": metrics["train_acc"],
                "val_acc": metrics["val_acc"],
                "test_acc": metrics["test_acc"],
            }
        )

        if metrics["val_acc"] > best_val:
            best_val = metrics["val_acc"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, pd.DataFrame(curve)


def aggregate_state_dicts(state_dicts, weights):
    total_weight = float(sum(weights))
    new_state = copy.deepcopy(state_dicts[0])

    for key in new_state.keys():
        new_state[key] = torch.zeros_like(new_state[key])
        for state, weight in zip(state_dicts, weights):
            new_state[key] += state[key] * (weight / total_weight)

    return new_state


def fedavg_aggregate(state_dicts, weights):
    return aggregate_state_dicts(state_dicts, weights)

def fedsira_aggregate(global_state, local_states, weights, personal_memories):
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

        mask = (consistency >= SIRA_CONSENSUS_TAU).float()
        consensus_masks[key] = mask

        global_delta = mean_delta * mask + mean_delta * (1.0 - mask) * SIRA_RESIDUAL_BETA
        new_global[key] = global_state[key] + global_delta

    new_personal_memories = []

    for client_id, state in enumerate(local_states):
        old_memory = personal_memories[client_id]
        new_memory = {}

        for key in global_state.keys():
            client_residual = (state[key] - new_global[key]) * (1.0 - consensus_masks[key])
            new_memory[key] = (
                SIRA_MEMORY_MOMENTUM * old_memory[key]
                + (1.0 - SIRA_MEMORY_MOMENTUM) * client_residual
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
def evaluate_global_on_clients(global_state, clients, in_dim, out_dim, model_name, device):
    model = GNN(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=out_dim,
        model_name=model_name,
        dropout=DROPOUT,
    )
    model.load_state_dict(global_state)
    model = model.to(device)
    model.eval()

    total_correct = 0
    total_count = 0
    mean_client_acc = []

    for client_data in clients:
        client_data = client_data.to(device)
        logits = model(client_data)
        acc, correct, total = accuracy_from_mask(logits, client_data.y, client_data.test_mask)

        total_correct += correct
        total_count += total
        mean_client_acc.append(acc)

    weighted_test_acc = total_correct / total_count
    mean_client_test_acc = float(np.mean(mean_client_acc))
    return weighted_test_acc, mean_client_test_acc

@torch.no_grad()
def evaluate_fedsira_on_clients(
    global_state,
    personal_memories,
    clients,
    in_dim,
    out_dim,
    model_name,
    device,
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
            alpha=SIRA_PERSONAL_ALPHA,
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
    return weighted_test_acc, mean_client_test_acc

def run_centralized(data, in_dim, out_dim, model_name, dataset_name, seed):
    model = GNN(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=out_dim,
        model_name=model_name,
        dropout=DROPOUT,
    )
    model, curve_df = train_full_batch(
        model=model,
        data=data,
        epochs=CENTRAL_EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
    )
    metrics = evaluate_model(model, data, DEVICE)

    curve_df["dataset"] = dataset_name
    curve_df["model"] = model_name
    curve_df["method"] = "centralized"
    curve_df["seed"] = seed
    curve_df.rename(columns={"epoch": "round"}, inplace=True)

    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "method": "centralized",
        "seed": seed,
        "test_acc": metrics["test_acc"],
        "mean_client_test_acc": np.nan,
    }
    return summary, curve_df


def run_local_only(clients, in_dim, out_dim, model_name, dataset_name, seed):
    total_correct = 0
    total_count = 0
    per_client_acc = []

    for client_data in clients:
        model = GNN(
            in_dim=in_dim,
            hidden_dim=HIDDEN_DIM,
            out_dim=out_dim,
            model_name=model_name,
            dropout=DROPOUT,
        )
        model, _ = train_full_batch(
            model=model,
            data=client_data,
            epochs=LOCAL_EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            device=DEVICE,
        )
        metrics = evaluate_model(model, client_data, DEVICE)
        total_correct += metrics["test_correct"]
        total_count += metrics["test_total"]
        per_client_acc.append(metrics["test_acc"])

    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "method": "local",
        "seed": seed,
        "test_acc": total_correct / total_count,
        "mean_client_test_acc": float(np.mean(per_client_acc)),
    }

    curve_df = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "model": model_name,
                "method": "local",
                "seed": seed,
                "round": LOCAL_EPOCHS,
                "train_acc": np.nan,
                "val_acc": np.nan,
                "test_acc": total_correct / total_count,
            }
        ]
    )
    return summary, curve_df


# def run_federated(clients, in_dim, out_dim, model_name, dataset_name, seed, method_name="fedavg"):
#     global_model = GNN(
#         in_dim=in_dim,
#         hidden_dim=HIDDEN_DIM,
#         out_dim=out_dim,
#         model_name=model_name,
#         dropout=DROPOUT,
#     )
#     global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
#
#     curve_rows = []
#
#     for round_id in range(1, ROUNDS + 1):
#         local_states = []
#         local_weights = []
#
#         for client_data in clients:
#             local_state = client_update(
#                 global_state=global_state,
#                 client_data=client_data,
#                 in_dim=in_dim,
#                 out_dim=out_dim,
#                 model_name=model_name,
#                 device=DEVICE,
#             )
#             local_states.append(local_state)
#             local_weights.append(max(1, int(client_data.train_mask.sum())))
#
#         if method_name == "fedavg":
#             global_state = fedavg_aggregate(local_states, local_weights)
#         elif method_name == "fedsira":
#             global_state = fedsira_aggregate(local_states, local_weights)
#         else:
#             raise ValueError("Unknown method_name")
#
#         weighted_test_acc, mean_client_test_acc = evaluate_global_on_clients(
#             global_state=global_state,
#             clients=clients,
#             in_dim=in_dim,
#             out_dim=out_dim,
#             model_name=model_name,
#             device=DEVICE,
#         )
#
#         curve_rows.append(
#             {
#                 "dataset": dataset_name,
#                 "model": model_name,
#                 "method": method_name,
#                 "seed": seed,
#                 "round": round_id,
#                 "train_acc": np.nan,
#                 "val_acc": np.nan,
#                 "test_acc": weighted_test_acc,
#                 "mean_client_test_acc": mean_client_test_acc,
#             }
#         )
#
#     final_test_acc = curve_rows[-1]["test_acc"]
#     final_mean_client_test_acc = curve_rows[-1]["mean_client_test_acc"]
#
#     summary = {
#         "dataset": dataset_name,
#         "model": model_name,
#         "method": method_name,
#         "seed": seed,
#         "test_acc": final_test_acc,
#         "mean_client_test_acc": final_mean_client_test_acc,
#     }
#     return summary, pd.DataFrame(curve_rows)

def run_federated(clients, in_dim, out_dim, model_name, dataset_name, seed, method_name="fedavg"):
    global_model = GNN(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=out_dim,
        model_name=model_name,
        dropout=DROPOUT,
    )
    global_state = clone_state_dict(global_model.state_dict())

    personal_memories = None
    if method_name == "fedsira":
        personal_memories = [zeros_like_state_dict(global_state) for _ in clients]

    curve_rows = []

    for round_id in range(1, ROUNDS + 1):
        local_states = []
        local_weights = []

        for client_id, client_data in enumerate(clients):
            personal_residual = None
            personal_alpha = 0.0

            if method_name == "fedsira":
                personal_residual = personal_memories[client_id]
                personal_alpha = SIRA_PERSONAL_ALPHA

            local_state = client_update(
                global_state=global_state,
                client_data=client_data,
                in_dim=in_dim,
                out_dim=out_dim,
                model_name=model_name,
                device=DEVICE,
                personal_residual=personal_residual,
                personal_alpha=personal_alpha,
            )

            local_states.append(local_state)
            local_weights.append(max(1, int(client_data.train_mask.sum())))

        if method_name == "fedavg":
            global_state = fedavg_aggregate(local_states, local_weights)
            weighted_test_acc, mean_client_test_acc = evaluate_global_on_clients(
                global_state=global_state,
                clients=clients,
                in_dim=in_dim,
                out_dim=out_dim,
                model_name=model_name,
                device=DEVICE,
            )

        elif method_name == "fedsira":
            global_state, personal_memories = fedsira_aggregate(
                global_state=global_state,
                local_states=local_states,
                weights=local_weights,
                personal_memories=personal_memories,
            )
            weighted_test_acc, mean_client_test_acc = evaluate_fedsira_on_clients(
                global_state=global_state,
                personal_memories=personal_memories,
                clients=clients,
                in_dim=in_dim,
                out_dim=out_dim,
                model_name=model_name,
                device=DEVICE,
            )

        else:
            raise ValueError("Unknown method_name")

        curve_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "method": method_name,
                "seed": seed,
                "round": round_id,
                "train_acc": np.nan,
                "val_acc": np.nan,
                "test_acc": weighted_test_acc,
                "mean_client_test_acc": mean_client_test_acc,
            }
        )

    final_test_acc = curve_rows[-1]["test_acc"]
    final_mean_client_test_acc = curve_rows[-1]["mean_client_test_acc"]

    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "method": method_name,
        "seed": seed,
        "test_acc": final_test_acc,
        "mean_client_test_acc": final_mean_client_test_acc,
    }
    return summary, pd.DataFrame(curve_rows)

def plot_learning_curves(curves_df):
    fed_curves = curves_df[curves_df["method"].isin(["fedavg", "fedsira"])].copy()
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
                (grouped["dataset"] == dataset_name) &
                (grouped["model"] == model_name)
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
            plt.savefig(os.path.join(OUT_DIR, f"curve_{dataset_name}_{model_name}.png"), dpi=200)
            plt.close()


def plot_final_bars(summary_df):
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
                    (part["dataset"] == dataset_name) &
                    (part["method"] == method_name)
                ]
                vals.append(row["mean"].iloc[0] if len(row) > 0 else 0.0)
                errs.append(row["std"].iloc[0] if len(row) > 0 else 0.0)

            plt.bar(x + i * width, vals, width=width, yerr=errs, label=method_name)

        plt.xticks(x + width * (len(methods) - 1) / 2, datasets)
        plt.ylabel("Test accuracy")
        plt.title(f"Final accuracy - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"bar_{model_name}.png"), dpi=200)
        plt.close()


def plot_partition_stats(partition_df):
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
        plt.savefig(os.path.join(OUT_DIR, f"client_size_{dataset_name}.png"), dpi=200)
        plt.close()


def save_latex_table(summary_df):
    grouped = (
        summary_df.groupby(["dataset", "model", "method"])["test_acc"]
        .agg(["mean", "std"])
        .reset_index()
    )

    grouped["score"] = (
        (grouped["mean"] * 100).round(2).astype(str) +
        " $\\pm$ " +
        (grouped["std"].fillna(0) * 100).round(2).astype(str)
    )

    table = grouped.pivot_table(
        index=["dataset", "model"],
        columns="method",
        values="score",
        aggfunc="first",
    )

    with open(os.path.join(OUT_DIR, "table_main.tex"), "w", encoding="utf-8") as f:
        f.write(table.to_latex(escape=False))


def main():
    summary_rows = []
    curve_dfs = []
    partition_dfs = []

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
                centralized_summary, centralized_curve = run_centralized(
                    data=data,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    seed=seed,
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
                )
                summary_rows.append(local_summary)
                curve_dfs.append(local_curve)

                fedavg_summary, fedavg_curve = run_federated(
                    clients=clients,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    seed=seed,
                    method_name="fedavg",
                )
                summary_rows.append(fedavg_summary)
                curve_dfs.append(fedavg_curve)

                fedsira_summary, fedsira_curve = run_federated(
                    clients=clients,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    seed=seed,
                    method_name="fedsira",
                )
                summary_rows.append(fedsira_summary)
                curve_dfs.append(fedsira_curve)

                print(
                    f"Done | dataset={dataset_name} | seed={seed} | model={model_name}"
                )

    summary_df = pd.DataFrame(summary_rows)
    curves_df = pd.concat(curve_dfs, ignore_index=True)
    partition_df = pd.concat(partition_dfs, ignore_index=True)

    summary_df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    curves_df.to_csv(os.path.join(OUT_DIR, "curves.csv"), index=False)
    partition_df.to_csv(os.path.join(OUT_DIR, "partition_stats.csv"), index=False)

    plot_learning_curves(curves_df)
    plot_final_bars(summary_df)
    plot_partition_stats(partition_df)
    save_latex_table(summary_df)

    print(summary_df.groupby(["dataset", "model", "method"])["test_acc"].mean())


if __name__ == "__main__":
    main()