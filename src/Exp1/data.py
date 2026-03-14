from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph, to_networkx

from config import Exp1Config



def make_stratified_masks(
    y: torch.Tensor,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    num_nodes = y.size(0)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    classes = torch.unique(y).cpu().tolist()
    for cls in classes:
        idx = torch.where(y == cls)[0].cpu().numpy()
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



def load_planetoid(name: str, seed: int, cfg: Exp1Config):
    dataset = Planetoid(root=cfg.data_root, name=name)
    data = dataset[0]

    train_mask, val_mask, test_mask = make_stratified_masks(
        data.y,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=seed,
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return dataset, data



def louvain_partition_to_clients(
    data: Data,
    num_clients: int = 10,
    resolution: float = 1.0,
    seed: int = 0,
) -> Tuple[list, pd.DataFrame]:
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

    for community in communities:
        idx = int(np.argmin(bucket_sizes))
        buckets[idx].extend(community)
        bucket_sizes[idx] += len(community)

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
