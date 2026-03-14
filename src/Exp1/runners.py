import copy
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

from config import Exp1Config
from models import GNN
from utils import accuracy_from_mask, clone_state_dict


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



def client_update(
    init_state,
    client_data,
    in_dim,
    out_dim,
    model_name,
    cfg: Exp1Config,
):
    model = GNN(
        in_dim=in_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=out_dim,
        model_name=model_name,
        dropout=cfg.dropout,
    )
    model.load_state_dict(clone_state_dict(init_state))
    model = model.to(cfg.device)

    client_data = client_data.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for _ in range(cfg.client_epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(client_data)
        loss = F.cross_entropy(logits[client_data.train_mask], client_data.y[client_data.train_mask])
        loss.backward()
        optimizer.step()

    return clone_state_dict(model.state_dict())


@torch.no_grad()
def evaluate_method_on_clients(
    method,
    global_state,
    context,
    clients,
    in_dim,
    out_dim,
    model_name,
    cfg: Exp1Config,
):
    model = GNN(
        in_dim=in_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=out_dim,
        model_name=model_name,
        dropout=cfg.dropout,
    )
    model = model.to(cfg.device)
    model.eval()

    total_correct = 0
    total_count = 0
    mean_client_acc = []

    for client_id, client_data in enumerate(clients):
        eval_state = method.personalize_for_evaluation(global_state, client_id, context)
        model.load_state_dict(eval_state)

        client_data = client_data.to(cfg.device)
        logits = model(client_data)
        acc, correct, total = accuracy_from_mask(logits, client_data.y, client_data.test_mask)

        total_correct += correct
        total_count += total
        mean_client_acc.append(acc)

    weighted_test_acc = total_correct / total_count if total_count > 0 else 0.0
    mean_client_test_acc = float(np.mean(mean_client_acc)) if mean_client_acc else 0.0
    return weighted_test_acc, mean_client_test_acc



def run_centralized(data, in_dim, out_dim, model_name, dataset_name, seed, cfg: Exp1Config):
    model = GNN(
        in_dim=in_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=out_dim,
        model_name=model_name,
        dropout=cfg.dropout,
    )
    model, curve_df = train_full_batch(
        model=model,
        data=data,
        epochs=cfg.central_epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        device=cfg.device,
    )
    metrics = evaluate_model(model, data, cfg.device)

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



def run_local_only(clients, in_dim, out_dim, model_name, dataset_name, seed, cfg: Exp1Config):
    total_correct = 0
    total_count = 0
    per_client_acc = []

    for client_data in clients:
        model = GNN(
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=out_dim,
            model_name=model_name,
            dropout=cfg.dropout,
        )
        model, _ = train_full_batch(
            model=model,
            data=client_data,
            epochs=cfg.local_epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            device=cfg.device,
        )
        metrics = evaluate_model(model, client_data, cfg.device)
        total_correct += metrics["test_correct"]
        total_count += metrics["test_total"]
        per_client_acc.append(metrics["test_acc"])

    weighted_test_acc = total_correct / total_count if total_count > 0 else 0.0
    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "method": "local",
        "seed": seed,
        "test_acc": weighted_test_acc,
        "mean_client_test_acc": float(np.mean(per_client_acc)) if per_client_acc else 0.0,
    }

    curve_df = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "model": model_name,
                "method": "local",
                "seed": seed,
                "round": cfg.local_epochs,
                "train_acc": np.nan,
                "val_acc": np.nan,
                "test_acc": weighted_test_acc,
            }
        ]
    )
    return summary, curve_df



def run_federated(
    clients,
    in_dim,
    out_dim,
    model_name,
    dataset_name,
    seed,
    method,
    cfg: Exp1Config,
):
    global_model = GNN(
        in_dim=in_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=out_dim,
        model_name=model_name,
        dropout=cfg.dropout,
    )
    global_state = clone_state_dict(global_model.state_dict())
    context: Dict = method.initialize_context(global_state, len(clients))

    curve_rows = []

    for round_id in range(1, cfg.rounds + 1):
        local_states = []
        local_weights = []

        for client_id, client_data in enumerate(clients):
            init_state = method.get_client_init_state(global_state, client_id, context)
            local_state = client_update(
                init_state=init_state,
                client_data=client_data,
                in_dim=in_dim,
                out_dim=out_dim,
                model_name=model_name,
                cfg=cfg,
            )
            local_states.append(local_state)
            local_weights.append(max(1, int(client_data.train_mask.sum())))

        global_state, context = method.aggregate(
            global_state=global_state,
            local_states=local_states,
            weights=local_weights,
            context=context,
        )
        weighted_test_acc, mean_client_test_acc = evaluate_method_on_clients(
            method=method,
            global_state=global_state,
            context=context,
            clients=clients,
            in_dim=in_dim,
            out_dim=out_dim,
            model_name=model_name,
            cfg=cfg,
        )

        curve_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "method": method.name,
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
        "method": method.name,
        "seed": seed,
        "test_acc": final_test_acc,
        "mean_client_test_acc": final_mean_client_test_acc,
    }
    return summary, pd.DataFrame(curve_rows)
