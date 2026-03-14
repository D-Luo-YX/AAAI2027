from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch


@dataclass
class Exp1Config:
    datasets: List[str] = field(default_factory=lambda: ["Cora", "CiteSeer", "PubMed"])
    models: List[str] = field(default_factory=lambda: ["gcn", "sage"])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    num_clients: int = 10
    hidden_dim: int = 64
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4

    central_epochs: int = 200
    local_epochs: int = 200
    rounds: int = 100
    client_epochs: int = 3

    train_ratio: float = 0.6
    val_ratio: float = 0.2
    resolution: float = 1.0

    sira_consensus_tau: float = 0.6
    sira_residual_beta: float = 0.25
    sira_personal_alpha: float = 1.0
    sira_memory_momentum: float = 0.5

    data_root: str = "data/Planetoid"

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    results_dir: Path = field(init=False)
    device: torch.device = field(init=False)

    federated_methods_to_run: List[str] = field(default_factory=lambda: ["fedavg", "fedsira"])
    reserved_method_slots: List[str] = field(
        default_factory=lambda: [
            "slot_gfl_sota_1",
            "slot_gfl_sota_2",
            "slot_gfl_sota_3",
            "slot_gfl_sota_4",
            "slot_gfl_sota_5",
        ]
    )

    def __post_init__(self):
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
