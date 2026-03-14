# Exp1

Modular node-classification experiment workspace for Planetoid-based federated graph learning.

## Run

```bash
cd src/Exp1
python main.py
```

## Directory layout

```text
src/Exp1/
├── __init__.py
├── README.md
├── config.py
├── data.py
├── main.py
├── models.py
├── plots.py
├── reporting.py
├── runners.py
├── utils.py
├── methods/
│   ├── __init__.py
│   ├── base.py
│   ├── fedavg.py
│   ├── fedsira.py
│   ├── placeholders.py
│   └── registry.py
└── results/
```

## Current active methods

- centralized
- local
- fedavg
- fedsira

## Reserved federated method slots

- slot_gfl_sota_1
- slot_gfl_sota_2
- slot_gfl_sota_3
- slot_gfl_sota_4
- slot_gfl_sota_5

To add a new federated baseline, implement a class under `methods/` using the interface in `methods/base.py`, then register it in `methods/registry.py`, and add its name to `federated_methods_to_run` in `config.py`.
