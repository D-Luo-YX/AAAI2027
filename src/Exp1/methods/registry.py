from config import Exp1Config
from methods.fedavg import FedAvgMethod
from methods.fedsira import FedSIRAMethod
from methods.placeholders import ReservedMethodSlot



def build_method_registry(cfg: Exp1Config):
    registry = {
        "fedavg": FedAvgMethod(cfg),
        "fedsira": FedSIRAMethod(cfg),
    }

    for slot_name in cfg.reserved_method_slots:
        registry[slot_name] = ReservedMethodSlot(cfg, slot_name)

    return registry
