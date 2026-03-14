from methods.base import FederatedMethod
from utils import weighted_average_state_dicts


class FedAvgMethod(FederatedMethod):
    name = "fedavg"

    def aggregate(self, global_state, local_states, weights, context):
        new_global = weighted_average_state_dicts(local_states, weights)
        return new_global, context
