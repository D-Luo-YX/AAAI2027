from methods.base import FederatedMethod


class ReservedMethodSlot(FederatedMethod):
    def __init__(self, cfg, name: str):
        super().__init__(cfg)
        self.name = name

    def aggregate(self, global_state, local_states, weights, context):
        raise NotImplementedError(
            f"Method slot '{self.name}' is reserved for future Exp1 federated baselines. "
            "Implement get_client_init_state / aggregate / personalize_for_evaluation in methods/."
        )
