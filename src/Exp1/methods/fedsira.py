import torch

from methods.base import FederatedMethod
from utils import add_state_dict, clone_state_dict, zeros_like_state_dict


class FedSIRAMethod(FederatedMethod):
    name = "fedsira"

    def initialize_context(self, global_state, num_clients: int):
        return {
            "personal_memories": [zeros_like_state_dict(global_state) for _ in range(num_clients)]
        }

    def get_client_init_state(self, global_state, client_id: int, context):
        personal_memory = context["personal_memories"][client_id]
        return add_state_dict(
            global_state,
            personal_memory,
            alpha=self.cfg.sira_personal_alpha,
        )

    def aggregate(self, global_state, local_states, weights, context):
        total_weight = float(sum(weights))
        new_global = clone_state_dict(global_state)
        consensus_masks = {}

        for key in global_state:
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

            mask = (consistency >= self.cfg.sira_consensus_tau).float()
            consensus_masks[key] = mask

            global_delta = (
                mean_delta * mask
                + mean_delta * (1.0 - mask) * self.cfg.sira_residual_beta
            )
            new_global[key] = global_state[key] + global_delta

        new_personal_memories = []
        for client_id, state in enumerate(local_states):
            old_memory = context["personal_memories"][client_id]
            new_memory = {}
            for key in global_state:
                client_residual = (state[key] - new_global[key]) * (1.0 - consensus_masks[key])
                new_memory[key] = (
                    self.cfg.sira_memory_momentum * old_memory[key]
                    + (1.0 - self.cfg.sira_memory_momentum) * client_residual
                )
            new_personal_memories.append(new_memory)

        context["personal_memories"] = new_personal_memories
        return new_global, context

    def personalize_for_evaluation(self, global_state, client_id: int, context):
        return add_state_dict(
            global_state,
            context["personal_memories"][client_id],
            alpha=self.cfg.sira_personal_alpha,
        )
