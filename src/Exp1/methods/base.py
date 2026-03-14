from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from config import Exp1Config


class FederatedMethod(ABC):
    name: str = "base"

    def __init__(self, cfg: Exp1Config):
        self.cfg = cfg

    def initialize_context(self, global_state, num_clients: int) -> Dict[str, Any]:
        return {}

    def get_client_init_state(self, global_state, client_id: int, context: Dict[str, Any]):
        return global_state

    @abstractmethod
    def aggregate(
        self,
        global_state,
        local_states: List,
        weights: List[int],
        context: Dict[str, Any],
    ) -> Tuple[Dict, Dict[str, Any]]:
        raise NotImplementedError

    def personalize_for_evaluation(self, global_state, client_id: int, context: Dict[str, Any]):
        return global_state
