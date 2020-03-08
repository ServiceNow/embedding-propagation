
import torch

class BaseWrapper(torch.nn.Module):
    def get_state_dict(self):
        raise NotImplementedError
    def load_state_dict(self, state_dict):
        raise NotImplementedError
