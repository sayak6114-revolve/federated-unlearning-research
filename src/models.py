import copy
import torch
import torch.nn as nn

class AdultMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for binary classification on the Adult Census dataset.
    """
    def __init__(self, input_dim):
        super(AdultMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------
# Parameter Manipulation & Vector Arithmetic Utilities
# -------------------------------------------------------------

def clone_weights(model):
    """Deep clones the state_dict of a PyTorch model."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

def load_weights(model, weights):
    """Loads a state_dict into the model."""
    model.load_state_dict(weights)

def compute_update(initial_weights, trained_weights):
    """
    Computes parameter update vector U = trained_weights - initial_weights.
    """
    return {k: trained_weights[k] - initial_weights[k] for k in initial_weights}

def apply_update(base_weights, update_vector):
    """
    Computes updated weights M = base_weights + update_vector.
    """
    return {k: base_weights[k] + update_vector[k] for k in base_weights}

def compute_l2_norm(update_vector):
    """
    Computes the overall Euclidean (L2) norm ||U||_2 across all layers in the model.
    """
    total_sq = 0.0
    for v in update_vector.values():
        if v.dtype in (torch.float32, torch.float64, torch.float16):
            total_sq += torch.sum(v ** 2).item()
    return total_sq ** 0.5

def compute_model_l2_distance(weights_a, weights_b):
    """
    Computes the parameter Euclidean distance ||w_a - w_b||_2 between two model states.
    """
    total_sq = 0.0
    for k in weights_a:
        if weights_a[k].dtype in (torch.float32, torch.float64, torch.float16):
            diff = weights_a[k] - weights_b[k]
            total_sq += torch.sum(diff ** 2).item()
    return total_sq ** 0.5
