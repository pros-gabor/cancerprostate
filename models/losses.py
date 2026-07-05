import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalFocalLoss(nn.Module):
    """Class-weighted focal loss used for ordinal Grade Group classification.

    Note: This implementation is focal loss with optional class weights. If you need a strict
    distance-aware ordinal loss, add an ordinal-distance penalty separately.
    """

    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        y = F.one_hot(targets, num_classes=logits.shape[-1]).float()
        focal = (1.0 - probs).pow(self.gamma)
        loss = -focal * y * log_probs
        if self.class_weights is not None:
            loss = loss * self.class_weights.view(1, -1)
        return loss.sum(dim=-1).mean()


def graph_smoothness_loss(S_prop: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Smoothness regularizer sum_k s_k^T L s_k. S_prop: [M,K]."""
    D = torch.diag(A.sum(dim=1))
    L = D - A
    return torch.trace(S_prop.t() @ L @ S_prop) / (S_prop.numel() + 1e-6)
