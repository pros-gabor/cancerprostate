import torch
import torch.nn.functional as F


class SemanticAlignment(torch.nn.Module):
    """Cosine alignment between refined patch features and concept anchors."""

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, concept_embeds: torch.Tensor) -> torch.Tensor:
        """Return semantic evidence matrix S in R^{M x K}."""
        z = F.normalize(z, dim=-1)
        concept_embeds = F.normalize(concept_embeds, dim=-1)
        return z @ concept_embeds.t()
