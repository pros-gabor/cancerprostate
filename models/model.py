import torch
import torch.nn as nn
import torch.nn.functional as F
from .semantic_alignment import SemanticAlignment
from .gabor_graph import LearnableGaborGraph


class MorphologyAwareSemanticGraph(nn.Module):
    """End-to-end model for morphology-aware semantic evidence aggregation."""

    def __init__(self, visual_dim, fm_dim, text_dim, num_classes, num_concepts,
                 top_m=64, hidden_dim=256, graph_k=8, propagation_steps=1,
                 gabor_orientations=6, gabor_frequencies=3, tau_p=1.0, tau_g=1.0):
        super().__init__()
        self.top_m = top_m
        self.L = propagation_steps

        self.score_head = nn.Linear(fm_dim, 1)
        self.visual_proj = nn.Linear(visual_dim, text_dim) if visual_dim != text_dim else nn.Identity()
        self.alignment = SemanticAlignment()
        self.graph = LearnableGaborGraph(
            orientations=gabor_orientations,
            frequencies=gabor_frequencies,
            tau_p=tau_p,
            tau_g=tau_g,
            k=graph_k,
        )
        self.classifier = nn.Linear(num_concepts, num_classes)

    def refine_patches(self, h, u):
        scores = torch.sigmoid(self.score_head(u)).squeeze(-1)
        k = min(self.top_m, h.shape[0])
        top_scores, idx = torch.topk(scores, k=k, largest=True)
        z = h[idx] * top_scores.unsqueeze(-1)
        return z, idx, top_scores

    def forward(self, features, fm_features, coords, patches, concept_embeds):
        z, idx, scores = self.refine_patches(features, fm_features)
        z = self.visual_proj(z)
        coords_sel = coords[idx]
        patches_sel = patches[idx] if patches is not None else None

        S = self.alignment(z, concept_embeds)  # [M,K]
        A = self.graph(coords_sel, patches_sel)  # [M,M]

        S_prop = S
        for _ in range(self.L):
            S_prop = A @ S_prop

        alpha = torch.softmax(S_prop, dim=0)
        e = (alpha * S_prop).sum(dim=0)  # [K]
        logits = self.classifier(e.unsqueeze(0)).squeeze(0)
        return {
            "logits": logits,
            "evidence": S,
            "evidence_prop": S_prop,
            "adjacency": A,
            "selected_indices": idx,
            "scores": scores,
        }
