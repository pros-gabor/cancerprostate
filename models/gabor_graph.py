import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableGaborGraph(nn.Module):
    """Build morphology-aware adjacency using spatial and learnable Gabor texture similarity."""

    def __init__(self, orientations=6, frequencies=3, kernel_size=15, tau_p=1.0, tau_g=1.0, k=8):
        super().__init__()
        self.T = orientations
        self.F = frequencies
        self.kernel_size = kernel_size
        self.tau_p = tau_p
        self.tau_g = tau_g
        self.k = k
        n_filters = orientations * frequencies

        self.theta = nn.Parameter(torch.linspace(0, math.pi, orientations).repeat_interleave(frequencies))
        self.freq = nn.Parameter(torch.linspace(0.05, 0.25, frequencies).repeat(orientations))
        self.sigma = nn.Parameter(torch.ones(n_filters) * 3.0)
        self.gamma = nn.Parameter(torch.ones(n_filters) * 0.5)
        self.phi = nn.Parameter(torch.zeros(n_filters))

    def _make_kernels(self, device):
        r = self.kernel_size // 2
        yy, xx = torch.meshgrid(
            torch.arange(-r, r + 1, device=device),
            torch.arange(-r, r + 1, device=device),
            indexing="ij",
        )
        kernels = []
        for i in range(self.T * self.F):
            theta = self.theta[i]
            freq = F.softplus(self.freq[i])
            sigma = F.softplus(self.sigma[i]) + 1e-4
            gamma = F.softplus(self.gamma[i]) + 1e-4
            phi = self.phi[i]
            x_p = xx * torch.cos(theta) + yy * torch.sin(theta)
            y_p = -xx * torch.sin(theta) + yy * torch.cos(theta)
            env = torch.exp(-(x_p ** 2 + gamma ** 2 * y_p ** 2) / (2 * sigma ** 2))
            wave = torch.cos(2 * math.pi * freq * x_p + phi)
            k = env * wave
            k = k - k.mean()
            k = k / (k.norm() + 1e-6)
            kernels.append(k)
        return torch.stack(kernels).unsqueeze(1)

    def gabor_descriptors(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute descriptor g_m from raw image patches. patches: [M,3,H,W]."""
        if patches is None:
            raise ValueError("Raw patches are required for learnable Gabor graph construction.")
        x = patches.mean(dim=1, keepdim=True)  # grayscale
        kernels = self._make_kernels(x.device)
        resp = F.conv2d(x, kernels, padding=self.kernel_size // 2)
        energy = resp.pow(2).mean(dim=(2, 3))
        return F.normalize(energy, dim=-1)

    def forward(self, coords: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """Return row-normalized adjacency A_tilde in R^{M x M}."""
        M = coords.shape[0]
        coord_dist = torch.cdist(coords, coords, p=2).pow(2)
        s_space = torch.exp(-coord_dist / max(self.tau_p, 1e-6))

        g = self.gabor_descriptors(patches)
        g_dist = torch.cdist(g, g, p=2).pow(2)
        s_morph = torch.exp(-g_dist / max(self.tau_g, 1e-6))

        A = s_space * s_morph
        A.fill_diagonal_(0.0)

        if self.k is not None and self.k < M:
            _, idx = torch.topk(s_space, k=self.k, dim=1)
            mask = torch.zeros_like(A)
            mask.scatter_(1, idx, 1.0)
            A = A * mask

        D = A.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return A / D
