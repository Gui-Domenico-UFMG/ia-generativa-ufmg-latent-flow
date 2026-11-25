"""Figuras do VAE (para validar a base e usar nos slides)."""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch

from .dataset import CLASS_NAMES, NUM_CLASSES, index_to_char
from .vae import VAE


def _finish(fig, save_path: Optional[str], show: bool):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_dataset_grid(images, labels, n: int = 36, save_path=None, show=True):
    """Grade com exemplos reais do dataset (confere orientação e rótulos)."""
    cols = 12
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.0, rows * 1.15))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(index_to_char(int(labels[i])), fontsize=9)
    fig.suptitle("EMNIST-36: exemplos reais", fontsize=11)
    _finish(fig, save_path, show)


def plot_loss_curves(history: Dict[str, List[float]], save_path=None, show=True):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    for ax, key, title in zip(
        axes, ["loss", "recon_error", "latent_error"], ["Loss total", "Reconstrução R(x)", "Latente L(x)"]
    ):
        ax.plot(history[f"train_{key}"], label="treino")
        ax.plot(history[f"test_{key}"], label="teste")
        ax.set_title(title)
        ax.set_xlabel("época")
        ax.legend()
        ax.grid(alpha=0.3)
    _finish(fig, save_path, show)


@torch.no_grad()
def plot_reconstructions(vae: VAE, images, labels=None, n: int = 12, save_path=None, show=True):
    """Topo: x original. Baixo: x_hat = mu_D(mu_E(x))."""
    vae.eval()
    device = next(vae.parameters()).device
    x = images[:n].to(device)
    mu, _ = vae.encode(x)
    x_hat = vae.decode(mu).cpu()
    fig, axes = plt.subplots(2, n, figsize=(n * 1.1, 2.6))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(x_hat[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
        if labels is not None:
            axes[0, i].set_title(index_to_char(int(labels[i])), fontsize=9)
    fig.suptitle("Original (topo) vs. reconstrução μ_D(μ_E(x)) (baixo)", fontsize=11)
    _finish(fig, save_path, show)


@torch.no_grad()
def plot_prior_samples(vae: VAE, n: int = 64, seed: int = 0, save_path=None, show=True):
    """
    Decodifica z ~ N(0, I) direto, SEM o Flow. Serve de baseline:
    o Flow deve gerar caracteres melhores que isso.
    """
    vae.eval()
    device = next(vae.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n, vae.latent_dim, generator=g).to(device)
    imgs = vae.decode(z).cpu()
    cols = 8
    fig, axes = plt.subplots(n // cols, cols, figsize=(cols * 1.0, n // cols * 1.0))
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle("VAE sozinho: μ_D(z), z ~ N(0, I)  (baseline sem Flow)", fontsize=10)
    _finish(fig, save_path, show)


@torch.no_grad()
def plot_interpolation(vae: VAE, x_a, x_b, steps: int = 10, save_path=None, show=True):
    """Interpolação linear no latente entre duas imagens: mostra se o espaço latente é 'suave'."""
    vae.eval()
    device = next(vae.parameters()).device
    mu_a, _ = vae.encode(x_a[None].to(device))
    mu_b, _ = vae.encode(x_b[None].to(device))
    t = torch.linspace(0, 1, steps, device=device)[:, None]
    imgs = vae.decode((1 - t) * mu_a + t * mu_b).cpu()
    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.1, 1.4))
    for ax, img in zip(axes, imgs):
        ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle("Interpolação linear no espaço latente", fontsize=10)
    _finish(fig, save_path, show)


def plot_latent_pca(mu: torch.Tensor, y: torch.Tensor, max_points: int = 8000, save_path=None, show=True):
    """Projeção 2D (PCA) dos mu_E(x), colorida por classe, com o caractere no centro de cada classe."""
    idx = torch.randperm(len(mu))[:max_points]
    m, yy = mu[idx], y[idx]
    m = m - m.mean(0)
    _, _, v = torch.pca_lowrank(m, q=2, center=False)
    p = m @ v[:, :2]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(p[:, 0], p[:, 1], c=yy, cmap="nipy_spectral", s=2, alpha=0.5, vmin=0, vmax=NUM_CLASSES - 1)
    for c in range(NUM_CLASSES):
        sel = yy == c
        if sel.any():
            cx, cy = p[sel].mean(0)
            ax.text(float(cx), float(cy), CLASS_NAMES[c], fontsize=10, weight="bold", ha="center", va="center")
    ax.set_title("Espaço latente μ_E(x) — PCA 2D")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    _finish(fig, save_path, show)
