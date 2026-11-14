from typing import List, Optional
import torch
import matplotlib.pyplot as plt
from .dataset import CLASS_NAMES, index_to_char
from .vae import VAE
from .flow import LatentVelocityField


@torch.no_grad()
def plot_vae_reconstructions(
    vae: VAE,
    images: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    num_display: int = 10,
    save_path: Optional[str] = None,
):
    """Plota as imagens originais x vs. reconstruções determinísticas mu_D(mu_E(x))."""
    vae.eval()
    device = next(vae.parameters()).device
    x = images[:num_display].to(device)
    mu_e, _ = vae.encode(x)
    x_hat = vae.decode(mu_e)

    x_np = x.cpu().squeeze(1).numpy()
    x_hat_np = x_hat.cpu().squeeze(1).numpy()

    fig, axes = plt.subplots(2, num_display, figsize=(num_display * 1.4, 3.0))
    for i in range(num_display):
        axes[0, i].imshow(x_np[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if labels is not None:
            axes[0, i].set_title(f"Real: {index_to_char(int(labels[i]))}", fontsize=9)

        axes[1, i].imshow(x_hat_np[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Recon", fontsize=9)

    plt.suptitle("VAE — Originais (topo) vs. Reconstruções x̂ = μ_D(μ_E(x)) (base)", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def plot_all_36_classes(
    vae: VAE,
    flow: LatentVelocityField,
    device: torch.device,
    num_steps: int = 50,
    guidance_scale: float = 2.5,
    sampler: str = "heun",
    save_path: Optional[str] = None,
):
    """Gera e plota uma grade 6x6 contendo cada uma das 36 classes do EMNIST ('0'-'9', 'A'-'Z')."""
    vae.eval()
    flow.eval()

    labels = torch.arange(36, device=device, dtype=torch.long)
    z_1, _ = flow.sample(
        num_samples=36,
        device=device,
        labels=labels,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        sampler=sampler,
    )
    gen_imgs = vae.decode(z_1).cpu().squeeze(1).numpy()

    fig, axes = plt.subplots(6, 6, figsize=(8, 8.5))
    for idx in range(36):
        r, c = divmod(idx, 6)
        axes[r, c].imshow(gen_imgs[idx], cmap="gray", vmin=0, vmax=1)
        axes[r, c].set_title(f"'{CLASS_NAMES[idx]}' (id={idx})", fontsize=9)
        axes[r, c].axis("off")

    plt.suptitle(
        f"EMNIST 36 Classes Geradas via Latent Rectified Flow ({sampler.upper()}, CFG w={guidance_scale})",
        fontsize=12,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def plot_flow_trajectories(
    vae: VAE,
    flow: LatentVelocityField,
    device: torch.device,
    target_chars: List[str] = ["0", "2", "5", "8", "A", "G", "M", "R"],
    num_steps: int = 50,
    num_snapshots: int = 7,
    guidance_scale: float = 2.5,
    sampler: str = "heun",
    save_path: Optional[str] = None,
):
    """
    Visualiza a evolução temporal x(t) = mu_D(z(t)) de t=0 (ruído) até t=1 (imagem final),
    semelhante ao Slide 74 da Parte 6.
    """
    from .dataset import char_to_index

    vae.eval()
    flow.eval()

    label_indices = [char_to_index(ch) for ch in target_chars]
    labels = torch.tensor(label_indices, device=device, dtype=torch.long)
    num_cols = len(target_chars)

    _, trajectory = flow.sample(
        num_samples=num_cols,
        device=device,
        labels=labels,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        sampler=sampler,
        return_trajectory=True,
    )

    step_indices = torch.linspace(0, num_steps, num_snapshots).long().tolist()
    fig, axes = plt.subplots(num_snapshots, num_cols, figsize=(num_cols * 1.4, num_snapshots * 1.35))

    for row_idx, s_idx in enumerate(step_indices):
        t_val = s_idx / num_steps
        z_t = trajectory[s_idx].to(device)
        imgs_t = vae.decode(z_t).cpu().squeeze(1).numpy()

        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            ax.imshow(imgs_t[col_idx], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f" Classe '{target_chars[col_idx]}'", fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"t={t_val:.2f}", fontsize=9, fontweight="bold")

    plt.suptitle("Evolução do Fluxo Latente: x(t) = μ_D(z(t)) de t=0.00 até t=1.00", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
