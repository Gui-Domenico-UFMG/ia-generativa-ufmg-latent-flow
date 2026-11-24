"""
VAE da Parte 5: encoder probabilístico (mu_E, sigma_E) e decoder mu_D.

    Encoder:  Z(x) = mu_E(x) + sigma_E(x) * W_E,   W_E ~ N(0, I)
    Decoder:  x_hat = mu_D(z)

    Objetivo (slides 55-59):  min  sum_i  R(x_i) + beta * L(x_i)
        R(x) = E_W [ ||x - mu_D(mu_E(x) + sigma_E(x) W)||^2 / 2 ]
        L(x) = m * ((sigma_E^2 - 1)/2 - ln sigma_E) + ||mu_E(x)||^2 / 2        (sigma_E escalar)
             = sum_j ((sigma_j^2 - 1)/2 - ln sigma_j) + ||mu_E(x)||^2 / 2    (sigma_E diagonal)

    beta = 1 é exatamente a fórmula do enunciado. Com R(x) = ||x - x_hat||^2 / 2 em pixels [0, 1], o termo R
    fica pequeno (~10) perto de L, e beta = 1 tende a "desligar" dimensões latentes e borrar as imagens.
    beta < 1 (padrão do treino: 0.25) dá reconstruções mais nítidas; o Flow corrige depois a distribuição
    do latente, então não precisamos de um latente perfeitamente N(0, I).

Interface usada pelo resto do grupo:
    vae = load_vae("outputs/vae.pt", device)
    mu, sigma = vae.encode(x)       # x: (B, 1, 28, 28) em [0, 1]
    x_hat = vae.decode(z)           # z: (B, m)
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    """(B, 1, 28, 28) -> mu_E (B, m), log sigma_E (B, m) ou (B, 1)."""

    def __init__(self, latent_dim: int = 16, scalar_sigma: bool = False):
        super().__init__()
        # (1, 28, 28) -> (32, 14, 14) -> (64, 7, 7) -> (128, 4, 4)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(128 * 4 * 4, 256), nn.SiLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_sigma = nn.Linear(256, 1 if scalar_sigma else latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(self.conv(x))
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h).clamp(-10.0, 5.0)  # estabilidade numérica
        return mu, log_sigma


class Decoder(nn.Module):
    """(B, m) -> imagem (B, 1, 28, 28) em [0, 1]."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.SiLU(),
            nn.Linear(256, 128 * 7 * 7), nn.SiLU(),
        )
        # (128, 7, 7) -> (64, 14, 14) -> (32, 28, 28) -> (1, 28, 28)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.deconv(self.fc(z).view(-1, 128, 7, 7))


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 16, scalar_sigma: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.scalar_sigma = scalar_sigma
        self.encoder = Encoder(latent_dim, scalar_sigma)
        self.decoder = Decoder(latent_dim)

    @property
    def config(self) -> Dict:
        return {"latent_dim": self.latent_dim, "scalar_sigma": self.scalar_sigma}

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna (mu_E(x), sigma_E(x)). sigma_E tem shape (B, m) ou (B, 1)."""
        mu, log_sigma = self.encoder(x)
        return mu, log_sigma.exp()

    @staticmethod
    def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Z(x) = mu_E(x) + sigma_E(x) * W,  W ~ N(0, I)."""
        return mu + sigma * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """x_hat = mu_D(z)."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        return self.decode(z), mu, sigma, z

    def compute_loss(self, x: torch.Tensor, beta: float = 1.0, num_mc_samples: int = 1) -> Dict[str, torch.Tensor]:
        """R(x) + beta * L(x), média no batch. Ver docstring do módulo."""
        mu, log_sigma = self.encoder(x)
        sigma = log_sigma.exp()

        # R(x): Monte Carlo com K = num_mc_samples amostras de W
        recon = 0.0
        for _ in range(num_mc_samples):
            x_hat = self.decode(self.reparameterize(mu, sigma))
            recon = recon + 0.5 * ((x - x_hat) ** 2).sum(dim=(1, 2, 3))
        recon = (recon / num_mc_samples).mean()

        # L(x): fórmula fechada (KL entre N(mu, sigma^2 I) e N(0, I))
        per_dim = 0.5 * (sigma ** 2 - 1.0) - log_sigma
        if self.scalar_sigma:
            sigma_term = self.latent_dim * per_dim.squeeze(-1)
        else:
            sigma_term = per_dim.sum(dim=1)
        latent = (sigma_term + 0.5 * (mu ** 2).sum(dim=1)).mean()

        return {"loss": recon + beta * latent, "recon_error": recon.detach(), "latent_error": latent.detach()}


# --------------------------------------------------------------------------------------
# Salvar / carregar
# --------------------------------------------------------------------------------------
def save_vae(vae: VAE, path: str, **extra) -> None:
    torch.save({"config": vae.config, "state_dict": vae.state_dict(), **extra}, path)


def load_vae(path: str, device="cpu") -> VAE:
    ckpt = torch.load(path, map_location=device)
    vae = VAE(**ckpt["config"]).to(device)
    vae.load_state_dict(ckpt["state_dict"])
    vae.eval()
    return vae


@torch.no_grad()
def encode_dataset(vae: VAE, loader: DataLoader, device) -> Dict[str, torch.Tensor]:
    """Codifica um dataset inteiro: retorna {'mu': (N, m), 'sigma': (N, m|1), 'y': (N,)} na CPU."""
    vae.eval()
    mus, sigmas, ys = [], [], []
    for x, y in loader:
        mu, sigma = vae.encode(x.to(device))
        mus.append(mu.cpu())
        sigmas.append(sigma.cpu())
        ys.append(y.cpu())
    return {"mu": torch.cat(mus), "sigma": torch.cat(sigmas), "y": torch.cat(ys)}
