from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Rede neural do Encoder probabilístico do VAE (Part 5):
    Parametriza as funções determinísticas mu_E(x) e sigma_E(x) tais que:
        E(x) = mu_E(x) + sigma_E(x) * W_E,  com W_E ~ N(0, I).
    """

    def __init__(self, latent_dim: int = 16, scalar_sigma: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.scalar_sigma = scalar_sigma

        # Convoluções: (1, 28, 28) -> (32, 14, 14) -> (64, 7, 7) -> (128, 4, 4)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.fc_hidden = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.SiLU(),
        )

        # Cabeça para mu_E: R^n -> R^m
        self.fc_mu = nn.Linear(256, latent_dim)

        # Cabeça para log(sigma_E): R^n -> R (se scalar_sigma=True, Eq. 29 dos slides) ou R^m
        sigma_out_dim = 1 if scalar_sigma else latent_dim
        self.fc_log_sigma = nn.Linear(256, sigma_out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc_hidden(self.conv(x))
        mu_e = self.fc_mu(h)
        # Limitamos log_sigma para estabilidade numérica antes da exponencial
        log_sigma_e = torch.clamp(self.fc_log_sigma(h), min=-10.0, max=5.0)
        sigma_e = torch.exp(log_sigma_e)
        return mu_e, sigma_e


class Decoder(nn.Module):
    """
    Rede neural do Decoder do VAE (Part 5):
    Parametriza a função mu_D(z): R^m -> R^n (reconstruindo imagens 1x28x28).
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128 * 7 * 7),
            nn.SiLU(),
        )

        # Deconvoluções: (128, 7, 7) -> (64, 14, 14) -> (32, 28, 28) -> (1, 28, 28)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 128, 7, 7)
        x_hat = self.deconv(h)
        return x_hat


class VAE(nn.Module):
    """
    Variational Autoencoder completo (mu_E, sigma_E, mu_D) seguindo a formulação da Parte 5.
    """

    def __init__(self, latent_dim: int = 16, scalar_sigma: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.scalar_sigma = scalar_sigma
        self.encoder = Encoder(latent_dim=latent_dim, scalar_sigma=scalar_sigma)
        self.decoder = Decoder(latent_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna (mu_E(x), sigma_E(x))."""
        return self.encoder(x)

    def reparameterize(self, mu_e: torch.Tensor, sigma_e: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick (Slide 54 - Part 5):
            Z(x) = mu_E(x) + sigma_E(x) * W_E,  onde W_E ~ N(0, I).
        """
        w_e = torch.randn_like(mu_e)
        return mu_e + sigma_e * w_e

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Calcula x_hat = mu_D(z)."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_e, sigma_e = self.encode(x)
        z = self.reparameterize(mu_e, sigma_e)
        x_hat = self.decode(z)
        return x_hat, mu_e, sigma_e, z

    def compute_loss(
        self,
        x: torch.Tensor,
        beta: float = 0.25,
        num_mc_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula a função objetivo do VAE (Slides 55-59 da Parte 5):
            min  R(x; mu_D, mu_E, sigma_E) + beta * L(x; mu_D, mu_E, sigma_E)

        Onde:
        - R(x) = E_W [ ||x - mu_D(mu_E(x) + sigma_E(x) W)||^2 / 2 ]  (Eq. 43 / 46)
        - L(x) = m * ( (sigma_E(x)^2 - 1)/2 - ln(sigma_E(x)) ) + ||mu_E(x)||^2 / 2  (Eq. 47)
        """
        mu_e, sigma_e = self.encode(x)

        # Aproximação de Monte Carlo para R(x) (Eq. 46, K = num_mc_samples)
        recon_error = 0.0
        for _ in range(num_mc_samples):
            z = self.reparameterize(mu_e, sigma_e)
            x_hat = self.decode(z)
            # ||x - mu_D(z)||^2 / 2 por amostra no batch
            diff_sq = torch.sum((x - x_hat) ** 2, dim=(1, 2, 3)) * 0.5
            recon_error = recon_error + diff_sq.mean()
        recon_error = recon_error / num_mc_samples

        # Erro de distribuição latente L(x) (Eq. 47)
        if self.scalar_sigma:
            # sigma_e tem shape (B, 1); multiplicamos pela dimensão latente m
            sigma_term = self.latent_dim * (
                0.5 * (sigma_e.squeeze(-1) ** 2 - 1.0) - torch.log(sigma_e.squeeze(-1) + 1e-8)
            )
        else:
            # Caso geral diagonal: soma sobre as m coordenadas latentes
            sigma_term = torch.sum(
                0.5 * (sigma_e ** 2 - 1.0) - torch.log(sigma_e + 1e-8), dim=1
            )

        mu_term = 0.5 * torch.sum(mu_e ** 2, dim=1)
        latent_error = (sigma_term + mu_term).mean()

        total_loss = recon_error + beta * latent_error
        return {
            "loss": total_loss,
            "recon_error": recon_error.detach(),
            "latent_error": latent_error.detach(),
        }
