import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Codificação posicional senoidal para o tempo contínuo t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t[:, None].float() * freqs[None, :] * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaLNZeroBlock(nn.Module):
    """
    Bloco Residual MLP com Adaptive Layer Normalization (AdaLN) condicionado em (t, classe).
    Permite que o campo de velocidade Phi(z, t, c) module escala e deslocamento em cada camada.
    """

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Projeta o vetor de condicionamento (tempo + classe) em (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN_modulation(cond).chunk(6, dim=-1)

        h = self.norm1(x) * (1.0 + gamma1) + beta1
        h = F.silu(h)
        h = self.fc1(h)

        h = self.norm2(h) * (1.0 + gamma2) + beta2
        h = F.silu(h)
        h = self.dropout(h)
        h = self.fc2(h)

        return x + alpha1 * h + 0.1 * alpha2 * h


class LatentVelocityField(nn.Module):
    """
    Rede Neural do Campo de Velocidade Phi(z, t, c) para Rectified Flow no espaço latente (Part 6 & 7).
    Suporta geração incondicional e condicional (36 classes do EMNIST) com Classifier-Free Guidance (CFG).
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        num_blocks: int = 4,
        num_classes: int = 36,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # Índice reservado para o token nulo (incondicional) usado no Classifier-Free Guidance
        self.null_class_idx = num_classes

        # Embedding de tempo t in [0, 1]
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Embedding de classe (0..35 para EMNIST + 36 para incondicional/nulo)
        self.class_emb = nn.Embedding(num_classes + 1, cond_dim)

        # Projeção de entrada do espaço latente R^m -> hidden_dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Blocos residuais modulados por tempo e classe
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(hidden_dim=hidden_dim, cond_dim=cond_dim) for _ in range(num_blocks)]
        )

        # Camada final de saída hidden_dim -> R^m
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcula a velocidade predita v = Phi(z, t, c).
        - z: (B, latent_dim)
        - t: (B,) ou (B, 1) em [0, 1]
        - labels: (B,) inteiros em [0, num_classes] (se None, usa null_class_idx)
        """
        batch_size = z.size(0)
        if labels is None:
            labels = torch.full(
                (batch_size,), self.null_class_idx, device=z.device, dtype=torch.long
            )

        t_emb = self.time_mlp(t)
        c_emb = self.class_emb(labels)
        cond = t_emb + c_emb

        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h, cond)

        h = F.silu(self.final_norm(h))
        v = self.output_proj(h)
        return v

    def compute_flow_matching_loss(
        self,
        z_1: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        p_uncond: float = 0.15,
    ) -> torch.Tensor:
        """
        Calcula a perda de Rectified Flow / Flow Matching (Part 6, Slides 52 e 72):
        Parametrização Forward (t de 0 até 1):
            z(0) ~ N(0, I)  (ruído inicial no espaço latente)
            z(1) = z_1      (vetor latente da imagem real codificada pelo VAE)
            z(t) = (1 - t) * z(0) + t * z(1)
            v_target = dz/dt = z(1) - z(0)

        Para permitir Classifier-Free Guidance (Part 7), descartamos o rótulo da classe
        com probabilidade `p_uncond`, substituindo-o por `null_class_idx`.
        """
        batch_size = z_1.size(0)
        device = z_1.device

        # Amostra ruído inicial z(0) ~ N(0, I) e tempo t ~ U(0, 1)
        z_0 = torch.randn_like(z_1)
        t = torch.rand(batch_size, 1, device=device)

        # Interpolação retificada entre z(0) e z(1)
        z_t = (1.0 - t) * z_0 + t * z_1
        v_target = z_1 - z_0

        # Dropout de condicionamento para Classifier-Free Guidance
        if labels is not None:
            train_labels = labels.clone()
            if p_uncond > 0.0:
                drop_mask = torch.rand(batch_size, device=device) < p_uncond
                train_labels[drop_mask] = self.null_class_idx
        else:
            train_labels = None

        v_pred = self.forward(z_t, t.squeeze(-1), train_labels)
        loss = F.mse_loss(v_pred, v_target)
        return loss

    @torch.no_grad()
    def guided_velocity(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 2.0,
        neg_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Avalia o campo de velocidade com Classifier-Free Guidance (CFG) ou Negative Guidance (Part 7):
            v_CFG = v_uncond + w * (v_cond - v_uncond)
        Se `neg_labels` for fornecido (Negative Guidance):
            v_CFG = v_neg + w * (v_cond - v_neg)
        """
        if labels is None or guidance_scale == 0.0:
            return self.forward(z, t, labels=None)

        if guidance_scale == 1.0 and neg_labels is None:
            return self.forward(z, t, labels=labels)

        # Concatena batch condicional e incondicional/negativo para avaliar em uma única passada
        batch_size = z.size(0)
        if neg_labels is None:
            uncond_labels = torch.full(
                (batch_size,), self.null_class_idx, device=z.device, dtype=torch.long
            )
        else:
            uncond_labels = neg_labels

        z_double = torch.cat([z, z], dim=0)
        t_double = torch.cat([t, t], dim=0)
        labels_double = torch.cat([labels, uncond_labels], dim=0)

        v_double = self.forward(z_double, t_double, labels=labels_double)
        v_cond, v_uncond = v_double.chunk(2, dim=0)

        return v_uncond + guidance_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        labels: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        guidance_scale: float = 2.0,
        sampler: str = "heun",
        neg_labels: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        z_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Gera vetores latentes finais z(1) integrando a EDO dz/dt = Phi(z, t) de t=0 até t=1
        a partir de z(0) ~ N(0, I).

        Suporta os dois Samplers vistos na Parte 6 (Slides 67-68):
        - 'euler': First-Order Explicit Euler
        - 'heun': Método de Heun (Runge-Kutta de 2ª ordem)
        """
        self.eval()
        if z_0 is None:
            z = torch.randn(num_samples, self.latent_dim, device=device)
        else:
            z = z_0.clone().to(device)

        dt = 1.0 / num_steps
        trajectory = [z.detach().cpu()] if return_trajectory else None

        for step in range(num_steps):
            t_curr = torch.full((num_samples,), step * dt, device=device, dtype=torch.float32)

            if sampler.lower() == "euler":
                v = self.guided_velocity(
                    z, t_curr, labels=labels, guidance_scale=guidance_scale, neg_labels=neg_labels
                )
                z = z + v * dt

            elif sampler.lower() == "heun":
                v_a = self.guided_velocity(
                    z, t_curr, labels=labels, guidance_scale=guidance_scale, neg_labels=neg_labels
                )
                z_euler = z + v_a * dt
                t_next = torch.full(
                    (num_samples,), (step + 1) * dt, device=device, dtype=torch.float32
                )
                v_b = self.guided_velocity(
                    z_euler,
                    t_next,
                    labels=labels,
                    guidance_scale=guidance_scale,
                    neg_labels=neg_labels,
                )
                z = z + 0.5 * dt * (v_a + v_b)
            else:
                raise ValueError(f"Sampler desconhecido: '{sampler}'. Use 'euler' ou 'heun'.")

            if return_trajectory:
                trajectory.append(z.detach().cpu())

        return z, trajectory
