import os
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.dataset import get_dataloaders, NUM_CLASSES
from src.vae import VAE
from src.flow import LatentVelocityField
from src.vis import plot_vae_reconstructions, plot_all_36_classes, plot_flow_trajectories


@torch.no_grad()
def encode_dataset_to_latent(vae: VAE, dataloader: DataLoader, device: torch.device) -> DataLoader:
    """
    Codifica todas as imagens do dataset usando o Encoder treinado do VAE (mu_E(x), sigma_E(x)),
    acelerando drasticamente o treinamento do Rectified Flow no espaço latente (Slide 70 - Part 6).
    """
    vae.eval()
    all_mu = []
    all_sigma = []
    all_labels = []

    for x, y in tqdm(dataloader, desc="Codificando dataset para o espaço latente"):
        x = x.to(device)
        mu_e, sigma_e = vae.encode(x)
        all_mu.append(mu_e.cpu())
        all_sigma.append(sigma_e.cpu())
        all_labels.append(y.cpu())

    mu_tensor = torch.cat(all_mu, dim=0)
    sigma_tensor = torch.cat(all_sigma, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    latent_ds = TensorDataset(mu_tensor, sigma_tensor, labels_tensor)
    return DataLoader(latent_ds, batch_size=dataloader.batch_size, shuffle=True, drop_last=True)


def main():
    parser = argparse.ArgumentParser(description="Treinamento VAE + Latent Rectified Flow no EMNIST (36 classes)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./outputs")
    parser.add_argument("--latent-dim", type=int, default=16, help="Dimensão m do espaço latente")
    parser.add_argument("--scalar-sigma", action="store_true", help="Usar sigma_E escalar (Eq. 29/47 da Parte 5)")
    parser.add_argument("--beta", type=float, default=0.25, help="Peso do termo de regularização latente L(x)")
    parser.add_argument("--vae-epochs", type=int, default=15)
    parser.add_argument("--flow-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cfg-scale", type=float, default=2.5, help="Escala w do Classifier-Free Guidance")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo selecionado: {device}")

    # 1. Carregar EMNIST (36 classes: 0-9 e A-Z)
    train_loader, test_loader = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)

    # 2. Instanciar e treinar o VAE (mu_E, sigma_E, mu_D)
    vae = VAE(latent_dim=args.latent_dim, scalar_sigma=args.scalar_sigma).to(device)
    opt_vae = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=1e-4)

    print("\n=== Etapa 1: Treinando o Variational Autoencoder (VAE) ===")
    for epoch in range(1, args.vae_epochs + 1):
        vae.train()
        total_loss, total_r, total_l = 0.0, 0.0, 0.0
        for x, _ in tqdm(train_loader, desc=f"VAE Época {epoch}/{args.vae_epochs}", leave=False):
            x = x.to(device)
            opt_vae.zero_grad()
            losses = vae.compute_loss(x, beta=args.beta)
            losses["loss"].backward()
            opt_vae.step()

            total_loss += losses["loss"].item()
            total_r += losses["recon_error"].item()
            total_l += losses["latent_error"].item()

        n_batches = len(train_loader)
        print(
            f"[VAE {epoch:02d}/{args.vae_epochs:02d}] Loss: {total_loss/n_batches:.3f} "
            f"| Recon R(x): {total_r/n_batches:.3f} | Latent L(x): {total_l/n_batches:.3f}"
        )

    torch.save(vae.state_dict(), os.path.join(args.out_dir, "vae_emnist36.pt"))

    # Visualizar reconstruções do VAE
    test_x, test_y = next(iter(test_loader))
    plot_vae_reconstructions(
        vae, test_x, test_y, save_path=os.path.join(args.out_dir, "vae_reconstructions.png")
    )

    # 3. Codificar dataset para o espaço latente e treinar o Campo de Velocidade Phi
    latent_loader = encode_dataset_to_latent(vae, train_loader, device)

    flow = LatentVelocityField(
        latent_dim=args.latent_dim,
        hidden_dim=256,
        cond_dim=128,
        num_blocks=4,
        num_classes=NUM_CLASSES,
    ).to(device)
    opt_flow = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_flow = torch.optim.lr_scheduler.CosineAnnealingLR(opt_flow, T_max=args.flow_epochs)

    print("\n=== Etapa 2: Treinando o Campo de Velocidade (Rectified Flow + CFG) ===")
    for epoch in range(1, args.flow_epochs + 1):
        flow.train()
        total_flow_loss = 0.0
        for mu_e, sigma_e, y in latent_loader:
            mu_e, sigma_e, y = mu_e.to(device), sigma_e.to(device), y.to(device)
            # Amostra z(1) no espaço latente do VAE: z(1) = mu_E(x) + sigma_E(x) * W
            z_1 = mu_e + sigma_e * torch.randn_like(mu_e)

            opt_flow.zero_grad()
            loss = flow.compute_flow_matching_loss(z_1=z_1, labels=y, p_uncond=0.15)
            loss.backward()
            opt_flow.step()
            total_flow_loss += loss.item()

        scheduler_flow.step()
        avg_loss = total_flow_loss / len(latent_loader)
        if epoch % 5 == 0 or epoch == 1 or epoch == args.flow_epochs:
            print(f"[Flow {epoch:02d}/{args.flow_epochs:02d}] Flow Matching MSE Loss: {avg_loss:.4f}")

    torch.save(flow.state_dict(), os.path.join(args.out_dir, "flow_emnist36.pt"))

    # 4. Gerar amostras das 36 classes e trajetórias z(0) -> z(1)
    plot_all_36_classes(
        vae,
        flow,
        device=device,
        num_steps=50,
        guidance_scale=args.cfg_scale,
        sampler="heun",
        save_path=os.path.join(args.out_dir, "generated_36_classes.png"),
    )
    plot_flow_trajectories(
        vae,
        flow,
        device=device,
        num_steps=50,
        guidance_scale=args.cfg_scale,
        sampler="heun",
        save_path=os.path.join(args.out_dir, "flow_trajectories.png"),
    )


if __name__ == "__main__":
    main()
