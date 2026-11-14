"""
Base do trabalho: treina o VAE no EMNIST-36 e exporta o que o resto do grupo precisa.

Uso (Colab com GPU):
    python train_vae.py                      # treino completo (padrão: m=16, beta=0.25, 30 épocas)
    python train_vae.py --beta 1.0           # fórmula exata do enunciado (latente mais gaussiano, imagens mais borradas)
    python train_vae.py --fake --epochs 2    # teste rápido sem baixar o EMNIST

Saídas em --out-dir (padrão ./outputs):
    vae.pt        -> pesos + config do VAE           (carregar com src.vae.load_vae)
    latents.pt    -> mu_E(x), sigma_E(x) e rótulos de treino/teste (entrada do Flow)
    metrics.json  -> hiperparâmetros e curvas de loss
    *.png         -> figuras de validação (reconstrução, amostras, PCA, interpolação, curvas)
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import torch  # noqa: E402

from src.dataset import CLASS_NAMES, get_dataloaders  # noqa: E402
from src.vae import VAE, encode_dataset, save_vae  # noqa: E402
from src import vis_vae  # noqa: E402


@torch.no_grad()
def evaluate(vae: VAE, loader, device, beta: float):
    vae.eval()
    tot = {"loss": 0.0, "recon_error": 0.0, "latent_error": 0.0}
    n = 0
    for x, _ in loader:
        x = x.to(device)
        out = vae.compute_loss(x, beta=beta)
        for k in tot:
            tot[k] += float(out[k]) * len(x)
        n += len(x)
    return {k: v / n for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser(description="Treino do VAE no EMNIST (36 classes)")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./outputs")
    p.add_argument("--latent-dim", type=int, default=16, help="dimensão m do espaço latente")
    p.add_argument("--scalar-sigma", action="store_true", help="sigma_E escalar (fórmula com m* do enunciado)")
    p.add_argument("--beta", type=float, default=0.25,
                   help="peso de L(x). 1.0 = fórmula exata do enunciado; <1 = reconstruções mais nítidas")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fake", action="store_true", help="usa dados falsos (teste do código sem internet)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, name)  # noqa: E731
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # 1) Dados
    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers, fake=args.fake)
    print(f"Treino: {len(train_loader.dataset)} imagens | Teste: {len(test_loader.dataset)} imagens")
    x_vis, y_vis = next(iter(test_loader))
    vis_vae.plot_dataset_grid(x_vis, y_vis, save_path=out("dataset_grid.png"), show=False)

    # 2) Modelo
    vae = VAE(args.latent_dim, args.scalar_sigma).to(device)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"VAE: m={args.latent_dim}, sigma {'escalar' if args.scalar_sigma else 'diagonal'}, "
          f"beta={args.beta}, {n_params/1e6:.2f}M parâmetros")
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # 3) Treino
    history = {f"{s}_{k}": [] for s in ("train", "test") for k in ("loss", "recon_error", "latent_error")}
    best_test = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        vae.train()
        tot = {"loss": 0.0, "recon_error": 0.0, "latent_error": 0.0}
        n = 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            losses = vae.compute_loss(x, beta=args.beta)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            for k in tot:
                tot[k] += losses[k].item() * len(x)
            n += len(x)
        sched.step()
        train_m = {k: v / n for k, v in tot.items()}
        test_m = evaluate(vae, test_loader, device, args.beta)
        for k in tot:
            history[f"train_{k}"].append(train_m[k])
            history[f"test_{k}"].append(test_m[k])
        print(f"[{epoch:02d}/{args.epochs}] treino: loss {train_m['loss']:.2f} "
              f"(R {train_m['recon_error']:.2f}, L {train_m['latent_error']:.2f}) | "
              f"teste: loss {test_m['loss']:.2f} (R {test_m['recon_error']:.2f}, L {test_m['latent_error']:.2f}) "
              f"| {time.time() - t0:.0f}s")
        if test_m["loss"] < best_test:
            best_test = test_m["loss"]
            save_vae(vae, out("vae.pt"), epoch=epoch, test_metrics=test_m, beta=args.beta)

    # Recarrega o melhor checkpoint (menor loss de teste)
    ckpt = torch.load(out("vae.pt"), map_location=device)
    vae.load_state_dict(ckpt["state_dict"])
    print(f"Melhor época: {ckpt['epoch']} (loss de teste {best_test:.2f}) -> {out('vae.pt')}")

    # 4) Exporta os latentes para o Flow
    # (loaders sem embaralhar para os índices baterem com o dataset)
    plain_train = torch.utils.data.DataLoader(train_loader.dataset, batch_size=512, shuffle=False,
                                              num_workers=args.num_workers)
    lat_train = encode_dataset(vae, plain_train, device)
    lat_test = encode_dataset(vae, test_loader, device)
    mu = lat_train["mu"]
    # dimensões "ativas": variância de mu_E(x) entre imagens > 0.01 (as outras o VAE não usa)
    active_dims = int((mu.var(0) > 0.01).sum())
    torch.save(
        {
            "train": lat_train,
            "test": lat_test,
            "latent_dim": args.latent_dim,
            "class_names": CLASS_NAMES,
            "mu_mean": mu.mean(0),  # estatísticas para normalizar o latente no Flow, se quiserem
            "mu_std": mu.std(0),
        },
        out("latents.pt"),
    )
    print(f"Latentes salvos em {out('latents.pt')}: treino {tuple(mu.shape)}, teste {tuple(lat_test['mu'].shape)}")
    print(f"Estatísticas de mu_E(x): média {mu.mean():.3f}, desvio {mu.std():.3f}, "
          f"dimensões ativas {active_dims}/{args.latent_dim}")

    # 5) Figuras
    vis_vae.plot_loss_curves(history, save_path=out("vae_loss_curves.png"), show=False)
    vis_vae.plot_reconstructions(vae, x_vis, y_vis, save_path=out("vae_reconstructions.png"), show=False)
    vis_vae.plot_prior_samples(vae, save_path=out("vae_prior_samples.png"), show=False)
    vis_vae.plot_interpolation(vae, x_vis[0], x_vis[1], save_path=out("vae_interpolation.png"), show=False)
    vis_vae.plot_latent_pca(lat_test["mu"], lat_test["y"], save_path=out("vae_latent_pca.png"), show=False)

    with open(out("metrics.json"), "w") as f:
        json.dump({"args": vars(args), "best_epoch": ckpt["epoch"], "best_test": ckpt["test_metrics"],
                   "active_dims": active_dims, "n_params": n_params, "history": history}, f, indent=2)
    print(f"Figuras e métricas em {args.out_dir}/")


if __name__ == "__main__":
    main()
