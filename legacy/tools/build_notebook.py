"""
Gera notebooks/01_base_dados_vae.ipynb a partir dos arquivos de src/ e train_vae.py.

O notebook é autossuficiente: grava o código com %%writefile, então roda no Colab
sem precisar clonar o repositório. Sempre que mudar src/ ou train_vae.py, rode de novo:
    python tools/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
FILES = ["src/__init__.py", "src/dataset.py", "src/vae.py", "src/vis_vae.py", "train_vae.py"]

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        "# Parte 1 — Base: dados EMNIST-36 + VAE\n"
        "**Trabalho Final — IA Generativa (UFMG 2026/2)**\n\n"
        "Este notebook:\n"
        "1. carrega o EMNIST com as **36 classes** (`0`–`9`, `A`–`Z`);\n"
        "2. treina o **VAE** ($\\mu_E, \\sigma_E, \\mu_D$) com a loss $R(x) + \\beta L(x)$ da Parte 5;\n"
        "3. valida o VAE (reconstruções, amostras, espaço latente);\n"
        "4. exporta **`vae.pt`** e **`latents.pt`**, que são a entrada do Flow ($\\Phi$).\n\n"
        "➡️ No Colab: *Ambiente de execução → Alterar tipo → GPU (T4)*. O treino completo leva poucos minutos."
    ),
    code("!nvidia-smi -L || echo 'Sem GPU: vai funcionar, mas bem mais devagar'"),
    md("## 1. Código da base\nAs células abaixo gravam os módulos (`src/`) e o script de treino. "
       "São cópias exatas dos arquivos do repositório."),
    code("!mkdir -p src outputs"),
]
# %%writefile precisa ser a primeira linha da célula
cells += [code(f"%%writefile {f}\n{(ROOT / f).read_text()}") for f in FILES]

cells += [
    md("## 2. Configuração\n"
       "- `LATENT_DIM` ($m$): dimensão do latente. 16 é um bom ponto de partida.\n"
       "- `BETA`: peso de $L(x)$. **1.0 = fórmula exata do enunciado**; valores menores dão imagens mais nítidas.\n"
       "- `FAKE = True` usa dados falsos só para testar o código rapidamente."),
    code("LATENT_DIM = 16\nBETA = 0.25\nEPOCHS = 30\nFAKE = False\n"
         "FLAGS = f\"--latent-dim {LATENT_DIM} --beta {BETA} --epochs {EPOCHS}\" + (\" --fake --num-workers 0\" if FAKE else \"\")\n"
         "print(FLAGS)"),
    md("## 3. Conferir os dados"),
    code(
        "import torch\n"
        "from src.dataset import get_dataloaders, CLASS_NAMES\n"
        "from src import vis_vae\n\n"
        "train_loader, test_loader = get_dataloaders(batch_size=256, num_workers=0, fake=FAKE)\n"
        "print('treino:', len(train_loader.dataset), '| teste:', len(test_loader.dataset))\n"
        "x, y = next(iter(train_loader))\n"
        "print('x:', tuple(x.shape), 'min/max:', float(x.min()), float(x.max()), '| y:', tuple(y.shape))\n"
        "print('classes:', ''.join(CLASS_NAMES))\n"
        "vis_vae.plot_dataset_grid(x, y)"
    ),
    md("## 4. Treinar o VAE\nSalva o melhor checkpoint (menor loss de teste), os latentes e as figuras em `outputs/`."),
    code("!python train_vae.py {FLAGS}"),
    md("## 5. Resultados"),
    code(
        "from IPython.display import Image, display\n"
        "for name in ['vae_loss_curves', 'vae_reconstructions', 'vae_prior_samples', "
        "'vae_interpolation', 'vae_latent_pca']:\n"
        "    print(name)\n"
        "    display(Image(f'outputs/{name}.png'))"
    ),
    md("**O que observar:**\n"
       "- **Reconstruções** nítidas e com o caractere certo → o decoder está bom.\n"
       "- **Amostras de $\\mathcal{N}(0, I)$** (sem Flow) costumam sair borradas ou misturadas: é o *baseline* que o Flow precisa superar.\n"
       "- **PCA** com classes agrupadas → o latente tem estrutura por classe (bom para o condicionamento/CFG).\n"
       "- **Dimensões ativas** (no log do treino): se forem bem menos que $m$, o VAE está ignorando parte do latente (diminua `BETA`)."),
    md("## 6. (Opcional) Comparar valores de β\nTreinos curtos para escolher β. Útil para um slide de \"decisões de engenharia\"."),
    code(
        "RUN_SWEEP = False\n"
        "if RUN_SWEEP:\n"
        "    import json\n"
        "    for b in [1.0, 0.25, 0.05]:\n"
        "        !python train_vae.py --latent-dim {LATENT_DIM} --beta {b} --epochs 10 --out-dir outputs/beta_{b} {'--fake --num-workers 0' if FAKE else ''} > /dev/null\n"
        "        m = json.load(open(f'outputs/beta_{b}/metrics.json'))\n"
        "        print(f\"beta={b}: R={m['best_test']['recon_error']:.2f}  L={m['best_test']['latent_error']:.2f}  \"\n"
        "              f\"dimensões ativas={m['active_dims']}/{LATENT_DIM}\")\n"
        "        display(Image(f'outputs/beta_{b}/vae_prior_samples.png'))"
    ),
    md("## 7. Como o resto do grupo usa a base\n"
       "Exemplo para quem vai treinar o Flow ($\\Phi$): carregar os latentes e o decoder."),
    code(
        "from src.vae import load_vae\n"
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "vae = load_vae('outputs/vae.pt', device)\n"
        "lat = torch.load('outputs/latents.pt')\n"
        "mu, sigma, labels = lat['train']['mu'], lat['train']['sigma'], lat['train']['y']\n"
        "print('mu:', tuple(mu.shape), '| sigma:', tuple(sigma.shape), '| labels:', tuple(labels.shape))\n\n"
        "# alvo do Flow: z(1) = mu_E(x)  (ou z(1) = mu + sigma * W, amostrando do encoder)\n"
        "z1 = mu[:8].to(device)\n"
        "with torch.no_grad():\n"
        "    imgs = vae.decode(z1)   # x = mu_D(z(1))\n"
        "print('imagens decodificadas:', tuple(imgs.shape))"
    ),
    md("## 8. Salvar e compartilhar\nOpção A: baixar um `.zip`. Opção B: copiar para o Google Drive e compartilhar a pasta com o grupo."),
    code(
        "!cd outputs && zip -q -r ../base_vae_outputs.zip vae.pt latents.pt metrics.json *.png\n"
        "try:\n"
        "    from google.colab import files\n"
        "    files.download('base_vae_outputs.zip')\n"
        "except ImportError:\n"
        "    print('Fora do Colab: arquivo em base_vae_outputs.zip')"
    ),
    code(
        "SAVE_TO_DRIVE = False\n"
        "if SAVE_TO_DRIVE:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "    !mkdir -p /content/drive/MyDrive/ia-generativa-tp/base\n"
        "    !cp outputs/vae.pt outputs/latents.pt outputs/metrics.json outputs/*.png /content/drive/MyDrive/ia-generativa-tp/base/"
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}
out = ROOT / "notebooks" / "01_base_dados_vae.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"Gerado {out.relative_to(ROOT)} ({len(cells)} células)")
