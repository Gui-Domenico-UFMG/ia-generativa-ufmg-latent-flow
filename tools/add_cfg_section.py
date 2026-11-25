"""Insere a Seção 9 (bônus: flow condicional + CFG) no notebook final.

Uso (na raiz do repo):  python tools/add_cfg_section.py
Alvo: notebooks/final_task_refactored.ipynb. O script:
  - coloca o badge do Colab no título;
  - acrescenta CFG_DIR e TRAIN_CFG_FLOW na célula de setup (Seção 0);
  - insere as células de tools/cfg_cells.py como Seção 9, antes de "Try it yourself";
  - renumera "Try it yourself" (10) e "Summary of findings" (11), e atualiza índice, overview e resumo.
É idempotente: rodar de novo substitui a Seção 9 pela versão atual de tools/cfg_cells.py.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cfg_cells import CELLS  # noqa: E402

NB = os.path.join(os.path.dirname(__file__), "..", "notebooks", "final_task_refactored.ipynb")
MARK = "# 9. Bonus (+5)"
BADGE = ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         "(https://colab.research.google.com/github/Gui-Domenico-UFMG/ia-generativa-ufmg-latent-flow/"
         "blob/main/notebooks/final_task_refactored.ipynb)")


def lines(src):
    parts = src.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def make(kind, src):
    cell = {"cell_type": kind, "metadata": {}, "source": lines(src)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def text(cell):
    return "".join(cell["source"])


def find(pred):
    return next(i for i, c in enumerate(cells) if pred(c))


def edit(i, old, new):
    """Substitui old por new na célula i (nada a fazer se new já estiver lá)."""
    src = text(cells[i])
    if new in src:
        return
    assert old in src, f"célula {i}: trecho não encontrado: {old[:60]!r}"
    cells[i]["source"] = lines(src.replace(old, new, 1))


nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

# título: badge do Colab
if "colab-badge" not in text(cells[0]):
    cells[0]["source"] = lines(BADGE + "\n\n" + text(cells[0]))

# overview: menciona o bônus
ov = find(lambda c: text(c).startswith("## Overview"))
edit(ov, "This is the same recipe used by large-scale latent diffusion models",
     "**Bonus (Section 9):** a second, *class-conditional* flow $\\Phi(z, t, c)$ on the unconditional VAE latents, "
     "sampled with **Classifier-Free Guidance**, so the class is chosen by the flow itself.\n\n"
     "This is the same recipe used by large-scale latent diffusion models")

# índice
toc = find(lambda c: text(c).startswith("### Table of contents"))
edit(toc, "9. **Interactive demo**\n10. **Summary of findings**",
     "9. **Bonus: conditional flow + Classifier-Free Guidance** — $\\Phi(z, t, c)$, guidance scale, classifier-based evaluation\n"
     "10. **Interactive demo**\n11. **Summary of findings**")

# setup: pasta e chave do bônus
setup = find(lambda c: c["cell_type"] == "code" and "TRAIN_FLOW = True" in text(c))
edit(setup, 'FLOW_DIR = os.path.join(BASE_DIR, "flow")     # rectified flow\n',
     'FLOW_DIR = os.path.join(BASE_DIR, "flow")     # rectified flow\n'
     'CFG_DIR = os.path.join(BASE_DIR, "cfg_flow")  # bonus: class-conditional flow + CFG (Section 9)\n')
edit(setup, "for d in (VAE_DIR, CVAE_DIR, FLOW_DIR):", "for d in (VAE_DIR, CVAE_DIR, FLOW_DIR, CFG_DIR):")
edit(setup, "TRAIN_FLOW = True\n", "TRAIN_FLOW = True\nTRAIN_CFG_FLOW = True\n")

# renumera as seções finais
demo = find(lambda c: "Try it yourself" in text(c) and text(c).lstrip("-\n ").startswith("# "))
edit(demo, "# 9. Try it yourself", "# 10. Try it yourself")
summ = find(lambda c: "Summary of findings" in text(c) and c["cell_type"] == "markdown" and "# 1" in text(c)
            and not text(c).startswith("### Table"))
edit(summ, "# 10. Summary of findings", "# 11. Summary of findings")
edit(summ, "\n\n**Possible extensions:**",
     "\n6. **Classifier-Free Guidance moves class control into the flow** (bonus). On the *unconditional* VAE latents, "
     "$\\Phi(z, t, c)$ with label dropout reaches **96.6%** class accuracy at $w = 3$ (chance-level at $w = 0$, 87.8% at "
     "$w = 1$), measured by a CNN trained on real EMNIST; larger $w$ trades diversity for fidelity (Section 9)."
     "\n\n**Possible extensions:**")
edit(summ, "a class-conditional flow (with classifier-free guidance); quantitative sample metrics (e.g. accuracy of a "
           "pretrained EMNIST classifier on the generated samples, or FID)",
     "FID or other distribution-level sample metrics; CFG on top of the CVAE latents")

# Seção 9: remove a versão anterior (se houver) e insere antes de "Try it yourself"
start = next((i for i, c in enumerate(cells) if MARK in text(c)), None)
if start is not None:
    del cells[start:find(lambda c: "# 10. Try it yourself" in text(c))]
pos = find(lambda c: "# 10. Try it yourself" in text(c))
cells[pos:pos] = [make(k, s) for k, s in CELLS]

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)  # mesmo formato que o Colab salva
print(f"OK: {len(cells)} células; Seção 9 em {pos}..{pos + len(CELLS) - 1}")
