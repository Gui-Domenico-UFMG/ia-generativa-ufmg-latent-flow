"""Insere no notebook final: badge do Colab, célula de setup e a seção de CFG.

Uso (na raiz do repo):  python tools/add_cfg_section.py
Idempotente: se a seção já existir, ela é substituída pela versão de tools/cfg_cells.py.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cfg_cells import CELLS  # noqa: E402

NB = os.path.join(os.path.dirname(__file__), "..", "notebooks", "final_task.ipynb")
MARK = "## Bonus (+5): Conditional Latent Flow"
SETUP_MARK = "# SETUP"
BADGE = ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         "(https://colab.research.google.com/github/Gui-Domenico-UFMG/ia-generativa-ufmg-latent-flow/"
         "blob/main/notebooks/final_task.ipynb)")
SETUP = """# SETUP: output folders used by the cells below (run this first)
import os
for d in ("data", "outputs", "cvae", "flow", "cfg_flow"):
    os.makedirs(d, exist_ok=True)

import torch
print("PyTorch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")"""


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


nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

# 1) badge do Colab no título
if "colab-badge" not in text(cells[0]):
    cells[0]["source"] = lines(BADGE + "\n\n" + text(cells[0]))

# 1b) introdução: menciona o bônus
INTRO_ITEM = ("4.   **Bonus — Classifier-Free Guidance:** a class-conditional velocity field $\\Phi(z, t, c)$ trained "
              "with label dropout on the unconditional VAE latents, sampled with CFG.")
if "Classifier-Free Guidance" not in text(cells[1]):
    src = text(cells[1])
    anchor = "class-specific character generation."
    cells[1]["source"] = lines(src.replace(anchor, anchor + "\n" + INTRO_ITEM, 1))

# 2) célula de setup logo após a introdução
if not any(c["cell_type"] == "code" and text(c).startswith(SETUP_MARK) for c in cells):
    cells.insert(2, make("code", SETUP))

# 3) seção de CFG (remove versão anterior, se houver) antes de "Try it yourself"
start = next((i for i, c in enumerate(cells) if text(c).strip().startswith(MARK)), None)
if start is not None:
    end = start + len(CELLS)
    del cells[start:end]
    pos = start
else:
    pos = next(i for i, c in enumerate(cells) if "Try it yourself" in text(c) and c["cell_type"] == "markdown")
cells[pos:pos] = [make(k, s) for k, s in CELLS]

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)  # mesmo formato que o Colab salva
print(f"OK: {len(cells)} células; seção CFG em {pos}..{pos + len(CELLS) - 1}")
