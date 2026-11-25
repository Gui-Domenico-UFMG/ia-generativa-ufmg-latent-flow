"""Células da seção "Bônus: Flow condicional + Classifier-Free Guidance" do notebook final.

Usado por `tools/add_cfg_section.py`, que insere estas células em
`notebooks/final_task_refactored.ipynb` como Seção 9 (antes de "Try it yourself").
Cada item é (tipo, fonte), com tipo "markdown" ou "code".
"""

CELLS = [
    ("markdown", r"""---
# 9. Bonus (+5): Class-conditional flow + Classifier-Free Guidance

In Section 8, class control came from the **CVAE decoder**, while the velocity field $v_\theta(x, t)$ was unconditional.
The assignment bonus (Part 7) asks for the conditioning signal **inside the flow**, $\Phi(z, t, c)$, plus
**Classifier-Free Guidance (CFG)** at sampling time.

**Setup.** We use the latents of the **unconditional VAE** (Section 4, `VAE_DIR/latents.pt`): there the class
information lives in $z$ itself (see its PCA), so the flow alone must steer each sample to the right class region,
and the plain VAE decoder $\mu_D(z)$ turns it into an image.
Note: this section follows the **course convention** (forward parametrization, noise at $t=0$), the opposite
time direction of Section 7:

- $z(0) \sim \mathcal{N}(0, I)$ (noise), $\; z(1) = \mu_E(x)$ (data, standardized per dimension)
- $z(t) = (1-t)\,z(0) + t\,z(1)$, target velocity $z(1) - z(0)$
- $c \in \{0, \dots, 35\}$, plus a **null token** $\varnothing = 36$

**Training (label dropout).** With probability $p_{\text{uncond}} = 0.15$ the label is replaced by $\varnothing$, so a single
network learns both $\Phi(z, t, c)$ and $\Phi(z, t, \varnothing)$:
$$\min_\theta \; \mathbb{E}\,\big\| \Phi(z(t), t, \tilde c;\theta) - (z(1) - z(0)) \big\|^2$$

**Sampling (CFG).** Integrate $dz/dt = v_{\text{CFG}}$ from $t=0$ to $t=1$ (Euler or Heun), with
$$v_{\text{CFG}}(z, t, c) = \Phi(z, t, \varnothing) + w \,\big(\Phi(z, t, c) - \Phi(z, t, \varnothing)\big)$$
$w = 0$ is unconditional, $w = 1$ is plain conditional, $w > 1$ pushes harder towards class $c$.
Finally $x = \mu_D(z(1))$ with the VAE decoder."""),

    ("markdown", r"""## 9.1 Conditional velocity network
Same idea as Section 7.2 (residual MLP + sinusoidal time embedding), with two changes: a **class embedding**
(37 entries: 36 classes + $\varnothing$) added to the time embedding, and **adaLN** blocks, where this
(time + class) vector produces a per-layer scale, shift and gate. The gates start at zero, so each block starts as the identity."""),

    ("code", r"""import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CondSinusoidalEmbedding(nn.Module):
    # Sinusoidal embedding for t in [0, 1]
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = 1000.0 * t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaLNBlock(nn.Module):
    # Residual MLP block modulated (scale/shift/gate) by the (time + class) embedding
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * hidden_dim))
        nn.init.zeros_(self.mod[-1].weight)  # starts as identity (adaLN-zero)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift, gate = self.mod(cond).chunk(3, dim=-1)
        u = self.norm(h) * (1 + scale) + shift
        u = self.fc2(F.silu(self.fc1(F.silu(u))))
        return h + gate * u


class ConditionalVelocityNet(nn.Module):
    # Phi(z, t, c): velocity field conditioned on the class c (36 classes + null token)
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 256, cond_dim: int = 128,
                 num_blocks: int = 4, num_classes: int = 36):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.null_class = num_classes  # index of the null token (unconditional)
        self.time_mlp = nn.Sequential(CondSinusoidalEmbedding(cond_dim), nn.Linear(cond_dim, cond_dim),
                                      nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.class_emb = nn.Embedding(num_classes + 1, cond_dim)
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([AdaLNBlock(hidden_dim, cond_dim) for _ in range(num_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim))

    def forward(self, z: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        cond = self.time_mlp(t) + self.class_emb(c)
        h = self.in_proj(z)
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h)


class CFGRectifiedFlow(nn.Module):
    # Rectified Flow in latent space (t=0 noise -> t=1 data) with Classifier-Free Guidance
    def __init__(self, net: ConditionalVelocityNet, mu_mean: torch.Tensor, mu_std: torch.Tensor):
        super().__init__()
        self.net = net
        self.latent_dim = net.latent_dim
        self.null_class = net.null_class
        # standardization of the latents (z_norm = (mu - mean) / std)
        self.register_buffer("mu_mean", mu_mean.float().clone())
        self.register_buffer("mu_std", mu_std.float().clamp_min(1e-3).clone())

    def normalize(self, mu):
        return (mu - self.mu_mean) / self.mu_std

    def denormalize(self, z):
        return z * self.mu_std + self.mu_mean

    def compute_loss(self, mu: torch.Tensor, y: torch.Tensor, p_uncond: float = 0.15) -> torch.Tensor:
        z1 = self.normalize(mu)                      # data,  t = 1
        z0 = torch.randn_like(z1)                    # noise, t = 0
        t = torch.rand(z1.shape[0], device=z1.device)
        zt = (1 - t[:, None]) * z0 + t[:, None] * z1
        drop = torch.rand(y.shape[0], device=y.device) < p_uncond
        c = torch.where(drop, torch.full_like(y, self.null_class), y)   # label dropout
        return F.mse_loss(self.net(zt, t, c), z1 - z0)

    @torch.no_grad()
    def velocity(self, z, t, c, w: float):
        # v_CFG = Phi(z,t,null) + w * (Phi(z,t,c) - Phi(z,t,null))
        null = torch.full_like(c, self.null_class)
        if w == 0.0:
            return self.net(z, t, null)
        if w == 1.0:
            return self.net(z, t, c)
        v = self.net(torch.cat([z, z]), torch.cat([t, t]), torch.cat([c, null]))
        v_cond, v_uncond = v.chunk(2)
        return v_uncond + w * (v_cond - v_uncond)

    @torch.no_grad()
    def sample(self, labels: torch.Tensor, w: float = 3.0, num_steps: int = 25, method: str = "heun",
               z0: torch.Tensor | None = None, return_trajectory: bool = False):
        # Integrates dz/dt = v_CFG from t=0 to t=1; returns mu_E-scale latents ready for the decoder
        self.net.eval()
        device = self.mu_mean.device
        c = labels.to(device).long()
        z = torch.randn(len(c), self.latent_dim, device=device) if z0 is None else z0.to(device).clone()
        traj = [z.cpu()]
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((len(c),), i * dt, device=device)
            v1 = self.velocity(z, t, c, w)
            if method == "euler":
                z = z + dt * v1
            elif method == "heun":
                z_pred = z + dt * v1
                v2 = self.velocity(z_pred, t + dt, c, w)
                z = z + 0.5 * dt * (v1 + v2)
            else:
                raise ValueError(f"Unknown method '{method}' (use 'euler' or 'heun')")
            if return_trajectory:
                traj.append(z.cpu())
        out = self.denormalize(z)
        return (out, traj) if return_trajectory else out"""),

    ("markdown", r"""## 9.2 Training
AdamW + cosine schedule, gradient clipping, and an **EMA** copy of the weights (decay 0.999), which is used for sampling.
The latents are standardized per dimension, and `sample()` undoes this before decoding.
> With `TRAIN_CFG_FLOW = False` (Section 0), the cell below loads `cfg_flow.pt` from `CFG_DIR` instead of training."""),

    ("code", r"""import copy
import time
from pathlib import Path


def train_cfg_flow(flow, train_loader, test_loader, epochs=100, lr=1e-3, p_uncond=0.15,
                   ema_decay=0.999, device="cpu"):
    # Trains Phi(z, t, c) with label dropout; keeps an EMA copy of the weights (used for sampling)
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    ema = copy.deepcopy(flow).eval()
    history = {"train_loss": [], "test_loss": []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        flow.train()
        tot, n = 0.0, 0
        for mu, y in train_loader:
            mu, y = mu.to(device), y.to(device)
            loss = flow.compute_loss(mu, y, p_uncond)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), flow.parameters()):
                    pe.lerp_(p, 1 - ema_decay)
            tot += loss.item() * len(mu)
            n += len(mu)
        # test loss (fixed seed so epochs are comparable)
        ema.eval()
        g_state = torch.random.get_rng_state()
        torch.manual_seed(0)
        with torch.no_grad():
            te, m = 0.0, 0
            for mu, y in test_loader:
                mu, y = mu.to(device), y.to(device)
                te += ema.compute_loss(mu, y, p_uncond).item() * len(mu)
                m += len(mu)
        torch.random.set_rng_state(g_state)
        history["train_loss"].append(tot / n)
        history["test_loss"].append(te / m)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"[{epoch:03d}/{epochs}] train {tot / n:.4f} | test (EMA) {te / m:.4f} | {time.time() - t0:.0f}s")
    return ema, history


def save_cfg_flow(flow, path, config, history):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": flow.state_dict(), "config": config, "history": history}, path)


def load_cfg_flow(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    net = ConditionalVelocityNet(**cfg)
    flow = CFGRectifiedFlow(net, torch.zeros(cfg["latent_dim"]), torch.ones(cfg["latent_dim"]))
    flow.load_state_dict(ckpt["state_dict"])
    return flow.to(device).eval(), ckpt"""),

    ("code", r"""# CFG FLOW PARAMETERS
cfg_config = dict(latent_dim=16, hidden_dim=256, cond_dim=128, num_blocks=4, num_classes=36)
cfg_epochs = 100        # ~5 min on CPU, faster on a T4
cfg_batch_size = 512
cfg_lr = 1e-3
p_uncond = 0.15
torch.manual_seed(42)

# Unconditional VAE (Section 4) as decoder + its latents as training data
vae = load_vae(f"{VAE_DIR}/vae.pt", device=device)
vae_latents = torch.load(f"{VAE_DIR}/latents.pt", map_location="cpu", weights_only=False)
cfg_train_loader, cfg_test_loader = get_latent_dataloaders(f"{VAE_DIR}/latents.pt", batch_size=cfg_batch_size)

if TRAIN_CFG_FLOW:
    cfg_flow = CFGRectifiedFlow(ConditionalVelocityNet(**cfg_config),
                                vae_latents["mu_mean"], vae_latents["mu_std"]).to(device)
    print(f"Phi(z, t, c): {sum(p.numel() for p in cfg_flow.parameters()) / 1e6:.2f}M parameters")
    cfg_flow, cfg_history = train_cfg_flow(cfg_flow, cfg_train_loader, cfg_test_loader, epochs=cfg_epochs,
                                           lr=cfg_lr, p_uncond=p_uncond, device=device)
    save_cfg_flow(cfg_flow, f"{CFG_DIR}/cfg_flow.pt", cfg_config, cfg_history)
else:
    cfg_flow, _ckpt = load_cfg_flow(f"{CFG_DIR}/cfg_flow.pt", device=device)
    cfg_history = _ckpt["history"]
    print(f"Loaded {CFG_DIR}/cfg_flow.pt")"""),

    ("code", r"""fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(cfg_history["train_loss"], label="train")
ax.plot(cfg_history["test_loss"], label="test (EMA)")
ax.set_xlabel("epoch")
ax.set_ylabel("flow matching MSE")
ax.set_title("Conditional velocity field $\\Phi(z,t,c)$")
ax.legend()
_finish(fig, f"{CFG_DIR}/cfg_loss_curves.png", show=True)"""),

    ("markdown", r"""## 9.3 Results

**All 36 classes with CFG.**
Each row is one class; every column starts from a different noise $z(0)$. Heun, 25 steps, $w = 3$."""),

    ("code", r"""@torch.no_grad()
def generate_cfg(labels, w=3.0, num_steps=25, method="heun", z0=None):
    z1 = cfg_flow.sample(labels, w=w, num_steps=num_steps, method=method, z0=z0)
    return vae.decode(z1).cpu()


def plot_class_grid(w=3.0, per_class=10, seed=0, save_path=None, show=True):
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(36).repeat_interleave(per_class)
    z0 = torch.randn(len(labels), cfg_config["latent_dim"], generator=g)
    imgs = generate_cfg(labels, w=w, z0=z0)
    # 36 rows x per_class columns, split in two panels of 18 classes for readability
    fig, axes = plt.subplots(18, 2 * per_class + 1, figsize=((2 * per_class + 1) * 0.55, 18 * 0.6))
    for ax in axes.flat:
        ax.axis("off")
    for cls in range(36):
        row, block = cls % 18, cls // 18
        for j in range(per_class):
            col = block * (per_class + 1) + j
            axes[row, col].imshow(imgs[cls * per_class + j, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, block * (per_class + 1)].set_title(CLASS_NAMES[cls], fontsize=8, loc="left", pad=1)
    fig.suptitle(f"Conditional latent flow + CFG (w = {w})", fontsize=11)
    _finish(fig, save_path, show)


plot_class_grid(w=3.0, save_path=f"{CFG_DIR}/cfg_all_classes_w3.png")"""),

    ("markdown", r"""**Effect of the guidance scale $w$.**
Same initial noise $z(0)$ for every column; only $w$ changes. $w=0$ ignores the label (unconditional sample),
$w=1$ is the plain conditional field, larger $w$ gives more class-typical (and less diverse) characters."""),

    ("code", r"""def plot_guidance_sweep(chars="0A7GKR", ws=(0.0, 1.0, 2.0, 3.0, 5.0), per_char=4, seed=1,
                        save_path=None, show=True):
    g = torch.Generator().manual_seed(seed)
    labels = torch.tensor([CLASS_NAMES.index(ch) for ch in chars]).repeat_interleave(per_char)
    z0 = torch.randn(len(labels), cfg_config["latent_dim"], generator=g)
    fig, axes = plt.subplots(len(ws), len(labels), figsize=(len(labels) * 0.55, len(ws) * 0.7))
    for i, w in enumerate(ws):
        imgs = generate_cfg(labels, w=w, z0=z0)
        for j in range(len(labels)):
            ax = axes[i, j]
            ax.imshow(imgs[j, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"w={w:g}", fontsize=8, rotation=0, labelpad=18)
            if i == 0 and j % per_char == 0:
                ax.set_title(f"'{CLASS_NAMES[labels[j]]}'", fontsize=8)
    fig.suptitle("CFG guidance scale sweep (same z(0) per column)", fontsize=10)
    _finish(fig, save_path, show)


plot_guidance_sweep(save_path=f"{CFG_DIR}/cfg_guidance_sweep.png")"""),

    ("markdown", r"""## 9.4 Quantitative check: does the generated character match the requested class?
We train a small CNN classifier on the real EMNIST-36 training images and use it to classify generated images.
**Class accuracy** = fraction of samples classified as the requested class $c$. We also report the mean
classifier confidence and the pixel-space diversity (mean std across samples of the same class).
Diversity is the pixel std across samples of the *requested* class; at $w=0$ the label is ignored, so those
samples mix several characters and the std is naturally higher.
Note: classes like `0/O`, `1/I`, `5/S`, `2/Z` are naturally confused even for real images, so the
classifier accuracy on the real test set is the practical upper bound."""),

    ("code", r"""class SmallCNN(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.net(x)


def train_classifier(train_loader, test_loader, epochs=3, device="cpu"):
    clf = SmallCNN().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for epoch in range(epochs):
        clf.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (clf(x.to(device)).argmax(1).cpu() == y).sum().item()
            n += len(y)
    print(f"Classifier accuracy on real test images: {correct / n:.3f}")
    return clf, correct / n


@torch.no_grad()
def evaluate_generation(w, per_class=100, num_steps=25, method="heun", seed=0):
    torch.manual_seed(seed)
    labels = torch.arange(36).repeat_interleave(per_class)
    imgs = torch.cat([generate_cfg(labels[i:i + 1200], w=w, num_steps=num_steps, method=method)
                      for i in range(0, len(labels), 1200)])
    probs = torch.cat([clf(imgs[i:i + 1200].to(device)).softmax(1).cpu() for i in range(0, len(imgs), 1200)])
    acc = (probs.argmax(1) == labels).float().mean().item()
    conf = probs[torch.arange(len(labels)), labels].mean().item()
    diversity = imgs.view(36, per_class, -1).std(dim=1).mean().item()
    return {"w": w, "method": method, "steps": num_steps, "class_acc": acc, "confidence": conf,
            "diversity": diversity}


clf, real_acc = train_classifier(train_dl, test_dl, epochs=3, device=device)"""),

    ("code", r"""import json

cfg_results = [evaluate_generation(w) for w in (0.0, 1.0, 2.0, 3.0, 5.0)]
cfg_results += [evaluate_generation(3.0, num_steps=s, method=m) for m, s in (("euler", 10), ("euler", 50), ("heun", 5))]

print(f"{'sampler':>8} {'steps':>5} {'w':>4} | {'class acc':>9} {'confidence':>10} {'diversity':>9}")
for r in cfg_results:
    print(f"{r['method']:>8} {r['steps']:>5} {r['w']:>4g} | {r['class_acc']:>9.3f} {r['confidence']:>10.3f} {r['diversity']:>9.3f}")
print(f"(classifier on real test images: {real_acc:.3f})")

with open(f"{CFG_DIR}/cfg_metrics.json", "w") as f:
    json.dump({"classifier_real_test_acc": real_acc, "results": cfg_results}, f, indent=2)

sweep = [r for r in cfg_results if r["method"] == "heun" and r["steps"] == 25]
fig, ax1 = plt.subplots(figsize=(6, 3.5))
ax1.plot([r["w"] for r in sweep], [r["class_acc"] for r in sweep], "o-", label="class accuracy")
ax1.axhline(real_acc, ls="--", c="gray", lw=1, label="real test images")
ax1.set_xlabel("guidance scale w"); ax1.set_ylabel("class accuracy"); ax1.set_ylim(0, 1)
ax2 = ax1.twinx()
ax2.plot([r["w"] for r in sweep], [r["diversity"] for r in sweep], "s--", c="tab:orange", label="diversity")
ax2.set_ylabel("diversity (pixel std)")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
ax1.set_title("CFG: fidelity vs. diversity")
_finish(fig, f"{CFG_DIR}/cfg_accuracy_vs_w.png", show=True)"""),

    ("markdown", r"""**Results from our run** (100 epochs, 100 samples per class; classifier: 89.8% on real test images):

| Sampler | Steps | $w$ | Class accuracy | Confidence | Diversity |
|---|---|---|---|---|---|
| Heun | 25 | 0 (unconditional) | 2.6% (chance = 2.8%) | 0.03 | 0.208 |
| Heun | 25 | 1 (plain conditional) | 87.6% | 0.83 | 0.171 |
| Heun | 25 | 2 | 96.4% | 0.93 | 0.166 |
| Heun | 25 | **3** | **97.7%** | 0.94 | 0.168 |
| Heun | 25 | 5 | 98.6% | 0.95 | 0.175 |
| Euler | 10 | 3 | 97.8% | 0.94 | 0.154 |
| Euler | 50 | 3 | 97.8% | 0.94 | 0.165 |
| Heun | 5 | 3 | 97.7% | 0.94 | 0.166 |

- **CFG works:** with $w=0$ the label is ignored (chance-level accuracy). $w=1$ already reaches 88%, and $w \ge 2$ exceeds 96%, above the classifier's accuracy on *real* images, because guidance produces more class-typical characters.
- **Trade-off:** larger $w$ trades diversity for fidelity; $w \approx 2$–$3$ is a good balance.
- **Few steps suffice:** the rectified paths are nearly straight, so 5 Heun steps or 10 Euler steps match 50 steps."""),

    ("markdown", r"""## 9.5 Try it yourself (CFG)
Type a character and choose the guidance scale $w$.
> Requires the VAE (Section 4) and the CFG flow (Section 9.2) in the current runtime."""),

    ("code", r"""import ipywidgets as widgets
from IPython.display import display, clear_output

cfg_char = widgets.Text(value="A", placeholder="0-9 or A-Z", description="Character:",
                        layout=widgets.Layout(width="180px"))
cfg_w = widgets.FloatSlider(value=3.0, min=0.0, max=8.0, step=0.5, description="w:")
cfg_button = widgets.Button(description="Generate", button_style="primary", icon="play")
cfg_out = widgets.Output()


def on_cfg_click(_):
    ch = cfg_char.value.strip().upper()
    with cfg_out:
        clear_output(wait=True)
        if len(ch) != 1 or ch not in CLASS_NAMES:
            print(f"Invalid input '{ch}'. Use one of: {''.join(CLASS_NAMES)}")
            return
        imgs = generate_cfg(torch.full((8,), CLASS_NAMES.index(ch)), w=cfg_w.value)
        fig, axes = plt.subplots(1, 8, figsize=(8, 1.3))
        for ax, img in zip(axes, imgs):
            ax.imshow(img[0], cmap="gray", vmin=0, vmax=1); ax.axis("off")
        fig.suptitle(f"'{ch}'  (w = {cfg_w.value:g})", fontsize=10)
        _finish(fig, None, show=True)


cfg_button.on_click(on_cfg_click)
display(widgets.HBox([cfg_char, cfg_w, cfg_button]), cfg_out)"""),
]
