# Latent Diffusion / Rectified Flow no EMNIST (36 Classes) com Classifier-Free Guidance

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gui-Domenico-UFMG/ia-generativa-ufmg-latent-flow/blob/main/Latent_Flow_EMNIST_Colab.ipynb)

Trabalho Prático Final da disciplina de **IA Generativa (2026/2)** — **UFMG** (Prof. Vinicius M. Gonçalves).

---

## Objetivo do Projeto

Implementar um modelo generativo de **Latent Diffusion / Rectified Flow** em **PyTorch** para o dataset **EMNIST (36 classes: 10 dígitos `0-9` e 26 letras `A-Z`)**, incluindo **Condicionamento de Classe e Classifier-Free Guidance (CFG)** (+5 pontos bônus).

### Formulação Matemática Implementada

1. **Variational Autoencoder (VAE — Parte 5):**
   - **Encoder Probabilístico:** $Z(x) = \mu_E(x) + \sigma_E(x) W_E$, onde $W_E \sim \mathcal{N}(0, I)$.
   - **Decoder:** $\hat{x} = \mu_D(z) \in [0, 1]^{1 \times 28 \times 28}$.
   - **Função Objetivo (Slides 55–59):**
     $$\min_{\mu_D, \mu_E, \sigma_E} \sum_{i=1}^N \Big\{ R(x_i; \mu_D, \mu_E, \sigma_E) + \beta L(x_i; \mu_D, \mu_E, \sigma_E) \Big\}$$
     onde:
     $$R(x) = \mathbb{E}_W \left[ \frac{\|x - \mu_D(\mu_E(x) + \sigma_E(x) W)\|^2}{2} \right], \qquad L(x) = \sum_{j=1}^m \left( \frac{\sigma_{E,j}(x)^2 - 1}{2} - \ln(\sigma_{E,j}(x)) \right) + \frac{\|\mu_E(x)\|^2}{2}$$

2. **Campo de Velocidade no Espaço Latente ($\Phi$ — Parte 6):**
   - **Interpolação Retificada (Parametrização Forward $t \in [0, 1]$, Slide 72):**
     $$z(0) \sim \mathcal{N}(0, I), \qquad z(1) \sim \mathcal{N}(\mu_E(x), \sigma_E(x)^2 I), \qquad z(t) = (1 - t) z(0) + t z(1)$$
   - **Objetivo de Flow Matching (Slide 52):**
     $$J(\theta) = \frac{1}{M} \sum_{i=1}^M \left\| \Phi\big(z_i(t_i), t_i, c_i; \theta\big) - \big(z_i(1) - z_i(0)\big) \right\|^2$$

3. **Condicionamento & Classifier-Free Guidance (Parte 7 — Bônus):**
   - Durante o treino de $\Phi(z, t, c)$, o rótulo $c \in \{0, \dots, 35\}$ é substituído pelo token incondicional $\varnothing = 36$ com probabilidade $p_{\text{uncond}} = 0.15$.
   - Na geração, integramos $z(0) \sim \mathcal{N}(0, I)$ de $t=0$ até $t=1$ usando **Euler** (1ª ordem) ou **Heun** (2ª ordem) com velocidade guiada:
     $$v_{\text{CFG}}(z, t, c) = \Phi(z, t, \varnothing) + w \cdot \big(\Phi(z, t, c) - \Phi(z, t, \varnothing)\big)$$
   - Imagem final gerada: $x = \mu_D(z(1))$.

---

## Estrutura do Repositório

```text
├── Latent_Flow_EMNIST_Colab.ipynb  # Notebook completo pronto para rodar no Google Colab (GPU)
├── train.py                        # Script de treinamento via linha de comando
├── requirements.txt                # Dependências Python/PyTorch
└── src/
    ├── dataset.py                  # Dataloader EMNIST (36 classes: '0'-'9', 'A'-'Z') com orientação corrigida
    ├── vae.py                      # Arquitetura Conv-VAE (mu_E, sigma_E, mu_D) e função objetivo R(x) + L(x)
    ├── flow.py                     # Rede AdaLN-MLP para o campo de velocidade Phi(z, t, c), Samplers Euler/Heun e CFG
    └── vis.py                      # Funções de plotagem (reconstruções, grade 36 classes e trajetórias z(0)->z(1))
```

---

## Como Rodar

### Opção 1: Direto no Google Colab (Recomendado)
1. Clique no botão **[Open in Colab](https://colab.research.google.com/github/Gui-Domenico-UFMG/ia-generativa-ufmg-latent-flow/blob/main/Latent_Flow_EMNIST_Colab.ipynb)** no topo deste README.
2. No menu do Colab, verifique se a GPU está ativa em `Runtime` → `Change runtime type` → `T4 GPU`.
3. Execute todas as células (`Runtime` → `Run all`).

### Opção 2: Localmente via Terminal
```bash
pip install -r requirements.txt
python train.py --latent-dim 16 --vae-epochs 15 --flow-epochs 30 --cfg-scale 2.5
```

---

## Prazo e Instruções de Entrega
- **Data limite:** **30/11/2026 às 23:59**
- **E-mail:** `vinicius.marianog@gmail.com`
- **Assunto do e-mail:** `[GENERATIVE AI-2026/2] {NOME DE UM INTEGRANTE DO GRUPO}`
- **Entregáveis:** Link/arquivo do Google Colab + PDF da apresentação (apresentação online de 15 min em **01/12/2026**).
