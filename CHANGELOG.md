# Changelog do Grupo

Registro de **quem fez o quê e quando** no Trabalho Prático Final (IA Generativa — UFMG 2026/2).

## Como usar

1. Sempre que terminar uma tarefa (ou parte dela), adicione uma entrada **no topo** da seção [Registro](#registro).
2. Use o formato abaixo. Seja breve, mas diga o que foi feito, onde está e o que falta.
3. Faça commit junto com a sua mudança (ex.: `git commit -m "feat: treina VAE + changelog"`).
4. Antes de começar algo, olhe a seção [Em andamento](#em-andamento) para não duplicar trabalho, e coloque seu nome lá.

### Modelo de entrada

```markdown
### AAAA-MM-DD — Nome
- **Tarefa:** (ex.: Treinamento do VAE)
- **O que foi feito:** ...
- **Arquivos / links:** (ex.: `notebook.ipynb`, seção 3; link do Colab)
- **Resultados:** (ex.: loss final, imagens geradas, observações)
- **Próximos passos / pendências:** ...
```

Tags úteis no título da tarefa: `[dados]`, `[vae]`, `[flow]`, `[sampler]`, `[cfg]`, `[notebook]`, `[slides]`, `[entrega]`, `[docs]`.

---

## Integrantes

| Nome | GitHub | Responsabilidades principais |
| :--- | :--- | :--- |
| Guilherme Martins Dijan Domenico | Gui-Domenico-UFMG | Dados EMNIST-36 + VAE ✅ · CFG (bônus) ✅ · organização do repo |
| João Paulo Lara Pinto | joaopaulolarap | Notebook final: CVAE, análise de β, Flow Φ + samplers ✅ |
| Fábio | | _a combinar_: slides + entrega |

**Grupo 4** na [planilha oficial](https://docs.google.com/spreadsheets/d/1G2pjP_xNPxVwiSrMmVFaKX7bPj9JlPQ13X7mQiDTp4s/edit?usp=sharing).

---

## Em andamento

Coloque seu nome ao pegar uma tarefa; remova quando terminar (e registre no Registro).

| Tarefa (checklist do README) | Responsável | Status |
| :--- | :--- | :--- |
| Cadastrar o grupo na planilha | Grupo 4 | ✅ concluído |
| Dataloader EMNIST (36 classes) | Guilherme | ✅ concluído |
| Arquitetura + treino do VAE (+ análise de β) | Guilherme / João Paulo | ✅ concluído |
| CVAE (VAE condicional) | João Paulo | ✅ concluído |
| Flow Φ(z, t) + sampler Euler/Heun | João Paulo | ✅ concluído (`notebooks/final_task.ipynb`) |
| (Bônus) Φ(z, t, c) + CFG | Guilherme | ✅ concluído (seção *Bonus (+5)*, `results/cfg_flow/`) |
| Rodar *Executar tudo* no Colab (GPU) e salvar com saídas | | ⬜ a fazer |
| Slides em PDF (15 min) — meta 28/11 | | ⬜ a fazer |
| Envio do e-mail (até 30/11 23:59) | | ⬜ a fazer |

Status: ⬜ a fazer · 🟨 em andamento · ✅ concluído

---

## Registro

### 2026-11-24 — Guilherme
- **Tarefa:** `[cfg]` `[notebook]` `[docs]` Bônus (CFG) + organização do repositório
- **O que foi feito:**
  - Nova seção **Bonus (+5)** no `notebooks/final_task.ipynb`, antes de *Try it yourself*: campo de velocidade condicional Φ(z, t, c) com blocos AdaLN, embedding de classe + token nulo, *label dropout* (p = 0,15), EMA dos pesos e amostragem Euler/Heun com **Classifier-Free Guidance**. Treinado sobre os latentes do **VAE incondicional**.
  - Avaliação quantitativa: um CNN classificador treinado no EMNIST real mede se o caractere gerado é da classe pedida (acurácia × *w*, Euler × Heun, diversidade).
  - Widget com controle deslizante para *w*. Célula de setup (cria as pastas) e badge do Colab no topo do notebook.
  - README reescrito (visão geral, resultados, como rodar, estrutura). O enunciado foi para `docs/enunciado.md` e o plano de ensino para `docs/plano_de_ensino.md`. O código antigo em scripts (`src/`, `train_vae.py`, notebook 01) foi para `legacy/`.
- **Arquivos / links:** `tools/cfg_cells.py` (fonte das células), `tools/add_cfg_section.py`, `results/cfg_flow/`
- **Resultados:** 100 épocas em CPU (~5 min). Acurácia de classe das amostras (CNN treinado no EMNIST real, que acerta 89,8% no teste real): *w* = 0 → 2,6% (acaso), *w* = 1 → 87,6%, *w* = 3 → **97,7%**, *w* = 5 → 98,6%. Heun com 5 passos já é suficiente. Figuras e métricas em `results/cfg_flow/`. As saídas do notebook ainda precisam ser geradas no Colab.
- **Próximos passos / pendências:** rodar *Executar tudo* no Colab com GPU e salvar o notebook com as saídas; slides; e-mail.

### 2026-11-23 — João Paulo
- **Tarefa:** `[notebook]` `[vae]` `[flow]` `[sampler]` Notebook final
- **O que foi feito:** `notebooks/final_task.ipynb` com dados, VAE, análise de β, **CVAE** (decoder condicionado na classe), **Rectified Flow** no latente (VelocityNet com embedding senoidal, samplers Euler e Heun), visualização do campo de velocidade em PCA, geração completa flow + CVAE e widget "Try it yourself".
- **Arquivos / links:** `notebooks/final_task.ipynb`
- **Resultados:** flow treinado por 200 épocas nos latentes do CVAE (MSE de validação ≈ 1,32).
- **Observação:** registrado por Guilherme a partir do commit `de9aafb`. João / Fábio, corrijam se a divisão foi outra.

### 2026-11-14 — Guilherme
- **Tarefa:** `[entrega]` Cadastro do grupo
- **O que foi feito:** grupo cadastrado na planilha oficial como **Grupo 4** (João Paulo Lara, Fábio, Guilherme Martins Dijan Domenico).
- **Próximos passos / pendências:** João Paulo e Fábio combinarem quem pega o Flow (Φ) e quem pega o CFG + notebook + slides.

### 2026-11-14 — Guilherme
- **Tarefa:** `[vae]` Treino real do VAE + entrega para o grupo
- **O que foi feito:** VAE treinado no EMNIST-36 real (20 épocas, CPU, ~21 min). Checkpoint, latentes, figuras e log salvos no repositório. Download do EMNIST com espelho de reserva.
- **Arquivos / links:** `results/base_vae/` (`vae.pt`, `latents.pt`, figuras, `metrics.json`, `train_log.txt`)
- **Resultados:** teste R(x) = 5,37, L(x) = 23,0, 16/16 dimensões ativas, μ_E com média ≈ 0 e desvio ≈ 0,97. Reconstruções nítidas; amostras de N(0, I) sem Flow ainda misturadas (baseline).
- **Próximos passos / pendências:** **m = 16 combinado.** Flow (Φ) pode começar direto de `results/base_vae/latents.pt`.

### 2026-11-14 — Guilherme
- **Tarefa:** `[dados]` `[vae]` Base do projeto: dataloader EMNIST-36 + VAE
- **O que foi feito:**
  - `src/dataset.py`: EMNIST `balanced` filtrado para 36 classes (86.400 treino / 14.400 teste), orientação corrigida, modo `fake` para testes.
  - `src/vae.py`: VAE convolucional (encoder com μ_E e log σ_E, decoder μ_D), loss R(x) + β·L(x) do enunciado (σ diagonal ou escalar), `save_vae`, `load_vae`, `encode_dataset`.
  - `src/vis_vae.py`: figuras de validação (reconstruções, amostras de N(0, I), interpolação, PCA, curvas).
  - `train_vae.py`: treino + escolha do melhor checkpoint + exportação de `vae.pt` e `latents.pt`.
  - `notebooks/01_base_dados_vae.ipynb`: versão Colab autossuficiente (gerada por `tools/build_notebook.py`).
- **Arquivos / links:** ver seção 6 do `README.md`.
- **Resultados:** código testado ponta a ponta com dados falsos (CPU). Treino real ainda não rodado (precisa baixar o EMNIST no Colab).
- **Próximos passos / pendências:** rodar o notebook no Colab com GPU, conferir as figuras, compartilhar `vae.pt` e `latents.pt` (Drive) com quem fará o Flow. Combinar m = 16 com o grupo.

### 2026-11-08 — Guilherme
- **Tarefa:** `[docs]` Organização inicial do repositório
- **O que foi feito:** README com o resumo técnico do enunciado, prazos e checklist; `Plano de Ensino.md`; `.gitignore`; criação deste CHANGELOG.
- **Arquivos / links:** `README.md`, `Plano de Ensino.md`, `CHANGELOG.md`
- **Próximos passos / pendências:** cadastrar o grupo na planilha e dividir as tarefas da tabela "Em andamento".
