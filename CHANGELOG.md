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
| _a preencher_ | | |
| _a preencher_ | | |
| _a preencher_ | | |

---

## Em andamento

Coloque seu nome ao pegar uma tarefa; remova quando terminar (e registre no Registro).

| Tarefa (checklist do README) | Responsável | Status |
| :--- | :--- | :--- |
| Cadastrar o grupo na planilha | | ⬜ a fazer |
| Dataloader EMNIST (36 classes) | Guilherme | ✅ concluído |
| Arquitetura + treino do VAE | Guilherme | 🟨 código pronto; falta rodar no Colab com GPU |
| Validação das reconstruções do VAE | Guilherme | 🟨 figuras prontas; avaliar após treino real |
| Arquitetura + treino do Flow (Φ) | | ⬜ a fazer |
| Sampler z(0) → z(1) e geração x = μ_D(z(1)) | | ⬜ a fazer |
| (Bônus) Condicionamento 36 classes + CFG | | ⬜ a fazer |
| Notebook final no Colab | | ⬜ a fazer |
| Slides em PDF (15 min) | | ⬜ a fazer |
| Envio do e-mail (até 30/11 23:59) | | ⬜ a fazer |

Status: ⬜ a fazer · 🟨 em andamento · ✅ concluído

---

## Registro

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
