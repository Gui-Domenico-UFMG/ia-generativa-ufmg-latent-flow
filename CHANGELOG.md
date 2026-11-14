# Changelog do Grupo

Registro de **quem fez o quê e quando** no Trabalho Prático Final (IA Generativa — UFMG 2026/2).

## ⚠️ Regra de Coordenação (leia antes de começar)

> **Antes de começar qualquer tarefa, olhe a tabela "Em andamento" abaixo.**
> **Se uma tarefa já tem alguém (pessoa OU IA) com o nome marcado, NÃO comece a mesma tarefa.**
> Isso vale tanto para integrantes do grupo quanto para qualquer assistente de IA (Claude, ChatGPT, etc.)
> usado por qualquer um dos três. Como o grupo vai usar IA para ajudar no trabalho, é fácil duas pessoas
> pedirem a uma IA a mesma coisa ao mesmo tempo sem saber — **essa tabela existe exatamente para evitar isso.**
>
> Fluxo obrigatório:
> 1. Abra o CHANGELOG.md e veja a tabela "Em andamento".
> 2. Se a tarefa que você quer pegar estiver livre (⬜), coloque seu nome e mude o status para 🟨, **e já comite isso antes de começar a trabalhar** (`git commit -m "chore: claim tarefa X"` + push), para avisar os outros o mais rápido possível.
> 3. Se já tiver alguém, **escolha outra tarefa** ou fale com a pessoa antes de tocar no mesmo código.
> 4. Ao terminar, mude o status para ✅, tire seu nome da linha (ou deixe como "concluído por X") e adicione uma entrada no [Registro](#registro).
> 5. Isso também vale para pedir a uma IA (Claude Code, ChatGPT, Cursor, etc.) para "continuar o projeto" — a IA deve ler este arquivo primeiro e respeitar o que já está marcado como em andamento ou concluído por outra pessoa.
>
> Veja também [`CLAUDE.md`](CLAUDE.md) — regras específicas para quando qualquer um de nós usar uma IA neste repositório.

## Como usar

1. Sempre que terminar uma tarefa (ou parte dela), adicione uma entrada **no topo** da seção [Registro](#registro).
2. Use o formato abaixo. Seja breve, mas diga o que foi feito, onde está e o que falta.
3. Faça commit junto com a sua mudança (ex.: `git commit -m "feat: treina VAE + changelog"`).
4. Antes de começar algo, olhe a seção [Em andamento](#em-andamento) para não duplicar trabalho, coloque seu nome lá **e comite isso primeiro**.

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

> Preencham nome e GitHub de cada um na primeira vez que comitarem.

| Nome | GitHub | Responsabilidades principais |
| :--- | :--- | :--- |
| Guilherme | gui-domenico | Organização do repo, VAE + Flow + CFG (implementação inicial) |
| _integrante 2 — preencher_ | | |
| _integrante 3 — preencher_ | | |

---

## Em andamento

**Regra: só coloque seu nome em uma linha se ela estiver livre (⬜). Se já tiver nome, não pegue a mesma tarefa — veja a regra de coordenação no topo deste arquivo.**

| Tarefa (checklist do README) | Responsável | Status |
| :--- | :--- | :--- |
| Cadastrar o grupo na planilha | | ⬜ a fazer |
| Dataloader EMNIST (36 classes) | Guilherme | 🟨 implementado, aguardando validação com dataset real |
| Arquitetura + treino do VAE | Guilherme | 🟨 implementado, aguardando treino/validação |
| Validação das reconstruções do VAE | | ⬜ a fazer |
| Arquitetura + treino do Flow (Φ) | Guilherme | 🟨 implementado, aguardando treino/validação |
| Sampler z(0) → z(1) e geração x = μ_D(z(1)) | Guilherme | 🟨 implementado (Euler/Heun), aguardando validação |
| (Bônus) Condicionamento 36 classes + CFG | Guilherme | 🟨 implementado, aguardando treino/validação com GPU |
| Notebook final no Colab | | ⬜ a fazer |
| Slides em PDF (15 min) | | ⬜ a fazer |
| Envio do e-mail (até 30/11 23:59) | | ⬜ a fazer |

Status: ⬜ a fazer · 🟨 em andamento · ✅ concluído

---

## Registro

### 2026-11-14 — Guilherme
- **Tarefa:** `[docs]` Changelog para grupo de 3 pessoas + regra de coordenação (anti-duplicação de trabalho, humano ou IA)
- **O que foi feito:** Reescrita da tabela "Integrantes" para 3 vagas; adicionada regra explícita no topo do CHANGELOG dizendo que ninguém (pessoa ou IA) deve pegar uma tarefa que já está marcada como em andamento/concluída por outra pessoa; criado `CLAUDE.md` com a mesma regra para qualquer assistente de IA usado no repo; recuperado do histórico do git o código de VAE + Rectified Flow + CFG (estava em `189e8ce`, havia sido removido num commit de docs) e restaurado em `src/`, `train.py` e `Latent_Flow_EMNIST_Colab.ipynb`; README técnico antigo movido para `docs/IMPLEMENTACAO.md` (sem substituir o README com checklist/prazos do grupo).
- **Arquivos / links:** `CHANGELOG.md`, `CLAUDE.md`, `src/vae.py`, `src/flow.py`, `src/dataset.py`, `src/vis.py`, `train.py`, `docs/IMPLEMENTACAO.md`
- **Resultados:** Código de VAE (R + L / ELBO) e Flow (flow matching + CFG `v = v_uncond + w·(v_cond − v_uncond)`) conferem com as fórmulas do README/enunciado. Treino real (download do EMNIST + GPU) ainda não executado nesta sessão.
- **Próximos passos / pendências:** rodar `train.py` com EMNIST real (idealmente no Colab/GPU) para gerar os pesos, as reconstruções do VAE e as imagens geradas com CFG; cadastrar o grupo na planilha oficial; dividir as tarefas ainda ⬜ entre os 3 integrantes.

### 2026-11-08 — Guilherme
- **Tarefa:** `[docs]` Organização inicial do repositório
- **O que foi feito:** README com o resumo técnico do enunciado, prazos e checklist; `Plano de Ensino.md`; `.gitignore`; criação deste CHANGELOG.
- **Arquivos / links:** `README.md`, `Plano de Ensino.md`, `CHANGELOG.md`
- **Próximos passos / pendências:** cadastrar o grupo na planilha e dividir as tarefas da tabela "Em andamento".
