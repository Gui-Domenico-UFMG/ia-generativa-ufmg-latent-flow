# Instruções para IA neste repositório

Este repositório é usado por um grupo de **3 pessoas** no Trabalho Prático Final de IA Generativa
(UFMG 2026/2). Os três integrantes vão usar assistentes de IA (Claude Code, ChatGPT, Cursor, etc.)
para ajudar no trabalho. Esta é a regra número um para qualquer IA que operar neste repositório,
em qualquer sessão, de qualquer integrante:

## Regra obrigatória: não duplicar trabalho

1. **Antes de implementar, treinar, editar ou "continuar" qualquer parte do projeto**, leia
   [`CHANGELOG.md`](CHANGELOG.md), especialmente a seção **"Em andamento"**.
2. Se a tarefa que te pediram já está marcada como 🟨 (em andamento) ou ✅ (concluída) por outra
   pessoa, **não a refaça do zero**. Em vez disso:
   - avise o usuário explicitamente que aquela tarefa já tem alguém trabalhando nela (diga o nome);
   - pergunte se ele quer mesmo substituir/revisar o trabalho existente, ou se quer pegar outra tarefa
     da lista (⬜ a fazer);
   - só sobrescreva código/resultados já existentes se o usuário confirmar isso depois de ver o aviso.
3. Se a tarefa está livre (⬜), antes de começar a trabalhar de fato:
   - atualize a tabela "Em andamento" no `CHANGELOG.md` com o nome do integrante (não da IA) e status 🟨;
   - comite e, se possível, faça push dessa mudança **antes** de começar o trabalho pesado, para que os
     outros dois integrantes vejam o quanto antes que aquela tarefa foi pega.
4. Ao terminar, atualize o status para ✅ e adicione uma entrada na seção "Registro" do `CHANGELOG.md`
   (o que foi feito, arquivos, resultados, pendências), seguindo o modelo que já está no arquivo.
5. Nunca apague ou sobrescreva silenciosamente commits, branches ou arquivos de outro integrante.
   Se um merge/conflito aparecer, pare e peça para o usuário decidir, em vez de resolver escolhendo
   um lado arbitrariamente quando o conteúdo for substancialmente diferente.

## Contexto rápido do projeto

- Objetivo: VAE + Rectified Flow no espaço latente para gerar caracteres do EMNIST (36 classes:
  dígitos 0–9 e letras A–Z). Bônus de +5 pontos: condicionamento por classe + Classifier-Free Guidance (CFG).
- Framework obrigatório: PyTorch.
- Entrega: 30/11/2026 (Colab + PDF por e-mail) e apresentação em 01/12/2026. Ver `README.md` para
  prazos, regras de entrega e o checklist completo.
- Código principal: `src/vae.py` (VAE), `src/flow.py` (campo de velocidade Φ + CFG), `src/dataset.py`
  (EMNIST 36 classes), `train.py` (script de treino fim a fim), `Latent_Flow_EMNIST_Colab.ipynb`
  (notebook para rodar no Google Colab). Documentação técnica detalhada da implementação em
  `docs/IMPLEMENTACAO.md`.
- O código de VAE/Flow/CFG já existe e foi conferido contra as fórmulas do enunciado, mas **ainda não
  foi treinado/validado com o dataset EMNIST real** (falta rodar com GPU, idealmente no Colab).
