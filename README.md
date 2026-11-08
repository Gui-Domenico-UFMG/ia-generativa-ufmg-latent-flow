# Trabalho Prático Final — IA Generativa (UFMG 2026/2)
## Latent Diffusion / Rectified Flow no Dataset EMNIST (36 Classes)

Repositório do grupo para o **Trabalho Prático Final** da disciplina de **Inteligência Artificial Generativa (2026/2)** da **UFMG**, ministrada pelo **Prof. Vinicius M. Gonçalves**.

---

## 1. Visão Geral do Trabalho

- **Pontuação:** **20 pontos** + até **5 pontos extras (bônus)**.
- **Dataset:** **EMNIST** — variante contendo **36 classes** (`10 dígitos` de `0` a `9` + `26 letras` de `A` a `Z`).
- **Framework obrigatório:** **PyTorch**.
- **Tamanho do Grupo:** Até 3 integrantes.
  - **Planilha de Cadastro dos Grupos:** [Google Sheets Oficial](https://docs.google.com/spreadsheets/d/1G2pjP_xNPxVwiSrMmVFaKX7bPj9JlPQ13X7mQiDTp4s/edit?usp=sharing)

---

## 2. O Que Precisa Ser Feito (Resumo Técnico)

O objetivo é gerar novos caracteres manuscritos do EMNIST combinando **redução de dimensionalidade via VAE** com um **modelo de fluxo (Rectified Flow)** no espaço latente:

### Etapa 1 — Variational Autoencoder (VAE — Parte 5)
Treinar um modelo de **VAE** ($\mu_E, \sigma_E, \mu_D$) para comprimir as imagens $x \in \mathbb{R}^n$ ($28 \times 28 = 784$) em um espaço latente $\mathbb{R}^m$ ($m \ll n$):
- **Encoder Probabilístico:**
  $$Z(x) = \mu_E(x) + \sigma_E(x) W_E, \qquad W_E \sim \mathcal{N}(0, I)$$
- **Decoder:**
  $$\hat{x} = \mu_D(z)$$
- **Função Objetivo do VAE (Slides 55–59 da Parte 5):**
  $$\min_{\mu_D, \mu_E, \sigma_E} \sum_{i=1}^N \Big\{ R(x_i; \mu_D, \mu_E, \sigma_E) + L(x_i; \mu_D, \mu_E, \sigma_E) \Big\}$$
  onde:
  - **Erro de Reconstrução ($R$):**
    $$R(x; \mu_D, \mu_E, \sigma_E) = \mathbb{E}_W \left[ \frac{\|x - \mu_D(\mu_E(x) + \sigma_E(x)W)\|^2}{2} \right]$$
  - **Erro de Distribuição Latente ($L$):**
    $$L(x; \mu_D, \mu_E, \sigma_E) = m \left( \frac{\sigma_E(x)^2 - 1}{2} - \ln(\sigma_E(x)) \right) + \frac{\|\mu_E(x)\|^2}{2}$$

### Etapa 2 — Campo de Velocidade no Espaço Latente ($\Phi$ — Parte 6)
Treinar uma rede neural que parametriza o **campo de velocidade ($\Phi$)** no espaço latente do VAE:
- **Interpolação Retificada (Parametrização Forward $t \in [0, 1]$):**
  - Ponto inicial (ruído latente em $t=0$): $z(0) \sim \mathcal{N}(0, I)$
  - Ponto final (latente codificado em $t=1$): $z(1) = \mu_E(x)$ (ou amostrado do encoder)
  - Caminho reto entre $z(0)$ e $z(1)$:
    $$z(t) = (1 - t)\,z(0) + t\,z(1)$$
- **Treinamento via Flow Matching:**
  $$\min_{\theta} \frac{1}{M} \sum_{i=1}^M \left\| \Phi\big(z_i(t_i), t_i; \theta\big) - \big(z_i(1) - z_i(0)\big) \right\|^2$$

### Etapa 3 — Geração de Novas Imagens (Amostragem)
1. Amostrar um vetor latente inicial a partir da distribuição normal padrão:
   $$z(0) \sim \mathcal{N}(0, I)$$
2. Refinar esse vetor latente integrando a EDO $\frac{dz}{dt} = \Phi(z(t), t)$ de $t=0$ até $t=1$ (usando *Sampler* de **Euler** ou **Heun**), obtendo o latente final $z(1)$.
3. Decodificar $z(1)$ com o decoder do VAE para obter a imagem gerada:
   $$x = \mu_D(z(1))$$

### Etapa 4 (Bônus: +5 Pontos) — Condicionamento & Classifier-Free Guidance (Parte 7)
- Incluir um sinal de condicionamento $c \in \{0, 1, \dots, 35\}$ na rede de velocidade $\Phi(z, t, c)$ para controlar qual das **36 classes** (26 letras + 10 dígitos) será gerada.
- Implementar **Classifier-Free Guidance (CFG)** combinando a predição condicional e incondicional ($\varnothing$) durante a amostragem:
  $$v_{\text{CFG}}(z, t, c) = \Phi(z, t, \varnothing) + w \cdot \big(\Phi(z, t, c) - \Phi(z, t, \varnothing)\big)$$

---

## 3. Decisões de Engenharia do Grupo

Fica a critério do grupo definir e experimentar:
- Arquitetura das redes neurais do VAE ($\mu_E, \sigma_E, \mu_D$) e do campo de velocidade ($\Phi$)
- Número de parâmetros e camadas
- Dimensão $m$ do espaço latente
- Hiperparâmetros de treinamento e método de integração numérica (*Euler* vs. *Heun*, número de passos $\Delta t$)

---

## 4. Prazos e Regras de Entrega

| Evento | Data / Horário | Detalhes |
| :--- | :---: | :--- |
| **Entrega Final (Código + PDF)** | **30/11/2026 até 23:59** | Enviar **link/código no Google Colab** + **PDF da apresentação** por e-mail |
| **Apresentação Online** | **01/12/2026** | Apresentação de **15 minutos** via Google Meet (em Português ou Inglês) |

### Instruções de Envio do E-mail
- **Destinatário:** [vinicius.marianog@gmail.com](mailto:vinicius.marianog@gmail.com) (enviado por apenas **um membro** do grupo)
- **Assunto obrigatório do e-mail:**
  ```text
  [GENERATIVE AI-2026/2] {NOME DE UM INTEGRANTE DO GRUPO}
  ```

---

## 5. Checklist de Tarefas para o Grupo

> 📝 **Registre o que você fez no [CHANGELOG.md](CHANGELOG.md)** e marque seu nome na tabela "Em andamento" antes de pegar uma tarefa.

- [ ] Cadastrar o grupo na [planilha oficial](https://docs.google.com/spreadsheets/d/1G2pjP_xNPxVwiSrMmVFaKX7bPj9JlPQ13X7mQiDTp4s/edit?usp=sharing)
- [ ] Preparar o carregamento e pré-processamento do dataset **EMNIST (36 classes)** em PyTorch
- [ ] Definir a arquitetura e treinar o **VAE** ($\mu_E, \sigma_E, \mu_D$)
- [ ] Avaliar a qualidade de reconstrução e a organização do espaço latente do VAE
- [ ] Definir a arquitetura e treinar o modelo de **Flow ($\Phi$)** no espaço latente
- [ ] Implementar o *Sampler* ($z(0) \to z(1)$) e validar a geração $x = \mu_D(z(1))$
- [ ] **(Bônus +5 pts)** Implementar o condicionamento nas 36 classes + **Classifier-Free Guidance (CFG)**
- [ ] Consolidar o notebook final no **Google Colab**
- [ ] Montar os slides da **apresentação em PDF** (15 min)
- [ ] Enviar o e-mail até **30/11/2026 às 23:59**
