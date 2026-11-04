# Plano de Ensino — IA Generativa (2026/2)

## Informações Básicas
- **Contato do Professor:** via MinhaUFMG ou pelo e-mail [vinicius.marianog@gmail.com](mailto:vinicius.marianog@gmail.com)
- **Frequência Mínima:** 75% de presença obrigatória.
- **Aprovação:** Nota final mínima de **60 pontos**. Não há "Exame Especial".
- **Prova Substitutiva:** Oferecida ao final do semestre apenas para alunos que perderem uma das provas mediante apresentação de justificativa documentada (ex.: atestado médico ou viagem a trabalho).

---

## Avaliação e Notas (Total: 100 pontos + 5 bônus)

| Avaliação | Pontuação | Descrição / Conteúdo | Data |
| :--- | :---: | :--- | :---: |
| **Prova 1 (Exam 1)** | 35 pts | Capítulos / Partes 2, 3 e 4 (Escrita em inglês; respostas em português ou inglês) | 23/09/2026 |
| **Prova 2 (Exam 2)** | 35 pts | Capítulos / Partes 5, 6 e 7 (Escrita em inglês; respostas em português ou inglês) | 02/12/2026 |
| **Frequência (Attendance)** | 10 pts | Pontuação proporcional acima do mínimo de 75% (ver fórmula abaixo) | Ao longo do semestre |
| **Trabalho Prático (Practical Project)** | 20 pts | Implementação de *Latent Diffusion / Flow Models* em PyTorch (dataset EMNIST) + Apresentação | Entrega: 30/11/2026<br>Apresentação: 01/12/2026 |
| **Bônus (Extra)** | +5 pts | Implementação de Condicionamento / *Classifier-Free Guidance* (CFG) para controle das 36 classes | Junto ao Trabalho Prático |

### Cálculo da Nota de Frequência
Seja $N$ o número total de aulas planejadas no semestre ($N = 28$) e $n$ o número de aulas que você compareceu:
- Presença mínima exigida: $0{,}75N$ ($21$ aulas).
- Fórmula da nota de presença (para $n \ge 0{,}75N$):
$$\text{Nota de Frequência} = 10 \times \frac{n - 0{,}75N}{0{,}25N}$$
- Comparecer a 75% das aulas garante $0$ pontos; comparecer a 100% das aulas garante $10$ pontos.

---

## Bibliografia e Referências
1. **Madras Online Course — Mathematical Foundations of Generative AI:**
   - [Playlist no YouTube](https://www.youtube.com/watch?v=skWhn8W9P_Y&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=1)
2. **Blitzstein and Hwang — Introduction to Probability (2nd ed.):**
   - [Harvard Stat 110](https://stat110.hsites.harvard.edu)
3. **Art B. Owen — Monte Carlo Theory, Methods and Examples:**
   - [Livro Online](https://artowen.su.domains/mc/)
4. **Simon J. D. Prince — Understanding Deep Learning:**
   - [UDL Book Online](https://udlbook.github.io/udlbook/)

---

## Estrutura do Curso (Ementa)
- **Part 1:** Introduction
- **Part 2:** Review on Statistics and Probability
- **Part 3:** The Mathematical Problem of Generative AI
- **Part 4:** Generative Adversarial Network (GAN)
- **Part 5:** Variational Autoencoders (VAE)
- **Part 6:** Diffusion and Flow Models
- **Part 7:** Conditioning

---

## Cronograma de Aulas (Schedule)
*Sujeito a alterações ao longo do semestre.*

| Data | Dia | Tópico / Conteúdo |
| :---: | :---: | :--- |
| **10/08/2026** | Seg | **Part 1:** Introduction |
| **12/08/2026** | Qua | **Part 2:** Random Experiment, Outcomes, Random Variables, Probability Density Functions (PDF), Computational Approximation of a PDF (bins) |
| **17/08/2026** | Seg | **Part 2:** Normal Distribution, Independence, Marginal PDF |
| **19/08/2026** | Qua | **Part 2:** Algebra of Random Variables, Expected Value, Law of Large Numbers, Variance |
| **24/08/2026** | Seg | **Part 3:** Encoding and Decoding, Problem Definition, Inverse Transform Sampling, Computational Implementation of Inverse Transform Sampling |
| **26/08/2026** | Qua | **Part 3:** Box-Muller, Multidimensional Independent Normal Distribution (MIND) |
| **31/08/2026** | Seg | **Part 3:** Metropolis-Hasting Sampling, Implementation of Metropolis-Hasting Sampling, Why it fails in high dimensions |
| **02/09/2026** | Qua | **Part 3:** Mixture of Distributions, Mixture of Gaussians, Likelihood, Log Likelihood, Setting up the optimization problem for finding the parameters of the Gaussian Mixture Sampling |
| **07/09/2026** | Seg | *Feriado — Independência do Brasil* |
| **09/09/2026** | Qua | **Part 3:** Implementing the Gaussian Mixture Sampling computationally, stochastic gradient ascent/descent |
| **14/09/2026** | Seg | **Part 4:** Introduction, Entropy, Intuition about Entropy (Shannon Coding Theorem) |
| **16/09/2026** | Qua | **Part 4:** Differential entropy, Shannon-Jensen Divergence, Game-Theoretic interpretation |
| **21/09/2026** | Seg | **Part 4:** Game Theory, Adversarial Game, Zero Sum Game, GAN Formulation, GAN Drawbacks |
| **23/09/2026** | Qua | **FIRST EXAM (35 PTS)** — *Cobre Partes 2, 3 e 4* |
| **28/09/2026** | Seg | **Part 5:** Manifold Hypothesis, Latent Variable, (Traditional) Encoder-Decoder |
| **30/09/2026** | Qua | **Part 5:** Conditional Probability, Conditional Random Variable, Example of Conditional Random Variable, Law of Total Probability, Bayes' Rule |
| **05/10/2026** | Seg | **Part 5:** Variational Autoencoder (VAE) Formulation, Probabilistic Decoder, Comparison with Discrete Mixture of Gaussians, Maximum Log-Likelihood Formulation, Jensen Inequality, Evidence Lower Bound (ELBO) |
| **07/10/2026** | Qua | **PROF. CRISTIANO:** Transformers |
| **12/10/2026** | Seg | *Feriado — Nossa Senhora Aparecida* |
| **14/10/2026** | Qua | **Part 5:** (Review of the previous class) |
| **19/10/2026** | Seg | **Part 5:** Formulation of the Final Objective Function of the VAE |
| **21/10/2026** | Qua | **Part 5:** Computational Implementation of the VAE |
| **26/10/2026** | Seg | **Part 6:** Introduction to Diffusion Models, Forward Process (Recursive), Forward Process (Explicit), Scheduler |
| **28/10/2026** | Qua | **Part 6:** Denoiser, Deterministic Backward Process |
| **02/11/2026** | Seg | *Feriado — Finados* |
| **04/11/2026** | Qua | **Part 6:** Implementation of a Simple Diffusion Process (Gaussian to Uniform) |
| **09/11/2026** | Seg | **Part 6:** Rectified Flows, Velocity Learning (Flow Matching), Sampler, Latent Diffusion |
| **11/11/2026** | Qua | **Part 6:** Implementation of the Rectified Flow Model in PyTorch, Inpainting |
| **16/11/2026** | Seg | **Part 6:** Implementation of the Rectified Flow Model in PyTorch, Inpainting |
| **18/11/2026** | Qua | **Part 7:** Conditioning, Embedding |
| **23/11/2026** | Seg | **Part 7:** Conditional Flow, Classifier-Free Guidance, Negative Guidance |
| **25/11/2026** | Qua | **Part 7:** Implementation in PyTorch |
| **30/11/2026** | Seg | Open to discussion (**PRAZO FINAL DE ENTREGA DO TRABALHO PRÁTICO ÀS 23:59**) |
| **02/12/2026** | Qua | **SECOND EXAM (35 PTS)** — *Cobre Partes 5, 6 e 7* |
