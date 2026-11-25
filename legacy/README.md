# legacy/ — base inicial do projeto (VAE em scripts)

Código da primeira etapa (dataloader EMNIST-36 + VAE), feito antes do notebook final.
Ele **não é mais necessário** para rodar o trabalho: tudo está autocontido em
[`notebooks/final_task_refactored.ipynb`](../notebooks/final_task_refactored.ipynb). Fica aqui como referência
e porque gerou os artefatos de [`results/base_vae/`](../results/base_vae/).

```bash
cd legacy
pip install -r ../requirements.txt
python train_vae.py                    # m=16, β=0.25, 30 épocas -> outputs/
python train_vae.py --fake --epochs 2  # teste rápido sem baixar o EMNIST
python tools/build_notebook.py         # regera notebooks/01_base_dados_vae.ipynb
```
