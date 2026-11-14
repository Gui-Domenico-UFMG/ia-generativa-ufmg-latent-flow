"""
Carregamento do EMNIST com exatamente 36 classes: 10 dígitos ('0'-'9') + 26 letras ('A'-'Z').

Escolha do split
----------------
O EMNIST não tem um split pronto com 36 classes. Usamos o split 'balanced' (47 classes):
    rótulos  0..9   -> dígitos '0'..'9'
    rótulos 10..35  -> letras  'A'..'Z'  (maiúsculas; minúsculas parecidas, como c/C e o/O, já vêm juntas)
    rótulos 36..46  -> minúsculas distintas (a, b, d, e, f, g, h, n, q, r, t)  -> DESCARTADAS
Filtrando os rótulos < 36 ficamos com 36 classes balanceadas:
    treino: 2400 imagens/classe  -> 86.400 imagens
    teste :  400 imagens/classe  -> 14.400 imagens

Orientação
----------
As imagens do EMNIST no torchvision vêm transpostas (de lado). Corrigimos trocando H e W.

Formato de saída: tensores x em [0, 1] com shape (1, 28, 28) e rótulo y em {0, ..., 35}.
"""
import string
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

# Mapeamento das 36 classes: 0..9 -> '0'..'9', 10..35 -> 'A'..'Z'
CLASS_NAMES = list(string.digits) + list(string.ascii_uppercase)
NUM_CLASSES = 36
IMG_SHAPE = (1, 28, 28)


def index_to_char(idx: int) -> str:
    """Índice da classe (0..35) -> caractere ('0'-'9', 'A'-'Z')."""
    return CLASS_NAMES[idx] if 0 <= idx < NUM_CLASSES else "?"


def char_to_index(ch: str) -> int:
    """Caractere ('0'-'9' ou 'A'-'Z', sem diferenciar maiúscula/minúscula) -> índice 0..35."""
    ch = ch.upper()
    if ch in CLASS_NAMES:
        return CLASS_NAMES.index(ch)
    raise ValueError(f"Caractere '{ch}' inválido. Use '0'-'9' ou 'A'-'Z'.")


class TransposeEMNIST:
    """Corrige a orientação das imagens do EMNIST (troca altura e largura)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(-2, -1).contiguous()


# Cópia pública dos arquivos originais do NIST (usada só se o download oficial falhar).
MIRROR_URL = "https://media.githubusercontent.com/media/aurelienduarte/emnist/master/gzip/"
MIRROR_FILES = [f"emnist-balanced-{s}-ubyte" for s in
                ("train-images-idx3", "train-labels-idx1", "test-images-idx3", "test-labels-idx1")]


def download_from_mirror(data_dir: str = "./data") -> None:
    """Baixa os 4 arquivos do split 'balanced' do espelho para data_dir/EMNIST/raw (formato do torchvision)."""
    import gzip
    import os
    import shutil
    import urllib.request

    raw = os.path.join(data_dir, "EMNIST", "raw")
    os.makedirs(raw, exist_ok=True)
    for name in MIRROR_FILES:
        dst = os.path.join(raw, name)
        if os.path.exists(dst):
            continue
        print(f"Baixando {name}.gz do espelho...")
        urllib.request.urlretrieve(MIRROR_URL + name + ".gz", dst + ".gz")
        with gzip.open(dst + ".gz", "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        os.remove(dst + ".gz")


def get_emnist_36_datasets(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Retorna (treino, teste) do EMNIST com as 36 classes, imagens em [0, 1] e orientação corrigida."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), TransposeEMNIST()])

    def load(train: bool, download: bool):
        return datasets.EMNIST(root=data_dir, split="balanced", train=train, download=download, transform=transform)

    try:
        train_full, test_full = load(True, True), load(False, True)
    except Exception as e:  # site do NIST fora do ar ou bloqueado
        print(f"Download oficial falhou ({type(e).__name__}); usando o espelho.")
        download_from_mirror(data_dir)
        train_full, test_full = load(True, False), load(False, False)

    train_idx = torch.where(train_full.targets < NUM_CLASSES)[0].tolist()
    test_idx = torch.where(test_full.targets < NUM_CLASSES)[0].tolist()
    return Subset(train_full, train_idx), Subset(test_full, test_idx)


def get_fake_datasets(n_train: int = 2048, n_test: int = 512, seed: int = 0) -> Tuple[Dataset, Dataset]:
    """
    Dados falsos com o mesmo formato do EMNIST-36 (para testar o código sem baixar nada).
    Cada classe é um "borrão" numa posição diferente, para o VAE ter algo para aprender.
    """
    g = torch.Generator().manual_seed(seed)

    def make(n: int) -> TensorDataset:
        y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
        yy, xx = torch.meshgrid(torch.arange(28.0), torch.arange(28.0), indexing="ij")
        cy = 6 + (y // 6).float() * 3.2
        cx = 6 + (y % 6).float() * 3.2
        d2 = (yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2
        x = torch.exp(-d2 / 18.0) + 0.05 * torch.rand(n, 28, 28, generator=g)
        return TensorDataset(x.clamp(0, 1).unsqueeze(1), y)

    return make(n_train), make(n_test)


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
    fake: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """DataLoaders de treino (embaralhado) e teste. Use fake=True para um teste rápido sem internet."""
    train_ds, test_ds = get_fake_datasets() if fake else get_emnist_36_datasets(data_dir)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, drop_last=True
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader
