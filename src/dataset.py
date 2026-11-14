import string
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


# Mapeamento das 36 classes: 0..9 -> '0'..'9', 10..35 -> 'A'..'Z'
CLASS_NAMES = list(string.digits) + list(string.ascii_uppercase)
NUM_CLASSES = 36


def index_to_char(idx: int) -> str:
    """Converte o índice da classe (0 a 35) para o caractere correspondente ('0'-'9', 'A'-'Z')."""
    if 0 <= idx < NUM_CLASSES:
        return CLASS_NAMES[idx]
    return "?"


def char_to_index(ch: str) -> int:
    """Converte um caractere ('0'-'9' ou 'A'-'Z', case-insensitive) para o índice 0..35."""
    ch = ch.upper()
    if ch in CLASS_NAMES:
        return CLASS_NAMES.index(ch)
    raise ValueError(f"Caractere '{ch}' inválido. Use '0'-'9' ou 'A'-'Z'.")


class TransposeEMNIST:
    """Corrige a orientação padrão das imagens do EMNIST no torchvision (transpondo H e W)."""
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(-2, -1).contiguous()


def get_emnist_36_datasets(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """
    Carrega o dataset EMNIST contendo exatamente as 36 classes (10 dígitos '0'-'9' e 26 letras 'A'-'Z').
    Utiliza o split 'balanced' filtrando os rótulos 0..35, garantindo classes perfeitamente balanceadas
    e imagens com orientação corrigida em [0, 1].
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        TransposeEMNIST(),
    ])

    train_full = datasets.EMNIST(
        root=data_dir, split="balanced", train=True, download=True, transform=transform
    )
    test_full = datasets.EMNIST(
        root=data_dir, split="balanced", train=False, download=True, transform=transform
    )

    # No EMNIST 'balanced', as classes 0..9 são dígitos (0-9) e 10..35 são as 26 letras (A-Z).
    # As classes 36..46 são minúsculas duplicadas separadas; filtramos targets < 36 para obter as 36 classes.
    train_indices = torch.where(train_full.targets < NUM_CLASSES)[0]
    test_indices = torch.where(test_full.targets < NUM_CLASSES)[0]

    train_dataset = Subset(train_full, train_indices.tolist())
    test_dataset = Subset(test_full, test_indices.tolist())

    return train_dataset, test_dataset


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Retorna os DataLoaders de treino e teste para o EMNIST (36 classes)."""
    train_ds, test_ds = get_emnist_36_datasets(data_dir=data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
