"""Fonctions utilitaires : graine aléatoire et choix du périphérique de calcul."""
import random

import numpy as np
import torch


def set_seed(seed=42):
    """Fixe les graines pour obtenir des résultats reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    """Retourne le GPU NVIDIA s'il existe, sinon le processeur."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")