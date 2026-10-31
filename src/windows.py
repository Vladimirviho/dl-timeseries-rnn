"""Fabrication des fenêtres glissantes à partir de la série horaire.

Un exemple d'entraînement = les `input_len` dernières heures (entrées) et la
température `horizon` heures après la dernière heure connue (cible).

Règles de rigueur :
- La normalisation utilise les statistiques de l'entraînement seul.
- Une fenêtre appartient au jeu (entraînement, validation, test) de son heure CIBLE.
  Ses entrées peuvent venir de la période précédente : c'est du passé, donc permis.
- Toute fenêtre qui contient une valeur inventée (heure comblée) est exclue.
"""
import numpy as np
import pandas as pd

from src.dataset import (
    DATA_PATH,
    TARGET,
    chronological_split,
    clean_hourly,
    load_raw,
    normalize,
    unreliable_mask,
)

INPUT_LEN = 48  # heures d'historique données au réseau
HORIZON = 24    # heures à l'avance


def make_windows(values, target_col, input_len, horizon):
    """Découpe le tableau (n_heures, n_variables) en fenêtres glissantes.

    Retourne :
    - x : (n_fenêtres, input_len, n_variables), les entrées ;
    - y : (n_fenêtres,), la valeur à prédire (variable cible, normalisée) ;
    - target_pos : position, dans la série, de l'heure cible de chaque fenêtre.
    """
    last_known = np.arange(input_len - 1, len(values) - horizon)  # t : dernière heure connue
    x = np.stack([values[t - input_len + 1:t + 1] for t in last_known])
    target_pos = last_known + horizon
    y = values[target_pos, target_col]
    return x, y, target_pos


def prepare_windows(columns=(TARGET,), input_len=INPUT_LEN, horizon=HORIZON):
    """Charge la série, normalise, fabrique les fenêtres et les répartit par jeu.

    Retourne un dictionnaire avec, pour "train", "val" et "test" : les entrées x,
    les cibles y, les heures cibles `times` et le nombre de fenêtres exclues
    `excluded`, ainsi que la moyenne et l'écart-type de la température
    (pour revenir en °C).
    """
    raw, _ = load_raw(DATA_PATH)
    hourly, _, filled = clean_hourly(raw)
    hourly = hourly[list(columns)]

    train, val, test = chronological_split(hourly)
    train_n, val_n, test_n, mean, std = normalize(train, val, test)
    series = pd.concat([train_n, val_n, test_n])
    values = series.to_numpy(dtype=np.float32)
    target_col = list(columns).index(TARGET)

    x, y, target_pos = make_windows(values, target_col, input_len, horizon)

    # Une fenêtre est fiable si elle ne contient aucune heure comblée,
    # de la première heure d'entrée jusqu'à l'heure cible.
    unreliable = unreliable_mask(filled, window=input_len + horizon).to_numpy()
    reliable = ~unreliable[target_pos]

    n_train, n_val = len(train), len(val)
    split_id = np.where(target_pos < n_train, 0, np.where(target_pos < n_train + n_val, 1, 2))

    result = {"target_mean": float(mean[TARGET]), "target_std": float(std[TARGET])}
    for k, name in enumerate(("train", "val", "test")):
        in_split = split_id == k
        kept = in_split & reliable
        result[name] = {
            "x": x[kept],
            "y": y[kept],
            "times": series.index[target_pos[kept]],
            "excluded": int((in_split & ~reliable).sum()),
        }
    return result