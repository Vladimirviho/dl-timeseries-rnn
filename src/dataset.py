"""Chargement et préparation de la série météo de Jena (relevés toutes les 10 minutes).

Étapes : lecture du CSV, nettoyage des valeurs invalides, passage à une mesure
par heure, découpage chronologique, normalisation avec les statistiques du seul
jeu d'entraînement.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "jena_climate_2009_2016.csv"

TARGET = "T (degC)"        # variable à prévoir : la température en °C
INVALID_VALUE = -9999.0    # valeur utilisée par le capteur pour signaler une mesure invalide


def load_raw(path=DATA_PATH):
    """Lit le CSV (un relevé toutes les 10 minutes) et indexe par date.

    Les dates en double sont supprimées (on garde la première).
    Retourne le DataFrame et le nombre de lignes supprimées.
    """
    df = pd.read_csv(path)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time").sort_index()
    duplicated = df.index.duplicated(keep="first")
    return df[~duplicated], int(duplicated.sum())


def clean_hourly(raw):
    """Nettoie la série et la ramène à une mesure par heure.

    1. Les valeurs invalides (-9999) deviennent des valeurs manquantes.
    2. Les 6 relevés de chaque heure sont moyennés.
    3. Les heures ou valeurs manquantes sont comblées avec la dernière valeur
       connue (puis la suivante, pour un éventuel début de série manquant).

    Retourne le DataFrame horaire, un petit rapport de ce qui a été corrigé, et un
    masque indiquant les heures entièrement inventées (comblées).
    """
    invalid = raw == INVALID_VALUE
    n_invalid = int(invalid.to_numpy().sum())

    hourly = raw.mask(invalid).resample("1h").mean()
    missing = hourly.isna().all(axis=1)  # heures sans aucun relevé
    n_missing_cells = int(hourly.isna().to_numpy().sum())

    # Plus longue lacune : plus longue suite d'heures manquantes consécutives.
    groups = (~missing).cumsum()
    runs = missing.groupby(groups).sum()
    longest = int(runs.max())
    longest_start = None
    if longest > 0:
        group_id = runs.idxmax()
        longest_start = missing[(groups == group_id) & missing].index[0]

    hourly = hourly.ffill().bfill()
    report = {
        "valeurs_invalides": n_invalid,
        "heures_manquantes": int(missing.sum()),
        "valeurs_comblees": n_missing_cells,
        "plus_longue_lacune_heures": longest,
        "debut_plus_longue_lacune": longest_start,
    }
    return hourly, report, missing


def unreliable_mask(filled, window=49):
    """Heures à exclure de l'évaluation : une heure comblée et les 48 heures qui la suivent.

    Une prédiction qui s'appuie sur des valeurs inventées (comblées) n'est pas
    fiable : par exemple, après une lacune comblée avec la dernière valeur, la
    température reste artificiellement constante et la méthode « naïve » paraît
    parfaite.
    """
    return filled.astype(int).rolling(window, min_periods=1).max().astype(bool)


def chronological_split(df, train_frac=0.7, val_frac=0.15):
    """Découpe dans l'ordre du temps : entraînement, validation, test.

    Aucun mélange : le jeu de validation vient après l'entraînement, et le
    jeu de test après la validation.
    """
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


def normalize(train, val, test):
    """Normalise les trois jeux avec la moyenne et l'écart-type de l'ENTRAÎNEMENT seul.

    Utiliser les statistiques de toute la série serait une fuite de données.
    Retourne les trois jeux normalisés, la moyenne et l'écart-type.
    """
    mean = train.mean()
    std = train.std().replace(0, 1.0)  # évite une division par zéro
    return (train - mean) / std, (val - mean) / std, (test - mean) / std, mean, std