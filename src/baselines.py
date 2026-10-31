"""Baselines : méthodes simples que le réseau devra battre.

À lancer depuis la racine du dépôt :  python -m src.baselines

On prédit la température H heures à l'avance (H = 1, 6, 24). À l'instant t, on
connaît la série jusqu'à t, et on prédit la valeur à t + H. Ces méthodes n'ont
aucun paramètre à régler : on peut donc les mesurer sur la validation et le test
sans fuite de données.
"""
import json

import pandas as pd

from src.dataset import (
    DATA_PATH,
    ROOT,
    TARGET,
    chronological_split,
    clean_hourly,
    load_raw,
    unreliable_mask,
)

METRICS_DIR = ROOT / "reports" / "metrics"
HORIZONS = (1, 6, 24)  # nombre d'heures à l'avance


def baseline_predictions(series, horizon, train_mean):
    """Prédictions de chaque baseline, indexées par l'heure CIBLE (celle qu'on prédit).

    - repère : toujours la température moyenne de l'entraînement (le niveau « sans rien savoir »).
    - naïve : la dernière valeur connue (celle de l'instant t).
    - même heure hier : la valeur observée à la même heure la veille.
    - moyenne 24 h : la moyenne des 24 dernières heures connues.

    La série est horaire et sans trou : décaler de N lignes revient à décaler de N heures.
    """
    return {
        "repère : moyenne de l'entraînement": pd.Series(train_mean, index=series.index),
        "naïve : dernière valeur": series.shift(horizon),
        "même heure hier": series.shift(24),
        "moyenne des 24 dernières heures": series.rolling(24).mean().shift(horizon),
    }


def score(actual, predicted, reliable):
    """MAE et RMSE en °C.

    On ignore les instants sans prédiction possible et les instants peu fiables
    (valeurs comblées), pour que toutes les méthodes soient mesurées sur les mêmes points.
    """
    errors = (predicted - actual)[reliable.loc[actual.index]].dropna()
    return {
        "MAE": float(errors.abs().mean()),
        "RMSE": float((errors ** 2).mean() ** 0.5),
    }


def main():
    raw, _ = load_raw(DATA_PATH)
    hourly, _, filled = clean_hourly(raw)
    reliable = ~unreliable_mask(filled)
    series = hourly[TARGET]
    train, val, test = chronological_split(series)
    train_mean = float(train.mean())

    print(
        f"Heures exclues (valeurs comblées ou proches d'une lacune) : "
        f"validation {int((~reliable.loc[val.index]).sum())}, test {int((~reliable.loc[test.index]).sum())}\n"
    )

    results = {}
    print("Erreurs en °C (MAE = erreur moyenne, RMSE = pénalise les grosses erreurs)\n")
    print(f"{'Horizon':<9}{'Méthode':<38}{'Validation':>20}{'Test':>20}")
    for horizon in HORIZONS:
        predictions = baseline_predictions(series, horizon, train_mean)
        results[horizon] = {}
        for name, predicted in predictions.items():
            val_score = score(val, predicted.loc[val.index], reliable)
            test_score = score(test, predicted.loc[test.index], reliable)
            results[horizon][name] = {"validation": val_score, "test": test_score}
            print(
                f"+{horizon:>2} h    {name:<38}"
                f"{'MAE %.2f RMSE %.2f' % (val_score['MAE'], val_score['RMSE']):>20}"
                f"{'MAE %.2f RMSE %.2f' % (test_score['MAE'], test_score['RMSE']):>20}"
            )
        print()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "baselines.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Résultats enregistrés dans : {path}")


if __name__ == "__main__":
    main()