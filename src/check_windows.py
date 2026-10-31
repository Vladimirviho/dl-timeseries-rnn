"""Vérification des fenêtres glissantes : formes, comptes, un exemple vérifié à la main,
et l'erreur de la baseline sur ces mêmes fenêtres.

À lancer depuis la racine du dépôt :  python -m src.check_windows
"""
import numpy as np

from src.dataset import DATA_PATH, TARGET, clean_hourly, load_raw
from src.windows import HORIZON, INPUT_LEN, prepare_windows


def main():
    data = prepare_windows()
    mean, std = data["target_mean"], data["target_std"]
    print(f"Fenêtres : {INPUT_LEN} heures d'historique pour prédire +{HORIZON} h\n")

    print(f"{'Jeu':<13}{'Fenêtres':>9}{'Exclues':>9}   Forme de x")
    for name, label in (("train", "entraînement"), ("val", "validation"), ("test", "test")):
        part = data[name]
        print(f"{label:<13}{len(part['y']):>9}{part['excluded']:>9}   {part['x'].shape}")

    # Un exemple vérifié à la main : on retrouve les vraies températures en °C.
    raw, _ = load_raw(DATA_PATH)
    hourly, _, _ = clean_hourly(raw)
    temperature = hourly[TARGET]

    part = data["train"]
    k = 1000
    target_time = part["times"][k]
    first_input = target_time - np.timedelta64(HORIZON + INPUT_LEN - 1, "h")
    last_input = target_time - np.timedelta64(HORIZON, "h")
    x_celsius = part["x"][k, :, 0] * std + mean
    y_celsius = part["y"][k] * std + mean
    print(f"\nExemple n°{k} du jeu d'entraînement :")
    print(f"  entrées : du {first_input:%d/%m/%Y %H:%M} au {last_input:%d/%m/%Y %H:%M} ({INPUT_LEN} valeurs)")
    print(f"  cible   : {target_time:%d/%m/%Y %H:%M}")
    print(f"  dernière entrée : {x_celsius[-1]:.2f} °C (série : {temperature[last_input]:.2f} °C)")
    print(f"  cible           : {y_celsius:.2f} °C (série : {temperature[target_time]:.2f} °C)")
    assert abs(x_celsius[-1] - temperature[last_input]) < 1e-3
    assert abs(y_celsius - temperature[target_time]) < 1e-3
    print("  -> les valeurs correspondent à la série d'origine.")

    # Baseline sur ces mêmes fenêtres : prédire la dernière valeur connue
    # (à +24 h, c'est aussi la valeur de la même heure la veille).
    print("\nBaseline « dernière valeur » sur ces fenêtres (à battre) :")
    for name, label in (("val", "validation"), ("test", "test")):
        part = data[name]
        errors = (part["x"][:, -1, 0] - part["y"]) * std
        mae = float(np.abs(errors).mean())
        rmse = float(np.sqrt((errors ** 2).mean()))
        print(f"  {label:<11} MAE {mae:.2f} °C | RMSE {rmse:.2f} °C")


if __name__ == "__main__":
    main()