"""Évaluation finale d'un modèle sur le jeu de TEST.

À n'utiliser qu'une fois le modèle choisi (sur la validation) : le jeu de test
sert à donner la performance finale, pas à régler le modèle.

Exemple (depuis la racine du dépôt) :
    python -m src.evaluate --model lstm --hidden 32 --features T,p,rh,wv --weights models/lstm_24h_48l_32u_T-p-rh-wv.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.analysis import monthly_table, plot_monthly, plot_week, summarize
from src.models import build_model, count_parameters
from src.train import FEATURES, METRICS_DIR
from src.utils import get_device
from src.windows import prepare_windows


def predict(model, x, device, batch_size=1024):
    """Prédit toutes les fenêtres (valeurs normalisées)."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser(description="Évaluation finale sur le jeu de test")
    parser.add_argument("--model", choices=["rnn", "lstm", "gru"], default="lstm")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--features", default="T,p,rh,wv")
    parser.add_argument("--weights", required=True, help="fichier de poids du modèle (.pt)")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Fichier de poids introuvable : {weights}")

    keys = [key.strip() for key in args.features.split(",")]
    unknown = [key for key in keys if key not in FEATURES]
    if unknown:
        raise SystemExit(f"Variables inconnues : {unknown} (choix possibles : {list(FEATURES)})")
    keys = ["T"] + [key for key in keys if key != "T"]
    columns = tuple(FEATURES[key] for key in keys)

    device = get_device()
    data = prepare_windows(columns=columns, input_len=args.input_len, horizon=args.horizon)
    mean, std = data["target_mean"], data["target_std"]
    test = data["test"]

    model = build_model(
        args.model, input_size=len(columns), hidden_size=args.hidden, num_layers=args.layers
    ).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))

    # Retour en °C : prévisions du modèle, valeurs réelles, et baseline « dernière valeur ».
    prediction = predict(model, test["x"], device) * std + mean
    actual = test["y"] * std + mean
    baseline = test["x"][:, -1, 0] * std + mean
    times = test["times"]

    summary = summarize(prediction, actual, baseline)
    low, high = summary["gain_mae_ci95"]
    print(f"Modèle : {args.model} ({count_parameters(model)} paramètres) | poids : {weights.name}")
    print(f"Variables : {', '.join(columns)}")
    print(f"Jeu de test : {summary['n_windows']} fenêtres ({args.input_len} h d'historique pour prédire +{args.horizon} h)\n")
    print(f"{'':<22}{'MAE':>8}{'RMSE':>8}")
    print(f"{'modèle':<22}{summary['mae_model']:>8.2f}{summary['rmse_model']:>8.2f}   (°C)")
    print(f"{'baseline (dernière val.)':<22}{summary['mae_baseline']:>8.2f}{summary['rmse_baseline']:>8.2f}   (°C)")
    print(f"\nGain de MAE sur la baseline : {summary['gain_percent']:.1f} %")
    print(f"Écart de MAE (baseline - modèle) : intervalle de confiance à 95 % de {low:.2f} à {high:.2f} °C")
    if low > 0:
        print("  -> l'intervalle est entièrement positif : le modèle fait mieux que la baseline de façon nette.")
    else:
        print("  -> l'intervalle contient zéro : on ne peut pas affirmer que le modèle fait mieux.")

    table = monthly_table(prediction, actual, baseline, times)
    print("\nErreur moyenne par mois (MAE en °C) :")
    print(table.round(2).to_string())

    name = weights.stem
    week_path = plot_week(prediction, actual, baseline, times, "2016-07-01", name)
    month_path = plot_monthly(table, name)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / f"test_{name}.json", "w", encoding="utf-8") as f:
        json.dump(
            {"config": vars(args), "summary": summary, "par_mois": table.round(3).to_dict(orient="index")},
            f, indent=2, ensure_ascii=False,
        )
    print(f"\nGraphiques : {week_path} | {month_path}")


if __name__ == "__main__":
    main()