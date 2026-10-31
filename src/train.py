"""Entraînement d'un RNN, LSTM ou GRU pour prévoir la température.

Exemples (depuis la racine du dépôt) :
    python -m src.train --model rnn --epochs 15
    python -m src.train --model lstm --epochs 30 --features T,p,rh,wv --dropout 0.3 --patience 5
"""
import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import ROOT
from src.models import build_model, count_parameters
from src.utils import get_device, set_seed
from src.windows import prepare_windows

FIG_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"
MODELS_DIR = ROOT / "models"

# Variables météo utilisables en entrée (nom court -> colonne du fichier).
FEATURES = {
    "T": "T (degC)",    # température (toujours présente : c'est la variable à prévoir)
    "p": "p (mbar)",    # pression atmosphérique
    "rh": "rh (%)",     # humidité relative
    "wv": "wv (m/s)",   # vitesse du vent
    "sh": "sh (g/kg)",  # humidité spécifique
}


def make_loader(part, batch_size, shuffle, seed=None):
    """Transforme les fenêtres (tableaux NumPy) en lots utilisables par PyTorch."""
    dataset = TensorDataset(torch.from_numpy(part["x"]), torch.from_numpy(part["y"]))
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def run_epoch(model, loader, criterion, device, std, optimizer=None):
    """Une passe complète sur les données.

    Avec un optimiseur : entraînement. Sans optimiseur : évaluation (sans apprentissage).
    Retourne la perte (sur les valeurs normalisées) et les erreurs MAE et RMSE en °C.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, abs_error, sq_error, count = 0.0, 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            predictions = model(x)
            loss = criterion(predictions, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                # Limite la taille des corrections : évite qu'elles « explosent » dans un RNN.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            diff = predictions.detach() - y
            total_loss += loss.item() * len(y)
            abs_error += diff.abs().sum().item()
            sq_error += (diff ** 2).sum().item()
            count += len(y)
    return {
        "loss": total_loss / count,
        "mae": abs_error / count * std,               # retour en °C
        "rmse": (sq_error / count) ** 0.5 * std,
    }


def fit(model, train_loader, val_loader, epochs, lr, device, std, save_path, patience=0):
    """Entraîne le modèle et conserve les poids de la meilleure epoch (MAE de validation minimale).

    Si patience > 0, l'entraînement s'arrête dès que la MAE de validation n'a pas
    progressé pendant `patience` epochs de suite (arrêt anticipé).
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {key: [] for key in ("train_loss", "val_loss", "train_mae", "val_mae", "val_rmse")}
    best_mae, best_epoch, stale_epochs = float("inf"), 0, 0

    for epoch in range(1, epochs + 1):
        start = time.time()
        train = run_epoch(model, train_loader, criterion, device, std, optimizer)
        val = run_epoch(model, val_loader, criterion, device, std)

        history["train_loss"].append(train["loss"])
        history["val_loss"].append(val["loss"])
        history["train_mae"].append(train["mae"])
        history["val_mae"].append(val["mae"])
        history["val_rmse"].append(val["rmse"])

        improved = val["mae"] < best_mae
        if improved:
            best_mae, best_epoch, stale_epochs = val["mae"], epoch, 0
            torch.save(model.state_dict(), save_path)
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:>2}/{epochs} | "
            f"train MAE {train['mae']:.2f} °C | "
            f"val MAE {val['mae']:.2f} °C (RMSE {val['rmse']:.2f}) | "
            f"{time.time() - start:.1f}s{' *' if improved else ''}"
        )

        if patience and stale_epochs >= patience:
            print(f"Arrêt anticipé : pas d'amélioration depuis {patience} epochs.")
            break

    history["best_epoch"] = best_epoch
    return history


def plot_history(history, baseline_mae, name):
    """Courbes d'apprentissage : perte (à gauche), erreur en °C comparée à la baseline (à droite)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, history["train_loss"], label="entraînement")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set_title("Perte (valeurs normalisées)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()

    ax_mae.plot(epochs, history["train_mae"], label="entraînement")
    ax_mae.plot(epochs, history["val_mae"], label="validation")
    ax_mae.axhline(baseline_mae, color="gray", linestyle="--", label="baseline « dernière valeur » (validation)")
    ax_mae.set_title("Erreur moyenne MAE (°C)")
    ax_mae.set_xlabel("Epoch")
    ax_mae.legend()

    fig.suptitle(f"Courbes d'apprentissage : {name}")
    fig.tight_layout()
    path = FIG_DIR / f"courbes_{name}.png"
    try:
        fig.savefig(path, dpi=120)
    except OSError as error:
        print(f"\n[ATTENTION] Impossible d'enregistrer {path} : {error}")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Entraînement d'un modèle récurrent")
    parser.add_argument("--model", choices=["rnn", "lstm", "gru"], default="rnn")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64, help="taille de l'état caché")
    parser.add_argument("--layers", type=int, default=1, help="nombre de couches récurrentes")
    parser.add_argument("--input-len", type=int, default=48, help="heures d'historique")
    parser.add_argument("--horizon", type=int, default=24, help="heures à l'avance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="part des valeurs du résumé éteintes à l'entraînement (0 = aucune)")
    parser.add_argument("--patience", type=int, default=0,
                        help="arrêt anticipé après N epochs sans progrès (0 = désactivé)")
    parser.add_argument(
        "--features", default="T",
        help="variables d'entrée séparées par des virgules, parmi : " + ", ".join(FEATURES),
    )
    args = parser.parse_args()

    # La température est toujours la première variable (la baseline lit la colonne 0).
    keys = [key.strip() for key in args.features.split(",")]
    unknown = [key for key in keys if key not in FEATURES]
    if unknown:
        parser.error(f"variables inconnues : {unknown} (choix possibles : {list(FEATURES)})")
    keys = ["T"] + [key for key in keys if key != "T"]
    columns = tuple(FEATURES[key] for key in keys)

    set_seed(args.seed)
    device = get_device()
    print(f"Périphérique : {device}")
    print(f"Variables d'entrée : {', '.join(columns)}")

    data = prepare_windows(columns=columns, input_len=args.input_len, horizon=args.horizon)
    std = data["target_std"]
    train_loader = make_loader(data["train"], args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(data["val"], args.batch_size, shuffle=False)
    print(
        f"Fenêtres : {len(data['train']['y'])} entraînement, {len(data['val']['y'])} validation "
        f"({args.input_len} h d'historique pour prédire +{args.horizon} h)"
    )

    # Baseline à battre : prédire que la température restera celle de la dernière heure connue.
    baseline_mae = float(np.abs(data["val"]["x"][:, -1, 0] - data["val"]["y"]).mean() * std)
    print(f"Baseline « dernière valeur » (validation) : MAE {baseline_mae:.2f} °C")

    model = build_model(
        args.model, input_size=len(columns), hidden_size=args.hidden,
        num_layers=args.layers, dropout=args.dropout,
    ).to(device)
    print(f"Modèle : {args.model} ({count_parameters(model)} paramètres entraînables)\n")

    name = f"{args.model}_{args.horizon}h_{args.input_len}l_{args.hidden}u"
    if keys != ["T"]:
        name += "_" + "-".join(keys)
    if args.dropout > 0:
        name += f"_d{round(args.dropout * 100)}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{name}.pt"

    history = fit(model, train_loader, val_loader, args.epochs, args.lr, device, std, model_path,
                  patience=args.patience)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "baseline_val_mae": baseline_mae, "history": history}, f, indent=2)
    path = plot_history(history, baseline_mae, name)

    best = history["best_epoch"]
    best_mae = history["val_mae"][best - 1]
    gain = (baseline_mae - best_mae) / baseline_mae * 100
    print(f"\nMeilleure epoch : {best} (MAE de validation {best_mae:.2f} °C)")
    print(f"Baseline : {baseline_mae:.2f} °C | écart avec la baseline : {gain:+.1f} % d'erreur en moins")
    print(f"Courbes : {path}")
    print(f"Meilleur modèle : {model_path}")


if __name__ == "__main__":
    main()