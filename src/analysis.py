"""Analyse des prévisions : erreurs, intervalle de confiance, erreurs par mois, graphiques.

Ce module n'utilise ni PyTorch ni le réseau : il reçoit seulement des prévisions
(en °C) et les compare aux valeurs réelles.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset import ROOT

FIG_DIR = ROOT / "reports" / "figures"


def block_bootstrap_ci(diff, block=168, n_boot=2000, seed=0):
    """Intervalle de confiance à 95 % de la moyenne de `diff`, par rééchantillonnage en blocs.

    Les erreurs de deux heures voisines sont liées. On rééchantillonne donc des
    blocs de `block` heures consécutives (168 h = une semaine) et non des heures isolées.
    """
    rng = np.random.default_rng(seed)
    n = len(diff)
    block = min(block, n)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([diff[s:s + block] for s in starts])[:n]
        means[i] = sample.mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize(prediction, actual, baseline):
    """MAE et RMSE (°C) du modèle et de la baseline, et gain avec son intervalle de confiance."""
    err_model = prediction - actual
    err_base = baseline - actual
    mae_model = float(np.abs(err_model).mean())
    mae_base = float(np.abs(err_base).mean())
    # Positif quand le modèle se trompe moins que la baseline.
    gain_per_hour = np.abs(err_base) - np.abs(err_model)
    low, high = block_bootstrap_ci(gain_per_hour)
    return {
        "n_windows": int(len(actual)),
        "mae_model": mae_model,
        "mae_baseline": mae_base,
        "rmse_model": float(np.sqrt((err_model ** 2).mean())),
        "rmse_baseline": float(np.sqrt((err_base ** 2).mean())),
        "gain_percent": (mae_base - mae_model) / mae_base * 100,
        "gain_mae_ci95": [low, high],  # écart de MAE (baseline - modèle), en °C
    }


def monthly_table(prediction, actual, baseline, times):
    """MAE du modèle et de la baseline pour chaque mois de l'année."""
    frame = pd.DataFrame(
        {
            "modele": np.abs(prediction - actual),
            "baseline": np.abs(baseline - actual),
        },
        index=pd.DatetimeIndex(times),
    )
    table = frame.groupby(frame.index.month).agg(["mean", "count"])
    return pd.DataFrame(
        {
            "fenetres": table[("modele", "count")].astype(int),
            "mae_modele": table[("modele", "mean")],
            "mae_baseline": table[("baseline", "mean")],
        }
    )


def _save(fig, name):
    path = FIG_DIR / name
    try:
        fig.savefig(path, dpi=120)
    except OSError as error:
        print(f"[ATTENTION] Impossible d'enregistrer {path} : {error}")
    plt.close(fig)
    return path


def plot_week(prediction, actual, baseline, times, start, name):
    """Une semaine de températures réelles, prévues par le modèle et par la baseline."""
    times = pd.DatetimeIndex(times)
    begin = pd.Timestamp(start)
    mask = (times >= begin) & (times < begin + pd.Timedelta(days=7))
    if not mask.any():
        return None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times[mask], actual[mask], label="réel", color="black")
    ax.plot(times[mask], prediction[mask], label="modèle", color="tab:red")
    ax.plot(times[mask], baseline[mask], label="baseline (dernière valeur)", color="gray", linestyle="--")
    ax.set_ylabel("°C")
    ax.set_title(f"Prévision à +24 h, semaine du {begin:%d/%m/%Y} (jeu de test)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, f"semaine_test_{name}.png")


def plot_monthly(table, name):
    """Erreur moyenne par mois : modèle contre baseline."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["mae_modele"], width=0.4, label="modèle")
    ax.bar(x + 0.2, table["mae_baseline"], width=0.4, label="baseline", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_xlabel("Mois")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Erreur moyenne par mois (jeu de test)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, f"erreur_par_mois_{name}.png")