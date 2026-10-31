"""Exploration de la série météo de Jena : qualité, saisonnalité, découpage chronologique.

À lancer depuis la racine du dépôt :  python -m src.explore
Les figures sont enregistrées dans reports/figures/.
"""
import matplotlib.pyplot as plt

from src.dataset import (
    DATA_PATH,
    ROOT,
    TARGET,
    chronological_split,
    clean_hourly,
    load_raw,
    normalize,
    unreliable_mask,
)

FIG_DIR = ROOT / "reports" / "figures"


def describe_split(name, part, unreliable):
    n_bad = int(unreliable.loc[part.index].sum())
    print(
        f"  {name:<11} {len(part):>6} heures  du {part.index.min():%d/%m/%Y} au {part.index.max():%d/%m/%Y}"
        f"  (dont {n_bad} peu fiables)"
    )


def save_figure(fig, name):
    """Enregistre une figure ; en cas d'échec, affiche l'erreur et continue."""
    path = FIG_DIR / name
    try:
        fig.savefig(path, dpi=120)
    except OSError as error:
        print(f"\n[ATTENTION] Impossible d'enregistrer {path} : {error}\n")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Données brutes.
    raw, n_duplicates = load_raw(DATA_PATH)
    print(f"Relevés bruts : {len(raw)} (un toutes les 10 minutes)")
    print(f"Dates en double supprimées : {n_duplicates}")
    print(f"Période       : du {raw.index.min():%d/%m/%Y} au {raw.index.max():%d/%m/%Y}")
    print(f"Variables     : {raw.shape[1]}")
    print("\nValeurs minimale et maximale de chaque variable (avant nettoyage) :")
    print(raw.agg(["min", "max"]).T.to_string())

    # 2. Nettoyage et passage à une mesure par heure.
    hourly, report, filled = clean_hourly(raw)
    print("\nNettoyage :")
    print(f"  valeurs invalides (-9999) remplacées : {report['valeurs_invalides']}")
    print(f"  heures entièrement manquantes        : {report['heures_manquantes']}")
    print(f"  valeurs comblées au total            : {report['valeurs_comblees']}")
    if report["plus_longue_lacune_heures"] > 0:
        print(
            f"  plus longue lacune                   : {report['plus_longue_lacune_heures']} heures "
            f"à partir du {report['debut_plus_longue_lacune']:%d/%m/%Y %H:%M}"
        )
    print(f"  série horaire : {len(hourly)} heures")

    temperature = hourly[TARGET]

    # 3. Figure 1 : la série à trois échelles.
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].plot(temperature.index, temperature.values, linewidth=0.4)
    axes[0].set_title("Température horaire, 2009-2016")
    axes[1].plot(temperature["2012-07-01":"2012-07-31"])
    axes[1].set_title("Juillet 2012 : le cycle de chaque jour")
    axes[2].plot(temperature["2012-07-01":"2012-07-07"])
    axes[2].set_title("Première semaine de juillet 2012")
    for ax in axes:
        ax.set_ylabel("°C")
    fig.tight_layout()
    save_figure(fig, "serie_temperature.png")

    # 4. Figure 2 : la saisonnalité.
    by_hour = temperature.groupby(temperature.index.hour).mean()
    by_month = temperature.groupby(temperature.index.month).mean()
    fig, (ax_hour, ax_month) = plt.subplots(1, 2, figsize=(11, 4))
    ax_hour.plot(by_hour.index, by_hour.values, marker="o")
    ax_hour.set_title("Température moyenne selon l'heure de la journée")
    ax_hour.set_xlabel("Heure")
    ax_hour.set_ylabel("°C")
    ax_month.plot(by_month.index, by_month.values, marker="o")
    ax_month.set_title("Température moyenne selon le mois")
    ax_month.set_xlabel("Mois")
    ax_month.set_ylabel("°C")
    fig.tight_layout()
    save_figure(fig, "saisonnalite.png")

    print(
        f"\nHeure la plus chaude en moyenne : {by_hour.idxmax()} h ({by_hour.max():.1f} °C) | "
        f"la plus froide : {by_hour.idxmin()} h ({by_hour.min():.1f} °C)"
    )
    print(
        f"Mois le plus chaud : {by_month.idxmax()} ({by_month.max():.1f} °C) | "
        f"le plus froid : {by_month.idxmin()} ({by_month.min():.1f} °C)"
    )

    # 5. Découpage chronologique et normalisation.
    train, val, test = chronological_split(hourly)
    unreliable = unreliable_mask(filled)
    print("\nDécoupage chronologique (peu fiables : heure comblée ou à moins de 48 h d'une lacune) :")
    describe_split("entraînement", train, unreliable)
    describe_split("validation", val, unreliable)
    describe_split("test", test, unreliable)

    train_n, val_n, test_n, mean, std = normalize(train, val, test)
    print(
        f"\nTempérature moyenne : entraînement {train[TARGET].mean():.1f} °C | "
        f"validation {val[TARGET].mean():.1f} °C | test {test[TARGET].mean():.1f} °C"
    )
    print(f"Statistiques de l'entraînement pour {TARGET} : moyenne {mean[TARGET]:.2f}, écart-type {std[TARGET]:.2f}")
    print("Moyenne de la température normalisée :")
    print(f"  entraînement {train_n[TARGET].mean():+.2f} | validation {val_n[TARGET].mean():+.2f} | test {test_n[TARGET].mean():+.2f}")

    # 6. Figure 3 : le découpage (moyenne journalière pour la lisibilité).
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, part in (("entraînement", train), ("validation", val), ("test", test)):
        daily = part[TARGET].resample("D").mean()
        ax.plot(daily.index, daily.values, linewidth=0.8, label=name)
    ax.set_title("Découpage chronologique (moyenne journalière)")
    ax.set_ylabel("°C")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "decoupage_chronologique.png")

    print(f"\nDossier des figures : {FIG_DIR}")


if __name__ == "__main__":
    main()