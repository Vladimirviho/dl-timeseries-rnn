"""Architectures des modèles de prévision."""
from torch import nn

RECURRENT_LAYERS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class RecurrentForecaster(nn.Module):
    """Lit une fenêtre de N heures pas à pas, puis prédit une valeur.

    Formes des tenseurs (lot = nombre de fenêtres traitées ensemble) :
        entrée                  (lot, 48, nb_variables)  48 heures
        couche récurrente       (lot, 48, hidden)        un état caché par heure
        dernier pas de temps    (lot, hidden)            le résumé de toute la fenêtre
        couche linéaire         (lot, 1) -> (lot,)       la valeur prédite
    """

    def __init__(self, kind="rnn", input_size=1, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.recurrent = RECURRENT_LAYERS[kind](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # la première dimension est le lot
        )
        # Dropout : pendant l'entraînement, éteint au hasard une part des valeurs du résumé,
        # pour que la couche finale ne s'appuie pas sur quelques valeurs seulement.
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.recurrent(x)  # le second élément est l'état final (inutile ici)
        last_step = output[:, -1, :]   # état caché après avoir lu toute la fenêtre
        return self.head(self.dropout(last_step)).squeeze(-1)


def build_model(kind, input_size=1, hidden_size=64, num_layers=1, dropout=0.0):
    return RecurrentForecaster(kind, input_size, hidden_size, num_layers, dropout)


def count_parameters(model):
    """Nombre de paramètres entraînables (les « boutons » du réseau)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)