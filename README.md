\# dl-timeseries-rnn : prévision de la température à +24 h avec RNN, LSTM et GRU (PyTorch)



Projet d'apprentissage du deep learning sur les séries temporelles (semaine 14). On prévoit la température 24 heures à l'avance à partir des 48 dernières heures de mesures météo, en comparant des baselines simples à trois réseaux récurrents (RNN, LSTM, GRU).



\## Résultat principal (jeu de test)



| Modèle | MAE | RMSE |

|---|---|---|

| Baseline : dernière valeur connue | 2,49 °C | 3,24 °C |

| \*\*LSTM, 32 unités, 4 variables (4 897 paramètres)\*\* | \*\*2,28 °C\*\* | \*\*2,90 °C\*\* |



Gain de MAE : \*\*8,6 %\*\*. Intervalle de confiance à 95 % de l'écart de MAE (rééchantillonnage par blocs d'une semaine) : de 0,13 à 0,31 °C, donc le gain est réel mais modeste.



\## Données



\- \*\*Source\*\* : relevés météo de la station du Max Planck Institute for Biogeochemistry à Iéna (Allemagne), 2009-2016, un relevé toutes les 10 minutes, 14 variables. Fichier `jena\_climate\_2009\_2016.csv`, distribué avec les exemples TensorFlow.

\- Les données ne sont pas versionnées dans ce dépôt (voir « Utilisation »).



\### Préparation



| Étape | Résultat |

|---|---|

| Lecture | 420 551 lignes ; 327 dates en double supprimées |

| Valeurs invalides | 38 valeurs `-9999` (colonnes de vitesse du vent) remplacées |

| Passage à l'heure | moyenne des 6 relevés de chaque heure : 70 129 heures |

| Heures manquantes | 88, comblées avec la dernière valeur connue ; la plus longue lacune dure 73 heures (à partir du 25/10/2016) |

| Découpage chronologique 70 / 15 / 15 | entraînement du 01/01/2009 au 08/08/2014, validation jusqu'au 20/10/2015, test jusqu'au 01/01/2017 |

| Normalisation | moyenne et écart-type de l'entraînement seul (pas de fuite de données) |



Les fenêtres qui contiennent une heure comblée sont exclues de l'entraînement, de la validation et du test : sinon, la lacune de 73 heures donnerait une température artificiellement constante et une baseline faussement parfaite.



!\[Série de température](reports/figures/serie\_temperature.png)

!\[Saisonnalité](reports/figures/saisonnalite.png)

!\[Découpage chronologique](reports/figures/decoupage\_chronologique.png)



\## Protocole



\- \*\*Tâche\*\* : 48 heures d'historique pour prédire la température 24 heures après la dernière heure connue.

\- \*\*Fenêtres\*\* : 49 019 pour l'entraînement, 10 433 pour la validation, 10 376 pour le test.

\- \*\*Baseline\*\* : prédire la dernière température connue (à +24 h, c'est aussi la température de la même heure la veille).

\- \*\*Entraînement\*\* : perte MSE, optimiseur Adam (taux 0,001), lots de 64, écrêtage des gradients à 1, choix de la meilleure epoch sur la MAE de validation, arrêt anticipé possible, graine 42, processeur uniquement.

\- \*\*Mesures\*\* : MAE et RMSE en °C.



\### Baselines selon l'horizon (jeu de test)



| Horizon | Moyenne de l'entraînement | Dernière valeur | Même heure hier | Moyenne des 24 h |

|---|---|---|---|---|

| +1 h | 6,40 | \*\*0,65\*\* | 2,49 | 2,49 |

| +6 h | 6,40 | 3,01 | \*\*2,49\*\* | 2,74 |

| +24 h | 6,40 | \*\*2,49\*\* | \*\*2,49\*\* | 3,36 |



(MAE en °C.) La meilleure baseline change avec l'horizon : à court terme l'inertie de la température domine, à plus long terme c'est le cycle de la journée. À +24 h, « dernière valeur » et « même heure hier » sont la même chose.



\## Les essais (jeu de validation, MAE en °C)



Baseline de validation : 2,59 °C.



| Modèle | Variables en entrée | Meilleure MAE | Epoch |

|---|---|---|---|

| RNN simple (4 353 paramètres) | température | 2,52 | 11 |

| LSTM (17 217) | température | 2,50 | 8 |

| GRU (12 929) | température | 2,49 | 11 |

| GRU (13 505) | T, pression, humidité, vent | 2,40 | 4 |

| LSTM (17 985) | T, pression, humidité, vent | 2,37 | 6 |

| LSTM (17 985) avec Dropout 0,3 | T, pression, humidité, vent | 2,41 | 6 |

| \*\*LSTM 32 unités (4 897)\*\* | \*\*T, pression, humidité, vent\*\* | \*\*2,38\*\* | \*\*5\*\* |



Ce que ces essais ont montré :



1\. \*\*Avec la température seule, les trois architectures sont indiscernables\*\* (écart de 0,03 °C, du même ordre que les oscillations d'une epoch à l'autre) : la mémoire du réseau n'était pas le facteur limitant.

2\. \*\*Ajouter la pression, l'humidité et le vent aide\*\* (environ 0,1 °C de MAE en moins) : c'est l'information donnée au modèle qui comptait.

3\. \*\*Un surapprentissage net apparaît après l'epoch 5 ou 6\*\* (le LSTM à 64 unités passe de 2,37 °C à 2,67 °C en validation, pire que la baseline à l'epoch 15). On conserve donc la meilleure epoch.

4\. \*\*Le Dropout et la réduction de la taille n'ont pas amélioré la validation.\*\* On retient le LSTM à 32 unités : performance équivalente avec 3,7 fois moins de paramètres.



!\[Courbes du modèle final](reports/figures/courbes\_lstm\_24h\_48l\_32u\_T-p-rh-wv.png)



\## Résultats sur le test, par mois



| Mois | Modèle | Baseline | | Mois | Modèle | Baseline |

|---|---|---|---|---|---|---|

| Janvier | 2,67 | 3,00 | | Juillet | 2,32 | 2,91 |

| Février | 2,59 | 2,94 | | Août | 2,34 | 2,76 |

| Mars | 1,78 | 1,79 | | Septembre | 1,56 | 2,02 |

| Avril | 2,12 | 2,43 | | Octobre | 1,94 | 1,93 |

| Mai | 2,38 | 2,53 | | Novembre | 2,54 | 2,63 |

| Juin | 1,84 | 2,02 | | Décembre | 2,66 | 2,70 |



Le gain est le plus marqué de juillet à septembre et en janvier-février, et nul en mars et en octobre. Chaque mois n'apparaît qu'une ou deux fois dans le jeu de test : ces écarts sont indicatifs.



!\[Une semaine de juillet 2016](reports/figures/semaine\_test\_lstm\_24h\_48l\_32u\_T-p-rh-wv.png)

!\[Erreur par mois](reports/figures/erreur\_par\_mois\_lstm\_24h\_48l\_32u\_T-p-rh-wv.png)



\## Limites



\- Une seule graine aléatoire par configuration : les écarts de moins de 0,05 °C ne sont pas interprétables.

\- Le jeu de test couvre un an et demi, donc l'analyse mensuelle est indicative.

\- Pas de recherche d'hyperparamètres (taux d'apprentissage, longueur de la fenêtre).

\- Aucune comparaison avec une prévision météo professionnelle : seule une baseline simple est battue.

\- L'explication « le plafond vient de l'incertitude de la météo à +24 h avec ces variables » est une hypothèse, pas un résultat démontré.

\- L'apport de chaque variable prise séparément n'est pas documenté ici.



\## Structure du dépôt



```

dl-timeseries-rnn/

├── README.md

├── requirements.txt

├── src/

│   ├── dataset.py       lecture, nettoyage, passage à l'heure, découpage, normalisation

│   ├── explore.py       exploration et figures

│   ├── baselines.py     méthodes simples à +1 h, +6 h, +24 h

│   ├── windows.py       fenêtres glissantes, exclusion des heures comblées

│   ├── check\_windows.py vérification des fenêtres et baseline associée

│   ├── models.py        RNN, LSTM, GRU

│   ├── train.py         entraînement, arrêt anticipé, courbes

│   ├── evaluate.py      évaluation finale sur le test

│   ├── analysis.py      intervalle de confiance, erreurs par mois, graphiques

│   └── utils.py         graine aléatoire, périphérique

├── models/              poids du modèle final

└── reports/

&#x20;   ├── figures/

&#x20;   └── metrics/

```



\## Utilisation



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt



\# Données (non versionnées)

mkdir data

curl.exe -L -o data\\jena\_climate\_2009\_2016.csv.zip https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena\_climate\_2009\_2016.csv.zip

Expand-Archive data\\jena\_climate\_2009\_2016.csv.zip -DestinationPath data



python -m src.explore

python -m src.baselines

python -m src.check\_windows

python -m src.train --model lstm --epochs 30 --features T,p,rh,wv --hidden 32 --patience 5

python -m src.evaluate --model lstm --hidden 32 --features T,p,rh,wv --weights models/lstm\_24h\_48l\_32u\_T-p-rh-wv.pt

```



Le modèle final est versionné : la dernière commande fonctionne sans réentraîner.



\## Auteur



Vladimir : \[GitHub](https://github.com/Vladimirviho)

