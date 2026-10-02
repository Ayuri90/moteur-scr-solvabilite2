# Calcul complet du SCR d'une compagnie d'assurance composite

Implémentation complète de la formule standard Solvabilité II, d'un modèle ALM stochastique,
d'une étude d'impact de la réforme applicable en 2027, d'un modèle interne partiel non-vie et
d'un ORSA, sur une compagnie composite fictive calibrée au 31 décembre 2025.

Environ 3 700 lignes de Python, 60 tests automatisés, trois documents compilés.

## Résultats principaux

| Indicateur | Valeur |
|---|---|
| Actif total | 11 000 M€ |
| Best estimate total | 8 498 M€ |
| SCR, formule standard | 997 M€ |
| SCR après modèle ALM stochastique | 831 M€ |
| Ratio de couverture | 233 % |
| SCR sous le régime 2027 | 896 M€ |
| Ratio ORSA, scénario central à 5 ans | 220 % |

La décomposition du SCR a été rapprochée de celle d'un groupe réel publiant sous formule
standard : risque de marché, souscription vie et risque opérationnel tombent dans les mêmes
ordres de grandeur, ce qui constitue la première validation du modèle.

## Démarche

Le projet est organisé en cinq versions, chacune levant une limite de la précédente.

| Version | Contenu | Script |
|---|---|---|
| V1 | formule standard complète : marché, contrepartie, vie, santé, non-vie, opérationnel, BSCR, ajustements, SCR, MCR | `calculer_scr.py` |
| V2 | générateur de scénarios économiques, projection actif-passif du fonds euro, valeur temps des options, capacité d'absorption explicite | `calculer_v2.py` |
| V3 | étude d'impact du Règlement délégué (UE) 2026/269, sous forme d'un pont du ratio étape par étape | `calculer_v3.py` |
| V4 | modèle interne partiel non-vie : bootstrap, copules, paramètres propres à l'entreprise, comparaison avec le code CIMA | `calculer_v4.py` |
| V5 | ORSA : projection à cinq ans, six scénarios adverses, stress inversés, besoin global de solvabilité | `calculer_v5.py` |

## Installation et exécution

Python 3.11 ou plus récent.

```bash
pip install -r requirements.txt

python generer_donnees.py     # construit les données de la compagnie fictive
python calculer_scr.py        # V1
python calculer_v2.py         # V2
python calculer_v3.py         # V3
python calculer_v4.py         # V4
python calculer_v5.py         # V5

python -m pytest -q           # 60 tests, à lancer après generer_donnees.py
```

Les données ne sont pas versionnées : `generer_donnees.py` les reconstruit à l'identique en
quelques secondes, la graine aléatoire étant fixée.

Pour produire les documents :

```bash
python rapport/generer_rapport.py && cd rapport && pdflatex rapport_scr.tex
python pedagogie/generer_chiffres.py && cd pedagogie && pdflatex cahier_calculs.tex
cd docs && pdflatex specifications_fonctions.tex
```

## Organisation du dépôt

```
moteur-scr-solvabilite2/
├── config/              hypothèses et paramètres réglementaires, en YAML commenté
├── scr_data/            fabrication des données : courbe, mortalité, actifs, passifs, triangles
├── scr/                 moteur de calcul : provisions, modules de risque, ESG, ALM, ORSA
├── scripts/             extraction des fichiers officiels (EIOPA, tables de mortalité)
├── tests/               60 tests automatisés
├── rapport/             rapport de synthèse, sources LaTeX
├── pedagogie/           cahier de calculs manuels, sources LaTeX
├── docs/                documentation du code et spécifications fonctionnelles
├── livrables/           les trois documents compilés, à lire sans rien installer
├── sources/             fichiers officiels à déposer soi-même (non versionnés)
├── donnees/             données engendrées (non versionnées)
└── resultats/           sorties de calcul (non versionnées)
```

## Documents

| Document | Contenu |
|---|---|
| [`rapport_scr.pdf`](livrables/rapport_scr.pdf) | 18 pages : données, puis pour chaque version la question posée, les méthodes, les hypothèses, les analyses et l'apport. Deux annexes : formulaire et organisation du code |
| [`cahier_calculs.pdf`](livrables/cahier_calculs.pdf) | 12 pages : chaque module refait à la main sur les données du projet, avec les étapes intermédiaires, les pièges fréquents et des exercices |
| [`specifications_fonctions.pdf`](livrables/specifications_fonctions.pdf) | 14 pages : pour chaque fonction, sa raison d'être et la formule appliquée, avec un index des articles du règlement vers les fonctions |
| [`documentation_code.md`](docs/documentation_code.md) | documentation technique : chaque module, les pièges rencontrés et corrigés, les recettes d'extension |

Aucun chiffre de ces documents n'est saisi à la main : les scripts `generer_rapport.py` et
`generer_chiffres.py` lisent les sorties du modèle et engendrent les valeurs et les tableaux
LaTeX. Relancer les calculs suffit à mettre les documents à jour.

## Données et hypothèses

La compagnie est fictive, mais calibrée sur des sources publiques réelles.

| Élément | Source |
|---|---|
| Masses de bilan et décomposition du SCR | SFCR 2025 d'un groupe mutualiste, facteur d'échelle d'environ un quart |
| Allocation d'actifs | statistiques de placements de l'ACPR, après mise en transparence |
| Courbe des taux et facteurs de choc | fichier EIOPA du 31/12/2025, UFR 3,30 %, alpha 0,073632, CRA 10 pb |
| Ajustement symétrique | 7,90 %, valeur EIOPA du 31/12/2025 |
| Mortalité des garanties décès | tables réglementaires TH et TF 00-02 |
| Mortalité des rentes | tables TGH05 et TGF05, utilisées en générationnel |

Les fichiers officiels ne sont pas redistribués ici. Voir `sources/LISEZMOI.md` pour les
télécharger et lancer les deux scripts d'extraction. Sans eux, le projet reste exécutable avec
deux approximations documentées : reconstruction de la courbe par Smith-Wilson et mortalité
paramétrique.

Chaque paramètre du fichier d'hypothèses porte une étiquette indiquant sa nature : donnée réelle,
ordre de grandeur de marché, valeur vérifiée dans le règlement, valeur restant à confronter au
texte, ou choix de modélisation.

## Ce que le projet produit

- un inventaire d'actifs de 290 lignes, valorisées titre par titre et rechoquées sous chaque
  scénario ;
- 1 900 model points de passif vie et santé, cinq triangles de règlements sur dix ans ;
- un bilan prudentiel au format proche du S.02.01 et un tableau S.25.01 ;
- un classeur Excel de restitution, les figures du rapport et les sorties CSV de chaque version.

## Limites assumées

- Les tables de chocs du régime 2027, les probabilités de défaut, les seuils de concentration et
  les facteurs du MCR restent à confronter au texte réglementaire.
- Les volatilités du générateur de scénarios sont des ordres de grandeur, et non le résultat d'un
  calibrage sur prix de swaptions et d'options.
- Les modules catastrophe non-vie et santé sont paramétriques ; les annexes du règlement, avec
  leur découpage par zones, restent à implémenter.
- Chain Ladder sans facteur de queue sous-estime les réserves des branches longues, ce qui est
  mesuré et commenté dans le rapport.
- En ORSA, le SCR projeté est mis à l'échelle de volumes plutôt que recalculé, et aucune action
  du management n'est déclenchée au franchissement des seuils.

## Licence

MIT, voir `LICENSE`. Les fichiers officiels de l'EIOPA et les tables de mortalité restent la
propriété de leurs auteurs et ne sont pas redistribués.
