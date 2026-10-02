# Documentation du code — projet SCR

Ce document décrit chaque fichier, chaque classe et chaque fonction du projet, ce qu'ils
prennent en entrée, ce qu'ils produisent, et les subtilités à connaître pour les défendre.

---

> **Deux documents complémentaires.** Celui-ci décrit *ce que fait* chaque élément et comment les
> pièces s'emboîtent. `specifications_fonctions.tex` répond aux deux autres questions : *pourquoi*
> chaque fonction existe et *quelle formule* elle applique, avec un index reliant les articles du
> règlement délégué aux fonctions.

## Table des matières

1. [Vue d'ensemble et conventions](#1-vue-densemble-et-conventions)
2. [Fichiers de configuration](#2-fichiers-de-configuration)
3. [Package `scr_data` — génération des données](#3-package-scr_data--génération-des-données)
4. [`generer_donnees.py` — orchestration des données](#4-generer_donneespy--orchestration-des-données)
5. [Package `scr` — moteur de calcul](#5-package-scr--moteur-de-calcul)
6. [Scripts de calcul `calculer_*.py`](#6-scripts-de-calcul-calculer_py)
7. [Rapport](#7-rapport)
8. [Tests](#8-tests)
9. [Pièges rencontrés et corrigés](#9-pièges-rencontrés-et-corrigés)
10. [Recettes d'extension](#10-recettes-dextension)

---

## 1. Vue d'ensemble et conventions

### Chaîne de traitement

```
config/*.yaml
     │
     ▼
generer_donnees.py ──► donnees/*.csv        (inventaire, model points, triangles…)
     │
     ▼
calculer_scr.py    ──► resultats/*.csv      V1 : formule standard
     │
     ▼
calculer_v2.py     ──► resultats/v2_*.csv   V2 : ESG + ALM + LAC TP
     │
     ├─► calculer_v3.py ──► v3_*.csv        V3 : régime 2027
     ├─► calculer_v4.py ──► v4_*.csv        V4 : modèle interne
     └─► calculer_v5.py ──► v5_*.csv        V5 : ORSA
                  │
                  ▼
        rapport/generer_rapport.py ──► figures + chiffres.tex ──► rapport_scr.pdf
```

Chaque étage consomme les sorties de l'étage précédent. Les scripts V2 à V5 appellent
directement les fonctions `calculer()` des étages inférieurs avec `ecrire=False`, ce qui évite
de relire les CSV et garantit la cohérence.

### Conventions

| Convention | Détail |
|---|---|
| Unités | montants en millions d'euros, taux en décimal, spreads en points de base |
| CSV | séparateur `;`, décimale `,`, encodage `utf-8-sig` — ouverture directe dans Excel français |
| Aléa | une graine par module, via `generateur(hyp, decalage)`, pour que modifier un module ne déplace pas les tirages des autres |
| Nommage | tout en français, y compris les colonnes, pour coller au vocabulaire réglementaire |
| Étiquetage | chaque hypothèse porte `[MACSF]`, `[ACPR]`, `[VERIFIE]`, `[AVERIFIER]`, `[INDIC]` ou `[CHOIX]` |
| Séparation | `scr_data` fabrique les données, `scr` calcule, les scripts orchestrent et affichent |

---

## 2. Fichiers de configuration

### `config/hypotheses.yaml`

Décrit la compagnie. Sections, dans l'ordre :

| Section | Contenu |
|---|---|
| `general` | nom, date d'évaluation, graine, dossier de sortie |
| `courbe` | source (`eiopa` ou `smith_wilson`), UFR, LLP, point de convergence, CRA, taux swap |
| `bilan_cible` | masses d'actif et de passif visées, servant de cibles de calage |
| `allocation` | poids des neuf classes d'actifs hors UC |
| `obligations`, `actions`, `immobilier`, `monetaire`, `prets_infrastructure` | paramètres de génération de chaque poche |
| `uc` | allocation des supports en unités de compte, pour la transparence |
| `contreparties` | réassureurs, banques, créances |
| `epargne_euro`, `epargne_uc`, `temporaire_deces`, `rentes_viageres` | model points vie : effectifs, PM, TMG, chargements, lois de rachat |
| `mortalite` | source des tables et paramètres de Makeham |
| `non_vie`, `sante_nslt`, `sante_slt` | primes, S/P, cadences, provisions cibles, expositions CAT |
| `frais`, `projection` | frais de gestion, inflation, horizon de projection |
| `esg`, `alm` | V2 : générateur de scénarios et modèle ALM |
| `modele_interne`, `usp`, `cima` | V4 |
| `orsa` | V5 : plan d'affaires, appétence, scénarios, besoin global |

### `config/params_reglementaires.yaml`

Paramètres de la formule standard : chocs de taux, action, immobilier, change, table de spread
par échelon et bande de duration, seuils de concentration, probabilités de défaut, matrices de
corrélation, facteurs du risque opérationnel, du MCR, et taux d'impôt.

### `config/params_2027.yaml`

Uniquement les écarts du nouveau régime : extrapolation (FSP, LLFR, vitesse de convergence),
tables `a` et `b` des chocs de taux avec planchers négatifs, corrélation taux-spread, corridor
de l'ajustement symétrique, facteur de réassurance non proportionnelle, coût du capital et
dégressivité de la marge de risque.

---

## 3. Package `scr_data` — génération des données

### 3.1 `config.py` (49 lignes)

| Élément | Rôle |
|---|---|
| `RACINE` | chemin racine du projet, calculé depuis l'emplacement du fichier ; tous les chemins en dépendent |
| `charger_hypotheses(chemin=None)` | lit le YAML et lance les vérifications ; renvoie un dictionnaire |
| `_verifier(hyp)` | garde-fous : l'allocation d'actifs somme à 1, les répartitions par échelon de crédit somment à 1, les poids pays des souverains somment à 1. Lève une exception explicite sinon |
| `date_evaluation(hyp)` | convertit la date en `Timestamp` |
| `generateur(hyp, decalage)` | renvoie un `numpy.random.Generator` de graine `graine + decalage`. Les décalages utilisés : 100 actifs, 200 passif vie, 300 non-vie, 400 santé SLT |
| `poids_normalises(n, rng, sigma)` | tire `n` poids log-normaux normalisés à 1. Sert partout où il faut répartir un montant sur des lignes de tailles hétérogènes mais réalistes |

### 3.2 `courbe.py` (160 lignes)

| Élément | Rôle |
|---|---|
| `_wilson(t, u, alpha, omega)` | matrice du noyau de Wilson, cœur de l'interpolation Smith-Wilson |
| `SmithWilson.__init__` | construit la matrice de flux des swaps au pair (coupon `r` chaque année, `1+r` à l'échéance), résout `(C W Cᵀ) ζ = p − C μ` pour obtenir les coefficients `ζ` |
| `SmithWilson.facteur_actualisation(t)` | prix zéro-coupon à toute maturité, entière ou non |
| `SmithWilson.forward_instantane(t, h)` | forward instantané par différence finie centrée, utilisé pour le critère de convergence |
| `calibrer_alpha(...)` | cherche le plus petit `alpha ≥ 0,05` respectant la tolérance d'un point de base au point de convergence ; test direct à la borne inférieure, sinon `brentq` |
| `courbe_smith_wilson(hyp_courbe)` | applique le CRA aux taux swap, calibre `alpha`, renvoie la courbe et `alpha` |
| `courbe_eiopa(hyp_courbe)` | lit le fichier officiel EIOPA : localise la cellule portant le nom de la devise, puis cherche la première série de 150 valeurs numériques consécutives sous cette cellule. Robuste aux changements de mise en page |
| `_mettre_en_forme(t, df)` | à partir des facteurs d'actualisation, reconstruit taux spot et forwards à un an |
| `construire_courbe(hyp)` | point d'entrée : aiguille vers l'une des deux sources et renvoie `(courbe, info)` |
| `Actualisation` | objet léger d'actualisation. `df(t, spread)` interpole **en log** les facteurs d'actualisation, ce qui garantit la positivité et la régularité, puis applique un spread additif sur le taux annuel équivalent |

Subtilité : l'interpolation logarithmique compte. Une interpolation linéaire des facteurs
produirait des forwards en dents de scie, visibles dans les projections longues.

### 3.3 `mortalite.py` (74 lignes)

| Élément | Rôle |
|---|---|
| `TableMortalite.__init__` | stocke le vecteur `qx` borné à [0, 1] avec `q(120) = 1` |
| `.makeham(A, B, c)` | table paramétrique : `q_x = 1 − exp(−∫(A + B c^s)ds)`, intégrale résolue analytiquement. Sert de repli quand les tables réglementaires ne sont pas fournies |
| `.depuis_csv(chemin)` | lit un CSV `age;qx` (TH/TF 00-02, TGH/TGF 05) |
| `.ajuster(facteur)` | renvoie une table multipliée par un facteur — utilisé pour les chocs de mortalité et de longévité, et pour l'abattement générationnel des rentiers |
| `.survie(age, n)` | vecteur des `k_p_x` pour `k = 0…n`, complété par des 1 au-delà de la table |
| `.annuite_viagere(age, actu, terme_echu, duree_max)` | valeur actuelle probable d'une rente unitaire |
| `TableGenerationnelle` | table par génération (TGH05, TGF05). Une personne d'âge `x` à la date d'évaluation relève de la génération `année − x`, et c'est cette colonne qui sert à toute sa projection, gains de longévité futurs compris. L'attribut `qx` porte la diagonale, pour les traitements à une dimension |
| `.depuis_csv_long(chemin, annee, nom)` | lit un fichier `generation,age,qx` |
| `._colonne(generation)` | colonne demandée, à défaut la génération disponible la plus proche |
| `.survie(age, n)` | probabilités de survie le long de la génération, avec prolongement par la dernière valeur connue |
| `charger_tables(hyp)` | renvoie le dictionnaire `homme`, `femme`, `rente_homme`, `rente_femme` |
| `_charger_rente(chemin, annee, nom)` | bascule automatiquement en format générationnel si le fichier porte une colonne `generation` |

### 3.4 `actifs.py` (322 lignes)

Produit l'inventaire ligne à ligne. Les fonctions privées fabriquent chaque poche, la fonction
publique assemble.

| Élément | Rôle |
|---|---|
| `prix_obligation(actu, coupon, maturite, spread)` | prix pour 100 de nominal : somme des coupons et du remboursement actualisés au taux sans risque majoré du spread |
| `duration_effective(...)` | duration modifiée calculée par choc symétrique d'un point de base — donc cohérente avec la courbe réelle, et non avec une approximation analytique |
| `_tirer_maturites(...)` | maturités tirées dans une loi gamma, tronquées entre 1 et 30 ans |
| `_coupon(rng, actu, maturite, spread, ecart)` | coupon « historique » : rendement actuel plus un bruit de moyenne négative, arrondi au huitième de point. C'est ce qui crée des titres sous le pair, donc des moins-values latentes réalistes |
| `_lignes_obligataires(...)` | valorise une liste de spécifications, puis met les nominaux à l'échelle pour atteindre la valeur de marché cible |
| `_souverains` | répartit les lignes par pays selon les poids, applique un spread souverain par pays, puis recale la valeur de marché pays par pays |
| `_corporates` | crée un pool d'émetteurs avec un échelon de crédit tiré par émetteur et des tailles log-normales, ce qui produit naturellement de la concentration ; gère les obligations sécurisées |
| `_actions` | trois types : cotées (dont la moitié portées par des émetteurs obligataires déjà existants, pour alimenter le sous-module concentration), private equity, participations stratégiques. Les devises hors euro alimentent le sous-module change |
| `_immobilier` | lignes triées par taille, les plus grosses marquées « usage propre » à hauteur de la part visée |
| `_monetaire` | dépôts bancaires nominatifs (risque de contrepartie) et OPC monétaires mis en transparence (risque de spread) |
| `_prets_infra` | prêts et infrastructure qualifiante, avec respect exact de la part infrastructure en valeur de marché |
| `_uc_transparence` | supports en unités de compte mis en transparence : quatre poches actions, trois obligataires, immobilier, monétaire |
| `generer_inventaire(hyp, actu)` | assemble, ajoute les colonnes manquantes, type les booléens et les entiers nullables, attribue les identifiants `A00001…` |

Point d'attention : les obligations ont des maturités entières et des coupons annuels. C'est une
simplification assumée, qui rend les flux exactement projetables.

### 3.5 `passif_vie.py` (152 lignes)

| Élément | Rôle |
|---|---|
| `_ages`, `_sexes` | tirages d'âge tronqués et de sexe |
| `_taux_par_tranche(anciennete, tranches, cle)` | applique un barème par tranche d'ancienneté : sert aux TMG par génération et aux rachats structurels, y compris le pic fiscal à huit ans |
| `epargne_euro(hyp, rng)` | 600 model points : âge, ancienneté (loi bêta, donc portefeuille plutôt jeune avec une queue longue), PM, TMG, chargement, taux de rachat, taux de PB |
| `epargne_uc(hyp, rng)` | 300 model points, avec versements cumulés et indicateur de garantie plancher décès |
| `temporaire_deces(hyp, tables, rng)` | capitaux sous risque et primes tarifées `q_x × capital × (1 + chargement)`, puis mise à l'échelle pour atteindre le volume de primes visé |
| `_annuite_reversion(...)` | valeur d'une rente de réversion : versée au conjoint survivant après le décès du rentier, calculée sous hypothèse d'indépendance |
| `rentes_viageres(...)` | tire âges, sexes, réversions, calcule pour chaque model point le facteur de rente, puis **cale les arrérages** pour que le BE total égale la cible |
| `generer_passif_vie(hyp, tables, actu)` | renvoie les quatre tables de model points |

### 3.6 `non_vie_sante.py` (162 lignes)

| Élément | Rôle |
|---|---|
| `_parts_cumulees(cadence, facteur_queue)` | convertit une cadence de règlement en parts cumulées de l'ultime, la queue restant à payer au-delà du triangle |
| `reserve_theorique(p, …)` | réserve attendue d'une ligne, avant aléa : sert à calibrer |
| `simuler_triangle(nom, p, …)` | pour chaque année de survenance : primes acquises, ultime bruité (log-normal), incréments bruités (gamma), cumul. Renvoie le triangle long, la table des **ultimes vrais** et le facteur de calage. Avertit si ce facteur sort de [0,85 ; 1,15] |
| `triangle_en_matrice(tri_long)` | pivot long vers matriciel, pour l'export Excel |
| `generer_non_vie(hyp)` | boucle sur les lignes non-vie puis la santé NSLT, assemble triangles, ultimes, volumes et expositions CAT |
| `_volume(...)` | ligne de la table des volumes : primes N et N+1, primes futures, taux de cession, réserve vraie |
| `_expositions_cat(hyp, rng)` | capitaux MRH répartis par zone (loi de Dirichlet concentrée), nombre de véhicules, concentrations incendie, paramètres du traité XL, nombre d'assurés santé |
| `generer_sante_slt(hyp, tables, actu)` | rentes d'incapacité et d'invalidité en cours de service. Invalidité : maintien jusqu'à l'âge de fin de garantie avec surmortalité. Incapacité : maintien exponentiel mensuel. Les rentes sont calées sur le BE cible |

La table des ultimes vrais est l'atout pédagogique du projet : elle permet de mesurer l'erreur
des méthodes de provisionnement, ce qui est impossible sur données réelles.

### 3.7 `bilan.py` (142 lignes)

| Élément | Rôle |
|---|---|
| `generer_contreparties(hyp)` | expositions de type 1 (réassureurs au prorata des provisions cédées, banques) et de type 2 (créances, dont la part échue depuis plus de trois mois) |
| `construire_bilan(...)` | bilan au format proche du S.02.01 : chaque poste d'actif est agrégé depuis l'inventaire, chaque poste de passif porte un statut (`inventaire`, `hypothèse`, `indicatif`, `cible à recalculer`) |
| `controles(...)` | 21 contrôles : masses, poids d'allocation, PM, BE calés, facteurs de calage des triangles, plus des indicateurs (duration moyenne, échelon moyen, plus grande exposition, part en devises, plus-values latentes). La fonction interne `ajouter` gère la tolérance et le statut `OK` / `ALERTE` / `INFO` |

### 3.8 `export.py` (140 lignes)

| Élément | Rôle |
|---|---|
| `exporter_csv(tables, dossier)` | écrit toutes les tables au format français |
| `_entete`, `_style`, `_titre` | mise en forme openpyxl : en-têtes sur fond bleu, police unique, largeurs de colonnes, volets figés |
| `exporter_excel(...)` | classeur de synthèse en quatre onglets : bilan (avec totaux en **formules Excel**, pas en valeurs), allocation, contrôles colorés par statut, courbe des taux |

---

### 3.9 `scripts/extraire_tables_mortalite.py`

Convertit les classeurs officiels en CSV exploitables.

| Élément | Rôle |
|---|---|
| `_qx_depuis_lx(lx)` | `q_x = 1 − l_{x+1}/l_x`, calculé là où les effectifs sont renseignés |
| `extraire_th_tf(chemin)` | les quatre tables du classeur TH-TF 00-02 : TF décès, TF vie, TH décès, TH vie |
| `extraire_generationnelle(chemin, onglet)` | TGH05 ou TGF05 en format long `generation, age, qx` |
| `diagonale(table_longue, annee)` | table à une dimension : pour chaque âge, le `q_x` de la génération vivante à l'année choisie |
| `principal(...)` | écrit les huit fichiers dans `sources/` |

Usage : `python scripts/extraire_tables_mortalite.py TH-TF-00-02.xls TGF05-TGH05.xls 2025`

### 3.10 `scripts/extraire_parametres_eiopa.py`

| Élément | Rôle |
|---|---|
| `facteurs_choc(chemin)` | onglet `Shocks` : facteurs relatifs des articles 166 et 167, pour les 150 maturités |
| `parametres(chemin, devise)` | LLP, point de convergence, UFR, alpha, CRA et ajustement pour volatilité de la devise |
| `principal(...)` | écrit `sources/facteurs_choc_taux.csv` et `sources/parametres_eiopa.csv` |

## 4. `generer_donnees.py` — orchestration des données

| Élément | Rôle |
|---|---|
| `generer(chemin_config, ecrire)` | charge les hypothèses, construit courbe et tables, appelle les quatre générateurs, construit bilan et contrôles, exporte CSV, triangles matriciels et classeur de synthèse |
| `_resume(...)` | affichage console : totaux, fonds propres, nombre de contrôles OK et en alerte, avertissements de calage remontés par `warnings.catch_warnings` |

Les avertissements de calage sont capturés pendant la génération non-vie puis affichés dans le
résumé, ce qui évite qu'ils se perdent dans le flux de sortie.

---

## 5. Package `scr` — moteur de calcul

### 5.1 `passif.py` (260 lignes) — best estimate déterministe

| Élément | Rôle |
|---|---|
| `Chocs` | dataclass gelée portant tous les chocs de passif : facteur de mortalité, choc absolu de mortalité (CAT), facteur de rachat, rachat massif, facteur de frais, inflation supplémentaire, facteur de valeur des UC, facteur de sinistres non-vie, facteur de récupération (santé), révision des rentes. Être *gelée* permet de comparer deux jeux de chocs par égalité, ce qui sert au raccourci de calcul dans le module marché |
| `_qx_projete(table, age, horizon, chocs)` | trajectoire de `q_x` avec multiplicateur et éventuel choc absolu la première année |
| `_frais_unitaires(base, hyp, chocs, horizon)` | frais unitaires inflatés, avec inflation supplémentaire en cas de choc |
| `be_epargne_euro(...)` | projection model point par model point sur 60 ans. Chaque année : taux brut `max(TMG + chargement, α × forward)`, PM revalorisée, sortie `q + rachat − q×rachat`, frais sur PM. Clôture du run-off en dernière année. `par_model_point=True` renvoie le vecteur détaillé, indispensable au rachat massif sélectif |
| `be_epargne_uc(...)` | même structure, mais la PM croît au forward diminué des chargements et rétrocessions ; le BE ressort donc naturellement inférieur à la PM |
| `be_temporaire_deces(...)` | valeur actuelle des capitaux décès et des frais, **nette des primes futures**, avec double décrément décès et chute |
| `be_rentes(...)` | somme des arrérages multipliés par le facteur de rente, réversion comprise, majorés des frais de gestion |
| `be_sante_slt(...)` | invalidité : maintien par mortalité aggravée jusqu'à la fin de garantie ; incapacité : maintien exponentiel en pas mensuel |
| `chain_ladder(triangle)` | facteurs de développement pondérés, complétion du triangle, ultimes et réserves |
| `cadence_reglement(facteurs)` | parts cumulées payées, déduites des facteurs |
| `be_non_vie(donnees, actu, hyp, chocs, modules)` | pour chaque ligne : Chain Ladder, cadence, **actualisation des réserves** au milieu de chaque année, majoration des frais de gestion de sinistres, puis provision pour primes `exposition × ratio combiné` actualisée |
| `Passifs.evaluer(actu, courbe, chocs)` | assemble tous les segments et le total. C'est l'unique point d'entrée utilisé par les modules de risque |

### 5.2 `marche.py` (250 lignes)

| Élément | Rôle |
|---|---|
| `_facteurs_officiels(chemin)` | lecture mise en cache du CSV des facteurs de choc extraits du fichier EIOPA |
| `facteurs_choc_taux(params, sens, maturites)` | facteurs relatifs par maturité : fichier officiel s'il est présent, sinon tables de repli du YAML |
| `courbe_choquee(courbe, params, sens)` | aiguille vers le régime 2027 si `params["taux"]["methode"] == "2027"`, sinon applique les facteurs relatifs, le plancher de un point à la hausse et la neutralisation des taux négatifs à la baisse |
| `revaloriser_obligations(inv, actu)` | revalorise chaque titre à revenu fixe sous une courbe donnée, en conservant coupon, maturité et spread |
| `facteur_stress_spread(cqs, duration, params, covered)` | table de l'article 176 : sélection de la bande de duration, plancher de duration à un an, traitement réduit des obligations sécurisées, repli sur la table des non notés |
| `matrice_marche(matrice, a, b)` | substitue les symboles `A` (corrélation du taux) et `B` (couple taux-spread du régime 2027) par leurs valeurs numériques |
| `ResultatSousModule` | petit conteneur nom / SCR / détail |
| `MoteurMarche.__init__` | mémorise l'inventaire, calcule le BE de base et les fonds propres de base de référence |
| `._actifs_hors_inventaire()` | provisions cédées, créances et trésorerie, constants sous choc |
| `._bof(valeurs, courbe, chocs)` | fonds propres de base sous un jeu de valeurs d'actif. **Raccourci de performance** : si la courbe est inchangée et que seul le facteur UC bouge, seul le BE des UC est reprojeté |
| `._perte(...)` | perte de fonds propres, plancher à zéro |
| `._facteur_uc(valeurs)` | variation relative de la valeur des supports UC, transmise au passif |
| `.taux()` | calcule hausse et baisse, mémorise les deux dans `pertes_taux` (utilisé par la V2), renvoie le pire et le sens retenu |
| `.action()` | chocs par type, agrégation type 1 et type 2 à 75 %, les participations stratégiques suivant le type 1 |
| `.immobilier()`, `.change()` | chocs de 25 % et de ±25 % par devise, la somme étant retenue entre devises |
| `.spread()` | stress ligne à ligne, exemption des souverains de l'EEE, réduction pour infrastructure qualifiante |
| `.concentration()` | excédent d'exposition par groupe d'émetteurs, hors UC et hors souverains, racine de la somme des carrés |
| `.calculer()` | agrège avec la matrice et le paramètre `A` correspondant au scénario de taux retenu |

### 5.3 `contrepartie.py` (68 lignes)

| Élément | Rôle |
|---|---|
| `_pd_exposition(cqs, params)` | probabilité de défaut par échelon, repli pour les non notés |
| `scr_type1(expositions, params)` | regroupe les LGD par probabilité de défaut, calcule la variance par la double somme de l'article 200, puis applique la règle à trois branches |
| `scr_type2(contreparties, params)` | 15 % des créances non échues, 90 % des créances échues depuis plus de trois mois |
| `calculer(...)` | construit les LGD (50 % pour la réassurance, 100 % pour la trésorerie) et agrège avec une corrélation de 0,75 |

### 5.4 `souscription.py` (190 lignes)

| Élément | Rôle |
|---|---|
| `agreger(scr, correlations)` | forme quadratique générique, utilisée par tous les modules de souscription |
| `MoteurVie._evaluer(chocs)` | BE des quatre segments vie sous un jeu de chocs |
| `MoteurVie._evaluer_mp(chocs)` | même chose, détaillée par model point |
| `MoteurVie._perte(chocs, segments)` | somme des hausses de BE, **retenues segment par segment** : un segment où le choc est favorable ne vient pas réduire le SCR |
| `.mortalite()`, `.longevite()` | multiplicateurs 1,15 et 0,80 |
| `.rachat()` | trois variantes, la pire étant retenue ; le détail conserve les trois valeurs |
| `._rachat_massif()` | rachat de 40 % appliqué **model point par model point**, en ne retenant que ceux dont le rachat accroît le BE |
| `.frais()`, `.revision()`, `.catastrophe()` | +10 % et +1 point d'inflation ; révision des rentes ; +0,15 point de mortalité la première année |
| `MoteurSante.slt()` | mortalité, longévité, invalidité (taux de sortie réduits), frais, révision |
| `MoteurSante.nslt()` | primes et réserves par la formule du sigma, rachat nul car le ratio combiné dépasse 1 |
| `MoteurSante.catastrophe()` | version paramétrique simplifiée : accident de masse, concentration, pandémie |

### 5.5 `non_vie.py` (101 lignes)

| Élément | Rôle |
|---|---|
| `.primes_reserves()` | volumes de primes et de réserves par ligne, sigma par ligne, agrégation par la matrice de corrélation, puis `3 σ V` |
| `.rachat()` | 40 % de cessations appliquées à la marge attendue sur primes non acquises : nul si le ratio combiné dépasse 1 |
| `.catastrophe()` | périls naturels agrégés quadratiquement, scénarios d'origine humaine (auto, incendie sur la plus grande concentration, responsabilité), puis traité XL |
| `._appliquer_xl(brut)` | `min(brut, priorité) + max(0, brut − priorité − portée)` |
| `.calculer()` | renvoie la table de synthèse, le détail par ligne et par péril, et le SCR agrégé |

### 5.6 `agregation.py` (104 lignes)

| Élément | Rôle |
|---|---|
| `bscr(scr_modules, params)` | forme quadratique sur les cinq modules, plus la table des parts |
| `operationnel(hyp, be, bscr, params, donnees)` | assiettes primes et provisions, maximum, plafond de 30 % du BSCR, puis 25 % des frais sur UC |
| `ajustements(bscr, scr_op, hyp, params)` | LAC TP (nulle en V1, l'absorption étant implicite) et LAC DT plafonnée par le montant d'impôts différés passifs augmenté d'une part de bénéfices futurs justifiables |
| `mcr(scr, be, donnees, hyp, params, detail_non_vie)` | MCR linéaire vie et non-vie, corridor 25 % – 45 %, minimum absolu |
| `tableau_s25(...)` | reconstitue le S.25.01, diversification comprise, calculée comme l'écart entre le BSCR et la somme des modules |

### 5.7 `esg.py` (161 lignes) — générateur de scénarios

| Élément | Rôle |
|---|---|
| `Scenarios` | dataclass des trajectoires : taux court, déflateurs, taux à 10 ans, rendements action et immobilier, plus la courbe initiale |
| `HullWhite.__init__` | extrait de la courbe le forward instantané par `np.gradient` sur `−ln P` |
| `.p0(t)`, `.f0(t)` | prix et forwards initiaux interpolés |
| `.alpha(t)` | terme d'ajustement `f(0,t) + σ²/(2a²)(1−e^{−at})²` assurant la reproduction exacte de la courbe |
| `.prix_zc(t, maturite, r)` | formule affine `A(t,T) e^{−B(t,T) r}` |
| `.simuler(...)` | schéma exact à pas annuel, sans biais de discrétisation |
| `_bruits_correles(...)` | tirages gaussiens corrélés par Cholesky, doublés en **variables antithétiques** |
| `generer_scenarios(courbe, hyp)` | assemble taux, déflateurs (intégrale par la règle du trapèze), taux à 10 ans conditionnels, rendements action et immobilier de dérive risque-neutre |
| `scenario_central(courbe, hyp)` | scénario unique déterministe fondé sur les forwards : référence pour isoler la valeur temps des options |
| `tests_martingale(sc)` | compare l'espérance des déflateurs aux prix de marché et vérifie que les actifs risqués déflatés valent 1 |

### 5.8 `alm.py` (230 lignes) — projection actif-passif

| Élément | Rôle |
|---|---|
| `construire_cohortes(mp, hyp, tables, horizon)` | regroupe les model points par TMG et quartile d'âge, et attache la matrice des `q_x` projetés dans `attrs["qx"]` |
| `construire_actif_euro(inventaire, hyp)` | agrège l'inventaire en quatre poches avec leurs valeurs comptables, le taux de rendement courant et la duration |
| `ResultatALM` | BE moyen, BE par scénario, flux moyens, taux servis, PPE finale, rachats moyens |
| `_taux_rachat_conjoncturel(ecart, lois)` | loi ONC en cinq segments, prise à mi-chemin des courbes plancher et plafond |
| `ModeleALM.projeter(sc, regime, …)` | boucle annuelle vectorisée sur les scénarios : revenus financiers, revalorisation des valeurs de marché, réalisation de plus-values latentes pour atteindre le taux cible, participation aux bénéfices, dotation puis reprise obligatoire de la PPE sur huit ans, rachats structurels et conjoncturels, décès, frais, adossement de l'actif, renouvellement du taux comptable obligataire |

Trois régimes d'actions du management : `dynamique` (net), `figee` (revalorisation maintenue au
taux cible, donc brut d'absorption) et `garantie` (TMG seul, ce qui isole les prestations
discrétionnaires futures). L'actif adossé au canton euro est proportionnel à ses engagements,
ce qui évite d'attribuer au fonds euro les revenus des actifs couvrant les autres passifs.

### 5.9 `reforme.py` (134 lignes) — régime 2027

| Élément | Rôle |
|---|---|
| `dernier_forward_liquide(courbe, fsp, poids)` | LLFR : moyenne pondérée des forwards liquides au-delà du premier point de lissage |
| `extrapoler_2027(...)` | au-delà du FSP, forward moyen convergeant vers l'UFR à la vitesse `a` ; reconstruit spots, facteurs et forwards |
| `_interpoler(table, maturites)` | interpolation linéaire des tables de chocs, avec prolongement aux bornes |
| `courbe_choquee_2027(...)` | choc multiplicatif plus décalage parallèle sur la partie liquide, planchers négatifs par terme à la baisse, puis ré-extrapolation avec UFR décalée de ±15 points de base |
| `params_2027(params, reforme, etapes)` | applique, sur une **copie profonde**, uniquement les étapes demandées : méthode de choc, corrélation taux-spread, corridor de l'ajustement symétrique, facteur de réassurance non proportionnelle |

L'argument `etapes` est ce qui rend le pont possible : on ajoute les modifications une à une.

### 5.10 `marge_risque.py` (38 lignes)

| Élément | Rôle |
|---|---|
| `profil_run_off(flux, actu)` | provisions restant à couvrir à chaque date, normalisées à 1 en t = 0 |
| `marge_risque(scr_initial, flux, actu, cout_capital, facteur_lambda, plancher)` | somme actualisée du coût du capital sur les SCR futurs, pondérés le cas échéant par `max(λ^t, plancher)` ; renvoie le total et le détail annuel |

Le SCR passé en entrée doit être celui des **risques non couvrables** : c'est le script V3 qui
l'assemble, en annulant le module de marché.

### 5.11 `modele_interne.py` (176 lignes)

| Élément | Rôle |
|---|---|
| `_increments(triangle)` | passage du cumulé à l'incrémental |
| `bootstrap_odp(triangle, nb_simulations, rng, ajustement_biais)` | triangle ajusté par rétro-projection des ultimes, résidus de Pearson corrigés du biais et recentrés, paramètre de dispersion `φ`, puis pour chaque simulation : rééchantillonnage, re-estimation des facteurs, projection, bruit de processus gamma de variance `φ × incrément` |
| `ajuster_sp(triangle, primes)` | moyenne et écart type logarithmiques des S/P ultimes estimés par Chain Ladder |
| `simuler_primes(...)` | charge de l'année à venir, log-normale, avec majoration pour incertitude de paramètre |
| `simuler_catastrophe(...)` | fréquence de Poisson, sévérités GPD au-delà d'un seuil par inversion, application du traité **événement par événement** |
| `copule(matrice, n, rng, famille, ddl)` | uniformes corrélés, gaussiens ou de Student |
| `coupler(marginales, uniformes)` | relie des marginales simulées indépendamment par les rangs, via les quantiles empiriques. Les marginales peuvent avoir des tailles différentes |
| `mesures_risque(pertes, quantile)` | moyenne, écart type, VaR, TVaR, SCR en écart à la moyenne, coefficient de variation |
| `usp_reserve(...)` | écart type spécifique et mélange de crédibilité avec le paramètre de marché |
| `ResultatModeleInterne` | conteneur prévu pour une future API ; non utilisé aujourd'hui |

### 5.12 `orsa.py` (232 lignes)

| Élément | Rôle |
|---|---|
| `EtatBilan` | dataclass du bilan projeté, avec les propriétés `actif_total`, `passif_total`, `fonds_propres` et les parts d'allocation |
| `etat_initial(be_segments, hyp, inventaire)` | bilan de départ, parts d'allocation calculées depuis l'inventaire réel |
| `ProjectionORSA.__init__` | mémorise les modules initiaux, le risque opérationnel, les deux ajustements et calcule le BSCR de départ |
| `.scr_initial()` | contrôle de cohérence : reconstitue le SCR de départ avec les mêmes briques que la projection |
| `._modules_projetes(...)` | inducteurs de volume : marché indexé sur les placements et, pour 30 %, sur la déformation de la part risquée ; vie sur le BE vie ; santé et non-vie sur les primes |
| `.projeter(scenario)` | boucle annuelle : environnement financier et chocs, déformation de l'allocation, activité vie, activité non-vie, compte de résultat, bouclage du bilan, SCR et ratio |
| `besoin_global_solvabilite(scr, hyp)` | ajoute les risques mal captés par la formule standard, avec abattement de diversification |

Le bouclage est le point clé : les fonds propres évoluent du résultat net diminué du dividende,
et l'actif se déduit du passif augmenté des fonds propres. L'égalité comptable est ainsi vraie
par construction, et un test la vérifie.

### 5.13 `rapport.py` (79 lignes)

`ecrire_rapport(chemin, hyp, resultats)` produit le classeur de restitution : onglet S.25.01
avec les ratios calculés, un onglet par module, BSCR, MCR et best estimate. `_ecrire_table`
applique les formats et met en évidence les lignes de total.

---

## 6. Scripts de calcul `calculer_*.py`

### 6.1 `calculer_scr.py` — V1

| Élément | Rôle |
|---|---|
| `TABLES` | liste des tables attendues dans `donnees/` |
| `charger_donnees(dossier)` | lit les CSV, avec un message explicite si la génération n'a pas été lancée |
| `charger_params(chemin)` | lit les paramètres réglementaires |
| `calculer(ecrire, hyp, params, courbe)` | enchaîne : passifs, marché, contrepartie, vie, santé, non-vie, BSCR, opérationnel, ajustements, SCR, MCR, S.25.01. Les trois derniers arguments permettent aux V2 à V5 de rejouer le calcul avec d'autres hypothèses ou une autre courbe |
| `_afficher(...)` | restitution console complète |

### 6.2 `calculer_v2.py` — ALM stochastique

| Élément | Rôle |
|---|---|
| `_stress_spread_moyen(inventaire, params)` | perte relative moyenne de la poche obligataire sous choc de spread, pour transmettre le choc au modèle ALM |
| `calculer(...)` | déroulé : V1 de référence, ESG, cohortes, BE central, stochastique, garanti et brut, puis chaque choc rejoué en net et en brut |
| `be_det(chocs, courbe_alt)` | BE déterministe du fonds euro sous le même choc, nécessaire à la substitution |
| `substituer(perte_v1, choc, regime)` | cœur de la V2 : `perte_V2 = perte_V1 − Δ déterministe + Δ stochastique`. Cette écriture isole proprement le canton euro sans réécrire les modules |
| `_rachat_massif_selectif(...)` | rachat massif cohorte par cohorte, avec PPE allouée au prorata, en ne retenant que les cohortes défavorables |
| `_sensibilite_tmg(...)` | valeur temps des options pour quatre niveaux de TMG |

La LAC TP est ensuite l'écart entre BSCR brut et BSCR net, plafonné par les prestations
discrétionnaires futures.

### 6.3 `calculer_v3.py` — régime 2027

| Élément | Rôle |
|---|---|
| `ETAPES` | liste ordonnée des étapes du pont, chacune portant l'ensemble cumulé des modifications à appliquer |
| `charger_reforme(hyp)` | lit `params_2027.yaml` et y injecte l'UFR de base |
| `scr_non_couvrable(resultat, params)` | SCR hors risque de marché, plus l'opérationnel : assiette correcte de la marge de risque |
| `_calculer_marge(...)` | marge de risque dans l'un ou l'autre régime |
| `calculer(...)` | rejoue la V2 à chaque étape, en remplaçant courbe et paramètres, et construit le pont |
| `_sensibilites(...)` | part du décalage parallèle des chocs de taux, et niveau de l'ajustement symétrique |

### 6.4 `calculer_v4.py` — modèle interne

| Élément | Rôle |
|---|---|
| `_triangles(donnees)` | dictionnaire des triangles matriciels |
| `calculer(...)` | marginales par ligne, agrégation entre lignes puis entre blocs, mesures de risque, USP, comparaison des quatre mesures, comparaison CIMA, graphique |
| `_recalculer_global(v2, scr_non_vie, hyp, params)` | remplace le module non-vie dans l'agrégation et recalcule LAC TP, LAC DT, SCR et ratio |
| `_marge_cima(...)` | marge de solvabilité du code CIMA, non-vie et vie |
| `_tracer(...)` | densités des deux copules, VaR en pointillés, repères de la formule standard |

### 6.5 `calculer_v5.py` — ORSA

| Élément | Rôle |
|---|---|
| `calculer(...)` | état initial, contrôle de reconstitution du SCR, trajectoire centrale, six scénarios, synthèse, stress inversés, besoin global, graphique |
| `_stress_inverses(projection, hyp)` | dichotomie sur 28 itérations pour trouver l'amplitude du choc amenant le ratio à la limite d'appétence |
| `_synthese(...)` | ratio minimal, année du minimum, franchissements, capital à injecter, feu de couleur |
| `_tracer(...)` | trajectoires et lignes d'appétence |

Les modules passés à la projection sont les modules **bruts** d'absorption, sans quoi la LAC TP
serait comptée deux fois.

---

## 7. Rapport

### `rapport/generer_rapport.py`

| Élément | Rôle |
|---|---|
| `lire(nom)` | lecture d'un CSV de résultats |
| `style(axes, …)` | mise en forme commune des graphiques |
| `figure_allocation`, `figure_modules`, `figure_pont`, `figure_tvog`, `figure_chocs_euro` | les cinq figures propres au rapport ; les deux autres sont copiées depuis `resultats/` |
| `macros()` | écrit une cinquantaine de `\newcommand` contenant les valeurs citées dans le texte |
| `nb`, `pourcent`, `texte` | formatage à la française et échappement des caractères réservés de LaTeX |
| `tableaux()` | écrit les corps de tableaux complets sous forme de macros |
| `principal()` | produit figures et `chiffres.tex` |

### `pedagogie/generer_chiffres.py`

Refait à la main chaque étape intermédiaire des modules (actualisation flux par flux, facteur de
stress, forme quadratique, formule de variance de la contrepartie, assiettes du risque
opérationnel) et écrit `chiffres_pedago.tex` : 112 macros et 9 corps de tableaux. Un bloc par
module — `bloc_obligation`, `bloc_action_immobilier_change`, `bloc_concentration`,
`bloc_agregation_marche`, `bloc_contrepartie`, `bloc_vie`, `bloc_non_vie`, `bloc_final` —
et deux utilitaires de formatage : `pc()` échappe le signe pourcent, `ecrire()` sérialise les
macros. Le cahier `cahier_calculs.tex` ne contient donc aucun chiffre en dur.

### `rapport/rapport_scr.tex` et `annexe_formulaire.tex`

Le document principal n'écrit aucun chiffre en dur : il appelle les macros. L'annexe A est un
formulaire autonome, l'annexe B décrit l'organisation du code. Le préambule bascule
automatiquement sur des noms français si `babel-french` est absent.

---

## 8. Tests

| Fichier | Ce qu'il protège |
|---|---|
| `test_generation.py` | repricing exact des swaps, convergence vers l'UFR, absence d'alerte de calage, total d'actif, reproductibilité du tirage, forme des triangles |
| `test_scr.py` | plancher du choc de taux à la hausse, sens du choc à la baisse, valeurs exactes de la table de spread, SCR de type 2, diversification du module marché, calage des BE, écart du Chain Ladder aux ultimes vrais, cohérence du CAT vie, non-prise en compte des chocs favorables, corridor du MCR, plausibilité du ratio |
| `test_v2.py` | reproduction de la courbe par Hull-White, martingale des déflateurs, déterminisme du scénario central, positivité des prestations discrétionnaires et plafonnement de la LAC TP, BSCR brut supérieur au net, cohérence du SCR, absorption positive sur les chocs d'actif, forme en cloche de la TVOG |
| `test_v3.py` | extrapolation inchangée avant le point de lissage et convergente au-delà, sens des chocs 2027, respect des planchers négatifs, effet de `params_2027` sans mutation des paramètres d'origine, décroissance du profil de run-off, baisse de la marge de risque |
| `test_v4.py` | bootstrap centré sur le Chain Ladder, écrêtement par le traité XL, corrélation restituée par la copule, dépendance de queue plus forte pour Student, préservation des marginales par le couplage, TVaR supérieure à la VaR, encadrement de l'USP |
| `test_v5.py` | égalité comptable à chaque pas, stabilité de la trajectoire centrale, dégradation par les scénarios adverses, crise combinée comme pire cas, sens des stress inversés, besoin global supérieur au SCR |

58 tests au total, exécutés en une quinzaine de secondes.

---

## 9. Pièges rencontrés et corrigés

Ces points sont les plus instructifs : ce sont des erreurs réelles, détectées et corrigées
pendant le développement.

**Allocation d'actif du canton euro.** La première version du modèle ALM attribuait au fonds
euro les revenus de la totalité du portefeuille, alors que celui-ci couvre aussi les autres
passifs et les fonds propres. La PPE explosait à 2,7 milliards. L'actif est désormais alloué au
prorata des engagements du canton.

**Référence du calcul brut.** Les variations de BE en régime figé étaient mesurées par rapport
au BE dynamique, ce qui mélangeait deux régimes et produisait une LAC TP absurde. Le calcul brut
se mesure maintenant par rapport à la base brute.

**Double comptage de l'absorption.** Dans l'ORSA, partir des modules nets tout en ajoutant la
LAC TP revenait à compter l'absorption deux fois. Le contrôle `scr_initial()` reconstitue le SCR
de départ et lève une exception si l'écart dépasse 1 %.

**Assiette de la marge de risque.** Calculée sur le SCR total, elle atteignait 597 M€. Elle ne
porte que sur les risques non couvrables, donc hors marché : 253 M€.

**Échappement LaTeX.** Le formatage des milliers transformait les séparateurs déjà échappés en
doubles antislashs, ce qui coupait les lignes des tableaux. Le formatage se fait désormais
nombre par nombre, avec une fonction d'échappement dédiée pour les libellés.

**Décalage de `ws.append` dans openpyxl.** Une ligne vide n'incrémente pas `max_row`, ce qui
désynchronisait le compteur de lignes. L'écriture du bilan se fait cellule par cellule.

**Chain Ladder sans facteur de queue.** L'écart aux ultimes vrais (−30 % sur la RC auto) n'est
pas un bug mais une limite de méthode, documentée et mesurée grâce aux données simulées.

**Copule et tailles de marginales.** Le couplage passe par les quantiles empiriques, ce qui
permet d'associer des marginales de tailles différentes — 5 000 tirages bootstrap et 50 000
tirages log-normaux.

---

## 10. Recettes d'extension

**Changer une hypothèse.** Modifier `config/hypotheses.yaml`, relancer `generer_donnees.py`
puis les scripts de calcul. Les contrôles signalent immédiatement une incohérence de calage.

**Brancher les données EIOPA officielles.** Déposer le fichier des courbes dans `sources/`,
passer `courbe.source` à `eiopa`. Pour les chocs, l'idéal est d'ajouter deux onglets choqués et
de court-circuiter `courbe_choquee`.

**Ajouter une ligne d'activité non-vie.** Une entrée dans `non_vie.lignes` (primes, S/P,
cadence, provision cible), une entrée dans `params_reglementaires.non_vie.sigma`, et une ligne
et une colonne dans les deux matrices de corrélation. Tout le reste suit.

**Ajouter un sous-module de marché.** Une méthode dans `MoteurMarche` renvoyant un
`ResultatSousModule`, son nom dans `correlations_marche.ordre`, et une ligne et une colonne dans
la matrice.

**Calibrer l'ESG sur des prix d'options.** Remplacer `esg.hull_white.sigma` et
`esg.action.volatilite` par les valeurs issues d'un calibrage sur swaptions et options sur
indice. Les tests de martingale restent valides, mais la TVOG et la LAC TP bougeront.

**Passer au modèle interne complet.** Le module `modele_interne.py` est indépendant du reste :
ajouter des marginales vie et marché, puis les agréger avec les mêmes fonctions de copule.
