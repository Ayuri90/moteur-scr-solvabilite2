"""V3 — étude d'impact du régime 2027 (Directive (UE) 2025/2, Règlement délégué (UE) 2026/269).

Le calcul V2 est rejoué en ajoutant une à une les modifications du nouveau régime, ce qui
produit un pont (waterfall) du ratio de solvabilité entre le régime actuel et le régime 2027 :

  1. extrapolation alternative de la courbe (FSP, LLFR, convergence vers l'UFR) ;
  2. chocs de taux révisés (multiplicatif + décalage, planchers négatifs, UFR stressé) ;
  3. corrélation taux / spread ramenée à 0,25 en scénario baissier ;
  4. corridor de l'ajustement symétrique élargi à +/- 13 % ;
  5. facteur de réassurance non proportionnelle en primes non-vie ;
  6. marge de risque : coût du capital 4,75 % et dégressivité temporelle.

Usage :
    python calculer_v3.py
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import yaml

import calculer_scr
import calculer_v2
from scr import agregation
from scr.marge_risque import marge_risque
from scr.reforme import extrapoler_2027, params_2027
from scr_data.config import RACINE, charger_hypotheses
from scr_data.courbe import Actualisation, construire_courbe

ETAPES = [
    ("reference", "Régime actuel (V2)", set()),
    ("marge_risque_recalculee", "Marge de risque recalculée (CoC 6 %)", set()),
    ("extrapolation", "Extrapolation alternative (FSP 20 ans, LLFR)", {"extrapolation"}),
    ("chocs_taux", "Chocs de taux révisés", {"extrapolation", "chocs_taux"}),
    ("correlation_taux_spread", "Corrélation taux / spread à 0,25",
     {"extrapolation", "chocs_taux", "correlation_taux_spread"}),
    ("ajustement_symetrique", "Corridor de l'ajustement symétrique +/- 13 %",
     {"extrapolation", "chocs_taux", "correlation_taux_spread", "ajustement_symetrique"}),
    ("reassurance_np", "Réassurance non proportionnelle (primes non-vie)",
     {"extrapolation", "chocs_taux", "correlation_taux_spread", "ajustement_symetrique",
      "reassurance_non_proportionnelle"}),
    ("marge_risque_2027", "Marge de risque : CoC 4,75 % et dégressivité",
     {"extrapolation", "chocs_taux", "correlation_taux_spread", "ajustement_symetrique",
      "reassurance_non_proportionnelle"}),
]


def charger_reforme(hyp) -> dict:
    with open(RACINE / "config" / "params_2027.yaml", encoding="utf-8") as f:
        reforme = yaml.safe_load(f)
    reforme["_ufr_base"] = hyp["courbe"]["ufr"]
    return reforme


def scr_non_couvrable(resultat, params) -> float:
    """SCR des risques non couvrables : le risque de marché est exclu de la marge de risque."""
    modules = resultat["bscr_net"].set_index("module")["scr"].to_dict()
    modules.pop("BSCR", None)
    modules["marche"] = 0.0
    total, _ = agregation.bscr(modules, params)
    operationnel = float(resultat["s25_v2"].set_index("poste")
                         .loc["Risque opérationnel", "montant"])
    return total + operationnel


def _calculer_marge(resultat, courbe, reforme, params, regime: str) -> float:
    """Marge de risque simplifiée : SCR non couvrable projeté au prorata du run-off."""
    p = reforme["marge_risque"]
    flux = resultat["flux"]["flux_moyen"].to_numpy()
    actu = Actualisation(courbe)
    scr_nh = scr_non_couvrable(resultat, params)
    if regime == "actuel":
        valeur, _ = marge_risque(scr_nh, flux, actu, p["cout_du_capital_actuel"])
    else:
        valeur, _ = marge_risque(scr_nh, flux, actu, p["cout_du_capital_2027"],
                                 facteur_lambda=p["lambda"], plancher=p["plancher_lambda"])
    return valeur


def calculer(ecrire: bool = True) -> dict:
    debut = time.time()
    hyp_base = charger_hypotheses()
    params_base = calculer_scr.charger_params()
    reforme = charger_reforme(hyp_base)
    courbe_base, _ = construire_courbe(hyp_base)
    courbe_2027 = extrapoler_2027(courbe_base, reforme, hyp_base["courbe"]["ufr"])

    lignes, precedent = [], None
    marge_courante = hyp_base["bilan_cible"]["passif"]["marge_risque"]
    for cle, libelle, etapes in ETAPES:
        hyp = copy.deepcopy(hyp_base)
        hyp["bilan_cible"]["passif"]["marge_risque"] = marge_courante
        params = params_2027(params_base, reforme, etapes) if etapes else copy.deepcopy(params_base)
        courbe = courbe_2027 if "extrapolation" in etapes else courbe_base
        resultat = calculer_v2.calculer(ecrire=False, hyp=hyp, params=params, courbe=courbe)

        if cle == "marge_risque_recalculee":
            marge_courante = _calculer_marge(resultat, courbe, reforme, params, "actuel")
            hyp["bilan_cible"]["passif"]["marge_risque"] = marge_courante
            resultat = calculer_v2.calculer(ecrire=False, hyp=hyp, params=params, courbe=courbe)
        elif cle == "marge_risque_2027":
            marge_courante = _calculer_marge(resultat, courbe, reforme, params, "2027")
            hyp["bilan_cible"]["passif"]["marge_risque"] = marge_courante
            resultat = calculer_v2.calculer(ecrire=False, hyp=hyp, params=params, courbe=courbe)

        bof = resultat["scr"] * resultat["ratio"]
        ligne = {"etape": libelle, "be_epargne_euro": resultat["be"]["stochastique"],
                 "marge_risque": marge_courante, "fonds_propres": bof, "scr": resultat["scr"],
                 "ratio": resultat["ratio"]}
        if precedent is not None:
            ligne["delta_scr"] = ligne["scr"] - precedent["scr"]
            ligne["delta_ratio"] = ligne["ratio"] - precedent["ratio"]
        lignes.append(ligne)
        precedent = ligne
        print(f"  {libelle:52s} SCR {ligne['scr']:9,.1f}   ratio {ligne['ratio']:6.0%}")

    pont = pd.DataFrame(lignes)
    sensibilites = _sensibilites(hyp_base, params_base, reforme, courbe_2027)

    if ecrire:
        dossier = RACINE / "resultats"
        dossier.mkdir(exist_ok=True)
        pont.to_csv(dossier / "v3_pont_2027.csv", index=False, sep=";", decimal=",",
                    encoding="utf-8-sig")
        sensibilites.to_csv(dossier / "v3_sensibilites.csv", index=False, sep=";", decimal=",",
                            encoding="utf-8-sig")
        _afficher(pont, sensibilites, time.time() - debut)
    return {"pont": pont, "sensibilites": sensibilites, "courbe_2027": courbe_2027}


def _sensibilites(hyp_base, params_base, reforme, courbe_2027) -> pd.DataFrame:
    """Le décalage parallèle des chocs de taux et l'ajustement symétrique sont les deux
    paramètres les plus incertains : on mesure leur poids sur le résultat."""
    lignes = []
    etapes = {"extrapolation", "chocs_taux", "correlation_taux_spread", "ajustement_symetrique",
              "reassurance_non_proportionnelle"}
    for part in (0.0, 0.5, 1.0):
        ref = copy.deepcopy(reforme)
        for cle in ("a_hausse", "a_baisse"):
            ref["taux"][cle] = {m: v * part for m, v in ref["taux"][cle].items()}
        params = params_2027(params_base, ref, etapes)
        r = calculer_v2.calculer(ecrire=False, hyp=hyp_base, params=params, courbe=courbe_2027)
        lignes.append({"parametre": "part du décalage parallèle des chocs de taux",
                       "valeur": part, "scr": r["scr"], "ratio": r["ratio"]})
    for sa in (-0.10, -0.13):
        ref = copy.deepcopy(reforme)
        params = copy.deepcopy(params_base)
        params["action"]["ajustement_symetrique"] = sa
        params = params_2027(params, ref, etapes)
        r = calculer_v2.calculer(ecrire=False, hyp=hyp_base, params=params, courbe=courbe_2027)
        lignes.append({"parametre": "ajustement symétrique en bas de corridor",
                       "valeur": sa, "scr": r["scr"], "ratio": r["ratio"]})
    return pd.DataFrame(lignes)


def _afficher(pont: pd.DataFrame, sensibilites: pd.DataFrame, duree: float):
    print("\nPont du ratio de solvabilité — régime actuel vers régime 2027")
    print(f"  {'étape':52s} {'SCR':>9s} {'Δ SCR':>9s} {'ratio':>7s} {'Δ ratio':>9s}")
    for r in pont.itertuples():
        delta_scr = "" if pd.isna(getattr(r, "delta_scr", np.nan)) else f"{r.delta_scr:+9,.1f}"
        delta_ratio = "" if pd.isna(getattr(r, "delta_ratio", np.nan)) else f"{r.delta_ratio:+9.1%}"
        print(f"  {r.etape:52s} {r.scr:9,.1f} {delta_scr:>9s} {r.ratio:7.0%} {delta_ratio:>9s}")
    depart, arrivee = pont.iloc[0], pont.iloc[-1]
    print(f"\n  Total : SCR {depart.scr:,.1f} → {arrivee.scr:,.1f} M€ "
          f"({arrivee.scr / depart.scr - 1:+.1%}), ratio {depart.ratio:.0%} → {arrivee.ratio:.0%}")
    print(f"  Marge de risque : {depart.marge_risque:,.1f} → {arrivee.marge_risque:,.1f} M€")
    print("\nSensibilités aux paramètres incertains")
    for r in sensibilites.itertuples():
        print(f"  {r.parametre:46s} {r.valeur:+7.2f}  SCR {r.scr:9,.1f}  ratio {r.ratio:6.0%}")
    print(f"\nCalcul effectué en {duree:.0f} s\n")


if __name__ == "__main__":
    calculer()
