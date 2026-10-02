"""Régime 2027 — Directive (UE) 2025/2 et Règlement délégué (UE) 2026/269.

Ce module construit les objets du nouveau régime :
- extrapolation alternative de la courbe (premier point de lissage, dernier taux forward
  liquide, convergence vers l'UFR) ;
- chocs de taux combinant un choc multiplicatif et un décalage parallèle, avec planchers
  négatifs dépendant du terme et UFR stressé de ±15 points de base ;
- ajustements des paramètres de la formule standard (corrélation taux / spread en scénario
  baissier, corridor de l'ajustement symétrique, réassurance non proportionnelle en non-vie,
  coût du capital de la marge de risque).

ATTENTION : les tables de chocs de taux sont saisies de mémoire et marquées [AVERIFIER].
La route fiable consiste à utiliser les courbes choquées publiées par l'EIOPA.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Extrapolation alternative
# ---------------------------------------------------------------------------
def dernier_forward_liquide(courbe: pd.DataFrame, fsp: int, poids: dict[int, float]) -> float:
    """LLFR : moyenne pondérée des taux forward liquides observés au-delà du FSP."""
    spot = courbe.set_index("maturite")["taux_spot"]
    log_df = -np.log1p(spot) * spot.index
    total_poids = sum(poids.values())
    llfr = 0.0
    for maturite, poids_m in poids.items():
        if maturite == fsp:
            llfr += poids_m * (-log_df.loc[fsp] / fsp)
        else:
            forward = (log_df.loc[fsp] - log_df.loc[maturite]) / (maturite - fsp)
            llfr += poids_m * forward
    return float(llfr / total_poids)


def extrapoler_2027(courbe: pd.DataFrame, params_reforme: dict[str, Any],
                    ufr: float, decalage_ufr: float = 0.0) -> pd.DataFrame:
    """Prolonge la partie liquide de la courbe par la méthode LLFR (convergence en a)."""
    p = params_reforme["extrapolation"]
    fsp, a = p["premier_point_lissage"], p["vitesse_convergence"]
    poids = {int(k): v for k, v in p["poids_llfr"].items()}
    llfr = dernier_forward_liquide(courbe, fsp, poids)
    ufr_continu = np.log1p(ufr + decalage_ufr)

    spot = courbe.set_index("maturite")["taux_spot"]
    log_df_fsp = -np.log1p(spot.loc[fsp]) * fsp
    maturites = courbe["maturite"].to_numpy(int)
    nouveau_spot = spot.to_numpy(float).copy()
    for i, m in enumerate(maturites):
        if m <= fsp:
            continue
        h = m - fsp
        # forward moyen extrapolé entre le FSP et la maturité m
        forward = llfr + (ufr_continu - llfr) * (1 - np.exp(-a * h)) / (a * h)
        log_df = log_df_fsp - forward * h
        nouveau_spot[i] = np.exp(-log_df / m) - 1.0
    df = (1 + nouveau_spot) ** (-maturites)
    forward_1an = np.concatenate([[1.0], df[:-1]]) / df - 1.0
    return pd.DataFrame({"maturite": maturites, "taux_spot": nouveau_spot,
                         "facteur_actualisation": df, "forward_1an": forward_1an})


# ---------------------------------------------------------------------------
# Chocs de taux 2027
# ---------------------------------------------------------------------------
def _interpoler(table: dict[int, float], maturites: np.ndarray) -> np.ndarray:
    points = np.array(sorted(table), dtype=float)
    valeurs = np.array([table[int(m)] for m in points])
    return np.interp(maturites, points, valeurs, left=valeurs[0], right=valeurs[-1])


def courbe_choquee_2027(courbe: pd.DataFrame, params_reforme: dict[str, Any], sens: str,
                        ufr: float) -> pd.DataFrame:
    """Choc multiplicatif + décalage parallèle sur la partie liquide, puis ré-extrapolation."""
    p = params_reforme["taux"]
    fsp = params_reforme["extrapolation"]["premier_point_lissage"]
    maturites = courbe["maturite"].to_numpy(float)
    spot = courbe["taux_spot"].to_numpy(float)
    if sens == "hausse":
        b = _interpoler({int(k): v for k, v in p["b_hausse"].items()}, maturites)
        a = _interpoler({int(k): v for k, v in p["a_hausse"].items()}, maturites)
        choque = spot * (1 + b) + a
        decalage_ufr = p["decalage_ufr_hausse"]
    else:
        b = _interpoler({int(k): v for k, v in p["b_baisse"].items()}, maturites)
        a = _interpoler({int(k): v for k, v in p["a_baisse"].items()}, maturites)
        planchers = _interpoler({int(k): v for k, v in p["planchers_baisse"].items()}, maturites)
        choque = np.maximum(spot * (1 - b) - a, planchers)
        decalage_ufr = p["decalage_ufr_baisse"]
    # les données de marché restent choquées jusqu'à la dernière maturité liquide
    # utilisée pour le LLFR ; au-delà du FSP, la courbe est ré-extrapolée
    derniere_liquide = max(int(m) for m in params_reforme["extrapolation"]["poids_llfr"])
    liquide = maturites <= derniere_liquide
    intermediaire = courbe.copy()
    intermediaire["taux_spot"] = np.where(liquide, choque, spot)
    intermediaire["facteur_actualisation"] = (1 + intermediaire["taux_spot"]) ** (-maturites)
    return extrapoler_2027(intermediaire, params_reforme, ufr, decalage_ufr)


# ---------------------------------------------------------------------------
# Paramètres de la formule standard révisés
# ---------------------------------------------------------------------------
def params_2027(params: dict[str, Any], reforme: dict[str, Any], etapes: set[str]) -> dict[str, Any]:
    """Applique les modifications retenues du Règlement 2026/269, étape par étape."""
    p = copy.deepcopy(params)
    if "chocs_taux" in etapes:
        p["taux"]["methode"] = "2027"
        p["taux"]["reforme"] = reforme
        p["taux"]["ufr_base"] = reforme["_ufr_base"]
    if "correlation_taux_spread" in etapes:
        # Art. 164(3) amendé : corrélation taux / spread ramenée de 0,50 à 0,25 en scénario baissier
        matrice = p["correlations_marche"]["matrice"]
        ordre = p["correlations_marche"]["ordre"]
        i, j = ordre.index("taux"), ordre.index("spread")
        matrice[i][j] = matrice[j][i] = "B"
        p["correlations_marche"]["b_baisse"] = reforme["correlations"]["taux_spread_baisse"]
        p["correlations_marche"]["b_hausse"] = reforme["correlations"]["b_hausse"]
        p["correlations_marche"]["a_baisse"] = reforme["correlations"]["a_baisse"]
    if "ajustement_symetrique" in etapes:
        borne = reforme["action"]["corridor_ajustement_symetrique"]
        p["action"]["ajustement_symetrique"] = float(
            np.clip(params["action"]["ajustement_symetrique"], -borne, borne))
    if "reassurance_non_proportionnelle" in etapes:
        facteur = reforme["non_vie"]["facteur_np"]
        for lob, actif in reforme["non_vie"]["couverture_np"].items():
            if actif and lob in p["non_vie"]["sigma"]:
                p["non_vie"]["sigma"][lob]["primes"] *= facteur
    return p
