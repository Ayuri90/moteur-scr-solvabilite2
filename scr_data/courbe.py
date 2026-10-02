"""Courbe des taux sans risque.

Deux sources possibles :
- le fichier officiel EIOPA (Term Structures, onglet RFR_spot_no_VA) — recommandé ;
- une reconstruction Smith-Wilson à partir de taux swap, selon la méthodologie
  EIOPA (calibrage sur swaps au pair, CRA, UFR, choix d'alpha au point de convergence).

La courbe produite contient, pour chaque maturité entière, le taux spot annuel
(composition annuelle), le facteur d'actualisation et le taux forward à 1 an.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .config import RACINE


# ---------------------------------------------------------------------------
# Smith-Wilson
# ---------------------------------------------------------------------------
def _wilson(t: np.ndarray, u: np.ndarray, alpha: float, omega: float) -> np.ndarray:
    """Matrice des fonctions de Wilson W(t_i, u_j)."""
    t = np.asarray(t, dtype=float)[:, None]
    u = np.asarray(u, dtype=float)[None, :]
    t_min, t_max = np.minimum(t, u), np.maximum(t, u)
    return np.exp(-omega * (t + u)) * (
        alpha * t_min - 0.5 * np.exp(-alpha * t_max) * (np.exp(alpha * t_min) - np.exp(-alpha * t_min))
    )


class SmithWilson:
    """Interpolation / extrapolation Smith-Wilson calibrée sur des swaps au pair."""

    def __init__(self, maturites: list[int], taux_pair: list[float], ufr: float, alpha: float):
        self.omega = np.log1p(ufr)
        self.alpha = alpha
        maturites = np.asarray(maturites, dtype=int)
        taux_pair = np.asarray(taux_pair, dtype=float)
        self.u = np.arange(1, maturites.max() + 1, dtype=float)       # dates de flux
        # Matrice des flux : un swap au pair de maturité m paie r chaque année, 1 + r à m
        C = np.zeros((len(maturites), len(self.u)))
        for i, (m, r) in enumerate(zip(maturites, taux_pair)):
            C[i, :m] = r
            C[i, m - 1] += 1.0
        self.C = C
        mu = np.exp(-self.omega * self.u)
        W = _wilson(self.u, self.u, alpha, self.omega)
        prix = np.ones(len(maturites))
        self.zeta = np.linalg.solve(C @ W @ C.T, prix - C @ mu)

    def facteur_actualisation(self, t: np.ndarray | float) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        W = _wilson(t, self.u, self.alpha, self.omega)
        return np.exp(-self.omega * t) + W @ (self.C.T @ self.zeta)

    def forward_instantane(self, t: float, h: float = 1e-4) -> float:
        p = self.facteur_actualisation(np.array([t - h, t + h]))
        return float(-(np.log(p[1]) - np.log(p[0])) / (2 * h))


def calibrer_alpha(maturites, taux_pair, ufr, point_convergence, tolerance_bp, alpha_min) -> float:
    """Plus petit alpha >= alpha_min tel que |f(CP) - omega| <= tolérance (méthode EIOPA)."""
    omega = np.log1p(ufr)
    tol = tolerance_bp / 1e4

    def ecart(alpha: float) -> float:
        sw = SmithWilson(maturites, taux_pair, ufr, alpha)
        return abs(sw.forward_instantane(point_convergence) - omega) - tol

    if ecart(alpha_min) <= 0:
        return alpha_min
    alpha_max = 1.0
    if ecart(alpha_max) > 0:
        raise RuntimeError("Impossible de respecter la tolérance au point de convergence.")
    return float(brentq(ecart, alpha_min, alpha_max, xtol=1e-6))


def courbe_smith_wilson(hyp_courbe: dict[str, Any]) -> tuple[pd.DataFrame, float]:
    swaps = {int(k): float(v) for k, v in hyp_courbe["taux_swap"].items()}
    maturites = sorted(swaps)
    cra = hyp_courbe["cra_bp"] / 1e4
    taux = [swaps[m] - cra for m in maturites]
    alpha = calibrer_alpha(
        maturites, taux, hyp_courbe["ufr"], hyp_courbe["point_convergence"],
        hyp_courbe["tolerance_convergence_bp"], hyp_courbe["alpha_min"],
    )
    sw = SmithWilson(maturites, taux, hyp_courbe["ufr"], alpha)
    t = np.arange(1, hyp_courbe["maturite_max"] + 1)
    return _mettre_en_forme(t, sw.facteur_actualisation(t)), alpha


# ---------------------------------------------------------------------------
# Fichier EIOPA
# ---------------------------------------------------------------------------
def courbe_eiopa(hyp_courbe: dict[str, Any]) -> pd.DataFrame:
    """Lit une colonne de l'onglet RFR_spot_no_VA (ou with_VA) du fichier EIOPA.

    La mise en page EIOPA place le nom de la devise dans une ligne d'en-tête,
    puis les taux spot des maturités 1 à 150 plus bas dans la même colonne.
    """
    chemin = RACINE / hyp_courbe["fichier_eiopa"]
    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier EIOPA introuvable : {chemin}. Téléchargez les 'Risk-free interest rate "
            "term structures' du 31/12/2025 sur le site de l'EIOPA, ou passez courbe.source "
            "à 'smith_wilson'."
        )
    brut = pd.read_excel(chemin, sheet_name=hyp_courbe["onglet_eiopa"], header=None)
    positions = np.argwhere(brut.astype(str).apply(lambda s: s.str.strip()).values
                            == hyp_courbe["colonne_eiopa"])
    if len(positions) == 0:
        raise ValueError(f"Colonne '{hyp_courbe['colonne_eiopa']}' absente de l'onglet EIOPA.")
    ligne, col = positions[0]
    valeurs = pd.to_numeric(brut.iloc[ligne + 1:, col], errors="coerce")
    # première série de 150 valeurs numériques consécutives
    serie = valeurs.dropna()
    idx = serie.index.to_numpy()
    debut = next(i for i in range(len(idx)) if np.all(np.diff(idx[i:i + 150]) == 1))
    spots = serie.iloc[debut:debut + 150].to_numpy(dtype=float)
    t = np.arange(1, len(spots) + 1)
    return _mettre_en_forme(t, (1 + spots) ** (-t))


# ---------------------------------------------------------------------------
def _mettre_en_forme(t: np.ndarray, df: np.ndarray) -> pd.DataFrame:
    spot = df ** (-1.0 / t) - 1.0
    df_prec = np.concatenate([[1.0], df[:-1]])
    forward = df_prec / df - 1.0
    return pd.DataFrame({"maturite": t, "taux_spot": spot, "facteur_actualisation": df,
                         "forward_1an": forward})


def construire_courbe(hyp: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    hc = hyp["courbe"]
    if hc["source"] == "eiopa":
        return courbe_eiopa(hc), {"source": "EIOPA", "alpha": None}
    courbe, alpha = courbe_smith_wilson(hc)
    return courbe, {"source": "Smith-Wilson (taux swap indicatifs)", "alpha": alpha}


class Actualisation:
    """Accès pratique à la courbe : facteurs d'actualisation à maturités non entières."""

    def __init__(self, courbe: pd.DataFrame):
        self.t = np.concatenate([[0.0], courbe["maturite"].to_numpy(float)])
        self.log_df = np.concatenate([[0.0], np.log(courbe["facteur_actualisation"].to_numpy())])

    def df(self, t: np.ndarray | float, spread: float = 0.0) -> np.ndarray:
        """Facteur d'actualisation, avec spread additif sur le taux annuel équivalent."""
        t = np.asarray(t, dtype=float)
        log_df = np.interp(t, self.t, self.log_df)       # interpolation log-linéaire
        if spread == 0.0:
            return np.exp(log_df)
        spot = np.exp(-log_df / np.maximum(t, 1e-12)) - 1.0
        return np.where(t > 0, (1.0 + spot + spread) ** (-t), 1.0)
