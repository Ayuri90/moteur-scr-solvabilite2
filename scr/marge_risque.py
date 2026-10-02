"""Marge de risque (Art. 37 à 39) et sa version révisée par la Directive (UE) 2025/2.

Simplification retenue (hiérarchie des simplifications de l'EIOPA, niveau 2) : les SCR futurs
sont supposés proportionnels au run-off des provisions techniques.
Le régime 2027 remplace le coût du capital de 6 % par 4,75 % et introduit une dégressivité
temporelle : le SCR de l'année t est pondéré par max(lambda^t, plancher).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scr_data.courbe import Actualisation


def profil_run_off(flux: np.ndarray, actu: Actualisation) -> np.ndarray:
    """Provisions restant à couvrir à la fin de chaque année, en proportion du BE initial."""
    horizon = len(flux)
    t = np.arange(1, horizon + 1)
    df = actu.df(t)
    valeurs = np.array([np.sum(flux[k:] * df[k:]) / df[k - 1] if k > 0 else np.sum(flux * df)
                        for k in range(horizon)])
    return valeurs / valeurs[0] if valeurs[0] else valeurs


def marge_risque(scr_initial: float, flux: np.ndarray, actu: Actualisation, cout_capital: float,
                 facteur_lambda: float | None = None, plancher: float = 0.5) -> tuple[float, pd.DataFrame]:
    profil = profil_run_off(np.asarray(flux, dtype=float), actu)
    horizon = len(profil)
    t = np.arange(horizon)
    ponderation = np.ones(horizon) if facteur_lambda is None else np.maximum(
        facteur_lambda ** t, plancher)
    scr_futurs = scr_initial * profil * ponderation
    df = actu.df(t + 1)
    contributions = cout_capital * scr_futurs * df
    detail = pd.DataFrame({"annee": t + 1, "part_run_off": profil, "ponderation": ponderation,
                           "scr_projete": scr_futurs, "contribution": contributions})
    return float(contributions.sum()), detail
