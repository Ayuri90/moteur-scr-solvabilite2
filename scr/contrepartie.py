"""Risque de défaut de la contrepartie (Art. 189 à 202) : expositions de type 1 et de type 2."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _pd_exposition(cqs, params: dict[str, Any]) -> float:
    p = params["contrepartie"]
    if cqs is None or (isinstance(cqs, float) and np.isnan(cqs)):
        return p["pd_non_note"]
    return p["pd_par_cqs"][int(cqs)]


def scr_type1(expositions: pd.DataFrame, params: dict[str, Any]) -> tuple[float, pd.DataFrame]:
    """expositions : colonnes contrepartie, cqs, lgd."""
    p = params["contrepartie"]
    exp = expositions.copy()
    exp["pd"] = exp["cqs"].apply(lambda c: _pd_exposition(c, params))
    groupes = exp.groupby("pd")["lgd"].agg(["sum", lambda s: float((s**2).sum())])
    groupes.columns = ["tlgd", "somme_lgd2"]

    V = 0.0
    for pd_j, rj in groupes.iterrows():
        for pd_k, rk in groupes.iterrows():
            denom = 1.25 * (pd_j + pd_k) - pd_j * pd_k
            V += (pd_j * pd_k) / denom * rj["tlgd"] * rk["tlgd"]
    for pd_j, rj in groupes.iterrows():
        V += (1.5 * pd_j * (1 - pd_j)) / (2.5 - pd_j) * rj["somme_lgd2"]

    total_lgd = float(exp["lgd"].sum())
    racine = float(np.sqrt(max(V, 0.0)))
    s1, s2 = p["seuils_racine_variance"]
    m1, m2 = p["multiplicateurs"]
    if racine <= s1 * total_lgd:
        scr = m1 * racine
    elif racine <= s2 * total_lgd:
        scr = m2 * racine
    else:
        scr = total_lgd
    return float(scr), exp


def scr_type2(contreparties: pd.DataFrame, params: dict[str, Any]) -> float:
    p = params["contrepartie"]
    echues = contreparties.loc[contreparties["type_exposition"] == "type2_creances_echues_3m", "montant"].sum()
    autres = contreparties.loc[contreparties["type_exposition"] == "type2_creances", "montant"].sum()
    return float(p["type2_facteur_standard"] * autres + p["type2_facteur_echues_3m"] * echues)


def calculer(contreparties: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, float]:
    p = params["contrepartie"]
    type1 = contreparties[contreparties["type_exposition"].str.startswith("type1")].copy()
    type1["lgd"] = np.where(type1["type_exposition"] == "type1_reassurance",
                            p["lgd_reassurance"] * type1["montant"],
                            p["lgd_tresorerie"] * type1["montant"])
    s1, detail = scr_type1(type1[["contrepartie", "cqs", "lgd"]], params)
    s2 = scr_type2(contreparties, params)
    scr = float(np.sqrt(s1**2 + 1.5 * s1 * s2 + s2**2))
    table = pd.DataFrame([
        {"composante": "type 1 (réassureurs, banques)", "scr": s1,
         "detail": f"LGD totale {detail['lgd'].sum():.1f} M€"},
        {"composante": "type 2 (créances)", "scr": s2, "detail": ""},
        {"composante": "SCR contrepartie (agrégé)", "scr": scr, "detail": "corrélation 0,75"},
    ])
    return table, scr
