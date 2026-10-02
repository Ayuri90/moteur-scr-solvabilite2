"""Model points du passif vie : épargne euro, UC, temporaire décès, rentes viagères."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import generateur, poids_normalises
from .courbe import Actualisation
from .mortalite import TableMortalite


def _ages(rng, n, moyenne, ecart, amin=18, amax=95) -> np.ndarray:
    return np.clip(np.round(rng.normal(moyenne, ecart, n)), amin, amax).astype(int)


def _taux_par_tranche(anciennete: np.ndarray, tranches: list[dict], cle: str) -> np.ndarray:
    res = np.full(len(anciennete), np.nan)
    for tr in tranches:
        m = (anciennete >= tr["anciennete_min"]) & (anciennete < tr["anciennete_max"])
        res[m] = tr[cle]
    return res


def _sexes(rng, n, part_hommes) -> np.ndarray:
    return np.where(rng.random(n) < part_hommes, "H", "F")


# ---------------------------------------------------------------------------
def epargne_euro(hyp: dict[str, Any], rng) -> pd.DataFrame:
    h = hyp["epargne_euro"]
    n = h["nb_model_points"]
    age = _ages(rng, n, h["age_moyen"], h["age_ecart_type"], h["age_min"], h["age_max"])
    anc_max = np.minimum(h["anciennete_max"], age - h["age_min"])
    anciennete = np.floor(rng.beta(1.3, 2.6, n) * (anc_max + 1)).astype(int)
    pm = poids_normalises(n, rng, 0.9) * h["pm_totale"]
    taille_moyenne = rng.lognormal(np.log(0.050), 0.5, n)          # M€ par contrat
    df = pd.DataFrame({
        "id_mp": [f"EUR{i + 1:04d}" for i in range(n)],
        "produit": "epargne_euro",
        "sexe": _sexes(rng, n, hyp["mortalite"]["part_hommes"]),
        "age": age,
        "anciennete": anciennete,
        "nb_contrats": np.maximum(1, np.round(pm / taille_moyenne)).astype(int),
        "pm": pm,
        "tmg": _taux_par_tranche(anciennete, h["tmg_par_generation"], "tmg"),
        "chargement_encours": h["chargement_encours"],
        "taux_rachat_structurel": _taux_par_tranche(anciennete, h["rachat_structurel"], "taux"),
        "taux_pb_financiere": h["taux_participation_financiere"],
        "taux_pb_technique": h["taux_participation_technique"],
    })
    return df


def epargne_uc(hyp: dict[str, Any], rng) -> pd.DataFrame:
    h = hyp["epargne_uc"]
    n = h["nb_model_points"]
    age = _ages(rng, n, h["age_moyen"], h["age_ecart_type"])
    anciennete = np.floor(rng.beta(1.2, 3.0, n) * (np.minimum(25, age - 18) + 1)).astype(int)
    pm = poids_normalises(n, rng, 0.9) * h["pm_totale"]
    garantie = rng.random(n) < h["garantie_plancher_part"]
    return pd.DataFrame({
        "id_mp": [f"UC{i + 1:04d}" for i in range(n)],
        "produit": "epargne_uc",
        "sexe": _sexes(rng, n, hyp["mortalite"]["part_hommes"]),
        "age": age,
        "anciennete": anciennete,
        "nb_contrats": np.maximum(1, np.round(pm / rng.lognormal(np.log(0.040), 0.5, n))).astype(int),
        "pm": pm,
        "versements_nets_cumules": pm / (1.0 + rng.normal(0.12, 0.12, n)).clip(0.7, None),
        "garantie_plancher_deces": garantie,
        "chargement_encours": h["chargement_encours"],
        "retrocession_encours": h["retrocession_encours"],
        "taux_rachat_structurel": h["rachat_structurel"],
    })


def temporaire_deces(hyp: dict[str, Any], tables: dict[str, TableMortalite], rng) -> pd.DataFrame:
    h = hyp["temporaire_deces"]
    n = h["nb_model_points"]
    age = _ages(rng, n, h["age_moyen"], h["age_ecart_type"], 20, 70)
    sexe = _sexes(rng, n, hyp["mortalite"]["part_hommes"])
    capital = rng.lognormal(np.log(h["capital_moyen"]), 0.6, n)       # M€ par assuré
    nb_assures = rng.integers(50, 600, n)
    qx = np.array([tables["homme" if s == "H" else "femme"].qx[a] for s, a in zip(sexe, age)])
    capitaux = capital * nb_assures
    prime = qx * capitaux * (1 + h["chargement_prime"])
    facteur = h["primes_annuelles"] / prime.sum()                        # calage des primes
    capitaux, prime = capitaux * facteur, prime * facteur
    return pd.DataFrame({
        "id_mp": [f"TD{i + 1:04d}" for i in range(n)],
        "produit": "temporaire_deces",
        "sexe": sexe, "age": age,
        "duree_residuelle": rng.integers(1, h["duree_residuelle_max"] + 1, n),
        "nb_assures": nb_assures,
        "capitaux_sous_risque": capitaux,
        "prime_annuelle": prime,
        "taux_chute": h["taux_chute"],
    })


def _annuite_reversion(t_princ: TableMortalite, t_conj: TableMortalite, age_x: int, age_y: int,
                       actu: Actualisation) -> float:
    """Valeur d'une rente de réversion de 1 : versée au conjoint après le décès du rentier."""
    n = 120 - min(age_x, age_y)
    px = t_princ.survie(age_x, n)[1:]
    py = t_conj.survie(age_y, n)[1:]
    df = actu.df(np.arange(1, n + 1))
    return float(np.sum(df * py * (1.0 - px)))


def rentes_viageres(hyp: dict[str, Any], tables: dict[str, TableMortalite], actu: Actualisation,
                    rng) -> pd.DataFrame:
    h = hyp["rentes_viageres"]
    n = h["nb_model_points"]
    age = _ages(rng, n, h["age_moyen"], h["age_ecart_type"], 60, 100)
    sexe = _sexes(rng, n, hyp["mortalite"]["part_hommes"])
    reversion = rng.random(n) < h["part_reversion"]
    age_conjoint = np.where(reversion, np.clip(age + np.where(sexe == "H", -3, 3)
                                                + rng.integers(-3, 4, n), 55, 100), 0)
    arrerages = poids_normalises(n, rng, 0.8)
    facteur = np.zeros(n)
    for i in range(n):
        tp = tables["rente_homme" if sexe[i] == "H" else "rente_femme"]
        tc = tables["rente_femme" if sexe[i] == "H" else "rente_homme"]
        a = tp.annuite_viagere(age[i], actu, terme_echu=False)
        if reversion[i]:
            a += h["taux_reversion"] * _annuite_reversion(tp, tc, age[i], age_conjoint[i], actu)
        facteur[i] = a * (1 + h["frais_gestion"])
    arrerages = arrerages * h["be_cible"] / np.sum(arrerages * facteur)   # calage du BE
    return pd.DataFrame({
        "id_mp": [f"RV{i + 1:04d}" for i in range(n)],
        "produit": "rente_viagere",
        "sexe": sexe, "age": age,
        "arrerages_annuels": arrerages,
        "reversion": reversion,
        "age_conjoint": np.where(reversion, age_conjoint, np.nan),
        "taux_reversion": np.where(reversion, h["taux_reversion"], 0.0),
        "frais_gestion": h["frais_gestion"],
        "be_indicatif": arrerages * facteur,
    })


def generer_passif_vie(hyp, tables, actu) -> dict[str, pd.DataFrame]:
    rng = generateur(hyp, 200)
    return {
        "mp_epargne_euro": epargne_euro(hyp, rng),
        "mp_epargne_uc": epargne_uc(hyp, rng),
        "mp_temporaire_deces": temporaire_deces(hyp, tables, rng),
        "mp_rentes_viageres": rentes_viageres(hyp, tables, actu, rng),
    }
