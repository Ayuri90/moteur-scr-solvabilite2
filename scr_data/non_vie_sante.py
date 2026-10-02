"""Données non-vie et santé : triangles, volumes de primes et réserves, expositions CAT,
model points santé SLT (incapacité / invalidité)."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .config import generateur, poids_normalises
from .courbe import Actualisation
from .mortalite import TableMortalite


# ---------------------------------------------------------------------------
# Triangles
# ---------------------------------------------------------------------------
def _parts_cumulees(cadence: list[float], facteur_queue: float) -> np.ndarray:
    """Part cumulée de l'ultime payée à chaque développement (la queue reste à payer)."""
    c = np.asarray(cadence, dtype=float)
    return c / (c[-1] * facteur_queue)


def reserve_theorique(p: dict[str, Any], premiere_annee: int, nb_annees: int) -> float:
    parts = _parts_cumulees(p["cadence_reglement"], p["facteur_queue"])
    reserve = 0.0
    for i in range(nb_annees):
        primes = p["primes_n"] / (1 + p["croissance_primes"]) ** (nb_annees - 1 - i)
        reserve += primes * p["sp_ultime"] * (1 - parts[nb_annees - 1 - i])
    return reserve


def simuler_triangle(nom: str, p: dict[str, Any], premiere_annee: int, nb_annees: int,
                     rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Simule un triangle de règlements cumulés et renvoie (triangle long, ultimes, facteur de calage)."""
    parts = _parts_cumulees(p["cadence_reglement"], p["facteur_queue"])
    facteur = p["provision_cible"] / reserve_theorique(p, premiere_annee, nb_annees)
    if not 0.85 <= facteur <= 1.15:
        warnings.warn(f"[{nom}] facteur de calage des ultimes = {facteur:.2f} : "
                      "cadence, S/P et provision cible sont peu cohérents.")
    lignes, ultimes = [], []
    for i in range(nb_annees):
        annee = premiere_annee + i
        primes = p["primes_n"] / (1 + p["croissance_primes"]) ** (nb_annees - 1 - i)
        cv = p["cv_sp"]
        ultime = primes * p["sp_ultime"] * facteur * rng.lognormal(-0.5 * np.log1p(cv**2),
                                                                     np.sqrt(np.log1p(cv**2)))
        increments_attendus = ultime * np.diff(np.concatenate([[0.0], parts]))
        k = 1.0 / p["cv_increments"] ** 2
        increments = increments_attendus * rng.gamma(k, 1.0 / k, nb_annees)
        dev_max = nb_annees - 1 - i
        cumul = np.cumsum(increments)
        for j in range(dev_max + 1):
            lignes.append({"lob": nom, "annee_survenance": annee, "developpement": j,
                           "reglements_incrementaux": increments[j],
                           "reglements_cumules": cumul[j]})
        ultimes.append({"lob": nom, "annee_survenance": annee, "primes_acquises": primes,
                        "ultime_vrai": ultime, "reglements_cumules_a_date": cumul[dev_max],
                        "reserve_vraie_attendue": ultime * (1 - parts[dev_max]),
                        "sp_ultime_vrai": ultime / primes})
    return pd.DataFrame(lignes), pd.DataFrame(ultimes), facteur


def triangle_en_matrice(tri_long: pd.DataFrame) -> pd.DataFrame:
    return tri_long.pivot(index="annee_survenance", columns="developpement",
                          values="reglements_cumules")


# ---------------------------------------------------------------------------
def generer_non_vie(hyp: dict[str, Any]) -> dict[str, Any]:
    rng = generateur(hyp, 300)
    hnv = hyp["non_vie"]
    triangles, ultimes, volumes, calages = [], [], [], {}
    for nom, p in hnv["lignes"].items():
        tri, ult, f = simuler_triangle(nom, p, hnv["premiere_annee_survenance"], hnv["nb_annees"], rng)
        triangles.append(tri)
        ultimes.append(ult)
        calages[nom] = f
        volumes.append(_volume(nom, p, "non_vie", hnv["reassurance"]["cession_primes"], ult))

    hs = hyp["sante_nslt"]
    tri, ult, f = simuler_triangle("sante_frais_soins", hs, hnv["premiere_annee_survenance"],
                                   hnv["nb_annees"], rng)
    triangles.append(tri)
    ultimes.append(ult)
    calages["sante_frais_soins"] = f
    volumes.append(_volume("sante_frais_soins", hs, "sante_nslt", 0.0, ult))

    return {
        "triangles_reglements": pd.concat(triangles, ignore_index=True),
        "ultimes_vrais": pd.concat(ultimes, ignore_index=True),
        "volumes_primes_reserves": pd.DataFrame(volumes),
        "exposition_cat": _expositions_cat(hyp, rng),
        "calages_triangles": calages,
    }


def _volume(nom, p, module, cession, ult: pd.DataFrame) -> dict[str, Any]:
    return {
        "lob": nom, "module": module, "lob_s2": p["lob_s2"],
        "primes_acquises_n": p["primes_n"],
        "primes_acquises_n_plus_1_estimees": p["primes_n_plus_1"],
        "fp_existant": 0.0, "fp_futur": 0.0,
        "taux_cession_reassurance": cession,
        "reserve_non_actualisee_vraie": ult["reserve_vraie_attendue"].sum(),
    }


def _expositions_cat(hyp, rng) -> pd.DataFrame:
    hc = hyp["non_vie"]["catastrophe"]
    n = hc["nb_zones_cresta"]
    w = rng.dirichlet(np.full(n, hc["concentration_dirichlet"]))
    zones = pd.DataFrame({"type": "capitaux_assures_mrh", "zone_cresta": [f"{i + 1:02d}" for i in range(n)],
                          "montant": w * hc["capitaux_assures_mrh"]})
    autres = pd.DataFrame([
        {"type": "nb_vehicules_assures", "zone_cresta": "FR", "montant": hc["nb_vehicules_assures"]},
        *[{"type": "concentration_incendie_200m", "zone_cresta": f"SITE_{k + 1}", "montant": m}
          for k, m in enumerate(hc["top_concentrations_incendie"])],
        {"type": "xl_cat_portee", "zone_cresta": "FR", "montant": hyp["non_vie"]["reassurance"]["xl_cat_portee"]},
        {"type": "xl_cat_priorite", "zone_cresta": "FR", "montant": hyp["non_vie"]["reassurance"]["xl_cat_priorite"]},
        {"type": "nb_assures_sante", "zone_cresta": "FR", "montant": hyp["sante_nslt"]["nb_assures"]},
    ])
    return pd.concat([zones, autres], ignore_index=True)


# ---------------------------------------------------------------------------
# Santé SLT : rentes d'incapacité et d'invalidité en cours de service
# ---------------------------------------------------------------------------
def generer_sante_slt(hyp: dict[str, Any], tables: dict[str, TableMortalite],
                      actu: Actualisation) -> pd.DataFrame:
    rng = generateur(hyp, 400)
    h = hyp["sante_slt"]
    n = h["nb_model_points"]
    invalidite = rng.random(n) < h["part_invalidite"]
    age = np.clip(np.round(rng.normal(h["age_moyen"], h["age_ecart_type"], n)), 25,
                  h["age_fin_garantie"] - 1).astype(int)
    sexe = np.where(rng.random(n) < hyp["mortalite"]["part_hommes"], "H", "F")
    anciennete = np.where(invalidite, rng.integers(0, 10, n), rng.uniform(0, 2, n).round(2))
    rente = poids_normalises(n, rng, 0.6)
    facteur = np.zeros(n)
    lam = 1.0 / h["duree_moyenne_incapacite"]
    for i in range(n):
        if invalidite[i]:
            duree = h["age_fin_garantie"] - age[i]
            t = tables["homme" if sexe[i] == "H" else "femme"]
            q = np.clip(t.qx[age[i]:age[i] + duree] * h["surmortalite_invalides"], 0, 1)
            p = np.cumprod(1 - q)
            facteur[i] = np.sum(p * actu.df(np.arange(1, duree + 1) - 0.5))
        else:
            reste = max(h["duree_max_incapacite"] - anciennete[i], 0.1)
            pas = np.arange(0.5, np.ceil(reste * 12)) / 12.0
            facteur[i] = np.sum(np.exp(-lam * pas) * actu.df(pas)) / 12.0
    rente = rente * h["be_cible"] / np.sum(rente * facteur)
    return pd.DataFrame({
        "id_mp": [f"SLT{i + 1:04d}" for i in range(n)],
        "garantie": np.where(invalidite, "invalidite", "incapacite"),
        "sexe": sexe, "age": age, "anciennete_sinistre": anciennete,
        "rente_annuelle": rente,
        "age_fin_garantie": h["age_fin_garantie"],
        "be_indicatif": rente * facteur,
    })
