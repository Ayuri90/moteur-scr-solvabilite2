"""Agrégation finale : BSCR, risque opérationnel, ajustements (LAC TP et LAC DT), SCR, MCR."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def bscr(scr_modules: dict[str, float], params: dict[str, Any]) -> tuple[float, pd.DataFrame]:
    ordre = params["bscr"]["ordre"]
    v = np.array([scr_modules[m] for m in ordre])
    M = np.array(params["bscr"]["matrice"], dtype=float)
    total = float(np.sqrt(v @ M @ v))
    table = pd.DataFrame({"module": ordre, "scr": v})
    table.loc[len(table)] = {"module": "BSCR", "scr": total}
    table["part_du_bscr"] = table["scr"] / total
    return total, table


def operationnel(hyp: dict[str, Any], be: dict[str, float], bscr_valeur: float,
                 params: dict[str, Any], donnees: dict[str, pd.DataFrame]) -> tuple[float, str]:
    """Art. 204 : maximum des assiettes primes et provisions, plafonné à 30 % du BSCR."""
    p = params["operationnel"]
    # primes vie hors UC (les contrats en UC passent par la composante frais)
    primes_vie = (hyp["epargne_euro"]["collecte_annuelle"]
                  + donnees["mp_temporaire_deces"]["prime_annuelle"].sum())
    primes_non_vie = (sum(l["primes_n"] for l in hyp["non_vie"]["lignes"].values())
                      + hyp["sante_nslt"]["primes_n"] + hyp["sante_slt"]["primes_annuelles"])

    pt_vie_hors_uc = be["epargne_euro"] + be["temporaire_deces"] + be["rentes"] + be["sante_slt"]
    pt_non_vie = be["non_vie_sinistres"] + be["non_vie_primes"] + be["sante_nslt"]

    op_primes = p["facteur_primes_vie"] * primes_vie + p["facteur_primes_non_vie"] * primes_non_vie
    op_provisions = (p["facteur_provisions_vie"] * max(pt_vie_hors_uc, 0.0)
                     + p["facteur_provisions_non_vie"] * max(pt_non_vie, 0.0))
    base = max(op_primes, op_provisions)
    plafonne = min(p["plafond_bscr"] * bscr_valeur, base)
    frais_uc = hyp["frais"]["gestion_pm_uc"] * hyp["epargne_uc"]["pm_totale"]
    total = plafonne + p["facteur_frais_uc"] * frais_uc
    detail = (f"primes {op_primes:.1f} / provisions {op_provisions:.1f} / "
              f"plafond 30 % du BSCR {p['plafond_bscr'] * bscr_valeur:.1f}")
    return float(total), detail


def ajustements(bscr_valeur: float, scr_op: float, hyp: dict[str, Any],
                params: dict[str, Any]) -> tuple[float, float, str]:
    """LAC TP (implicite en V1) et LAC DT plafonnée par les impôts différés disponibles."""
    lac_tp = -abs(params["lac"]["tp_explicite"])
    plafond = params["fiscalite"]["taux_impot"] * (bscr_valeur + scr_op + lac_tp)
    idp = hyp["bilan_cible"]["passif"]["impots_differes_passifs"]
    capacite = idp + params["lac"]["dt_part_benefices_futurs"] * max(plafond - idp, 0.0)
    lac_dt = -min(plafond, capacite)
    detail = f"plafond {plafond:.1f} / IDP {idp:.1f} / capacité retenue {capacite:.1f}"
    return lac_tp, float(lac_dt), detail


def mcr(scr: float, be: dict[str, float], donnees: dict[str, pd.DataFrame], hyp: dict[str, Any],
        params: dict[str, Any], detail_non_vie: dict[str, Any]) -> tuple[float, pd.DataFrame]:
    p = params["mcr"]
    capitaux = donnees["mp_temporaire_deces"]["capitaux_sous_risque"].sum()
    lineaire_vie = (p["vie"]["be_epargne_garantie"] * max(be["epargne_euro"], 0.0)
                    + p["vie"]["be_uc"] * max(be["epargne_uc"], 0.0)
                    + p["vie"]["be_autres_vie"] * max(be["rentes"] + be["temporaire_deces"], 0.0)
                    + p["vie"]["capitaux_sous_risque"] * capitaux)
    volumes = donnees["volumes_primes_reserves"].set_index("lob")
    lineaire_non_vie = 0.0
    for lob, (f_be, f_primes) in p["non_vie"].items():
        be_lob = detail_non_vie["be_sinistres"].get(lob, 0.0)
        primes = volumes.loc[lob, "primes_acquises_n"] if lob in volumes.index else 0.0
        lineaire_non_vie += f_be * max(be_lob, 0.0) + f_primes * primes
    lineaire = lineaire_vie + lineaire_non_vie
    combine = min(max(lineaire, p["plancher_scr"] * scr), p["plafond_scr"] * scr)
    valeur = max(combine, p["minimum_absolu"])
    table = pd.DataFrame([
        {"poste": "MCR linéaire vie", "montant": lineaire_vie},
        {"poste": "MCR linéaire non-vie et santé", "montant": lineaire_non_vie},
        {"poste": "MCR linéaire total", "montant": lineaire},
        {"poste": "Plancher (25 % du SCR)", "montant": p["plancher_scr"] * scr},
        {"poste": "Plafond (45 % du SCR)", "montant": p["plafond_scr"] * scr},
        {"poste": "Minimum absolu", "montant": p["minimum_absolu"]},
        {"poste": "MCR retenu", "montant": valeur},
    ])
    return float(valeur), table


def tableau_s25(scr_modules: dict[str, float], bscr_valeur: float, scr_op: float,
                lac_tp: float, lac_dt: float, scr: float) -> pd.DataFrame:
    libelles = {"marche": "Risque de marché", "defaut": "Risque de défaut de la contrepartie",
                "vie": "Risque de souscription vie", "sante": "Risque de souscription santé",
                "non_vie": "Risque de souscription non-vie"}
    lignes = [{"code": f"R00{10 + 10 * i}", "poste": libelles[m], "montant": scr_modules[m]}
              for i, m in enumerate(["marche", "defaut", "vie", "sante", "non_vie"])]
    diversification = bscr_valeur - sum(scr_modules.values())
    lignes += [
        {"code": "R0060", "poste": "Diversification", "montant": diversification},
        {"code": "R0070", "poste": "Risque lié aux immobilisations incorporelles", "montant": 0.0},
        {"code": "R0100", "poste": "Capital de solvabilité requis de base (BSCR)", "montant": bscr_valeur},
        {"code": "R0130", "poste": "Risque opérationnel", "montant": scr_op},
        {"code": "R0140", "poste": "Capacité d'absorption des provisions techniques", "montant": lac_tp},
        {"code": "R0150", "poste": "Capacité d'absorption des impôts différés", "montant": lac_dt},
        {"code": "R0200", "poste": "Capital de solvabilité requis (SCR)", "montant": scr},
    ]
    return pd.DataFrame(lignes)
