"""Inventaire d'actifs ligne par ligne (fonds euro + UC en transparence).

Chaque obligation est valorisée de façon cohérente avec la courbe sans risque :
prix = somme des flux actualisés au taux sans risque + spread. Les nominaux sont
ensuite mis à l'échelle pour atteindre la valeur de marché cible de chaque classe.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import generateur, poids_normalises
from .courbe import Actualisation

COLONNES = [
    "id_ligne", "portefeuille", "classe", "sous_classe", "emetteur", "groupe_emetteur",
    "secteur", "pays", "devise", "cqs", "type_action", "nominal", "coupon", "maturite",
    "spread_bp", "prix_pct", "valeur_marche", "valeur_comptable", "duration_modifiee",
    "covered_bond", "infra_eligible", "usage_propre", "rendement_courant",
]


# ---------------------------------------------------------------------------
# Outils de valorisation obligataire
# ---------------------------------------------------------------------------
def prix_obligation(actu: Actualisation, coupon: float, maturite: int, spread: float) -> float:
    t = np.arange(1, maturite + 1)
    df = actu.df(t, spread)
    return float(100.0 * (coupon * df.sum() + df[-1]))


def duration_effective(actu: Actualisation, coupon: float, maturite: int, spread: float,
                       choc: float = 1e-4) -> float:
    p0 = prix_obligation(actu, coupon, maturite, spread)
    p_bas = prix_obligation(actu, coupon, maturite, spread - choc)
    p_haut = prix_obligation(actu, coupon, maturite, spread + choc)
    return (p_bas - p_haut) / (2.0 * p0 * choc)


def _tirer_maturites(rng, n, moyenne, mmin, mmax) -> np.ndarray:
    forme = 2.0
    m = rng.gamma(forme, moyenne / forme, size=n)
    return np.clip(np.ceil(m), mmin, mmax).astype(int)


def _coupon(rng, actu, maturite, spread, ecart) -> float:
    """Coupon ≈ rendement actuel +/- bruit (titres émis à des époques de taux différents)."""
    spot = float(actu.df(maturite) ** (-1.0 / maturite) - 1.0)
    c = spot + spread + rng.normal(-0.004, ecart)
    return max(0.0, round(c / 0.00125) * 0.00125)


def _lignes_obligataires(rng, actu, specs: list[dict], montant_cible: float, hyp_obl) -> pd.DataFrame:
    """Valorise une liste de lignes (dict) puis met à l'échelle les nominaux."""
    poids = poids_normalises(len(specs), rng, sigma=0.7)
    lignes = []
    for spec, w in zip(specs, poids):
        s = spec["spread_bp"] / 1e4
        c = _coupon(rng, actu, spec["maturite"], s, hyp_obl["ecart_coupon_historique"])
        prix = prix_obligation(actu, c, spec["maturite"], s)
        duree = duration_effective(actu, c, spec["maturite"], s)
        lignes.append({**spec, "coupon": c, "prix_pct": prix, "duration_modifiee": duree,
                       "poids": w, "rendement_courant": c * 100.0 / prix})
    df = pd.DataFrame(lignes)
    df["valeur_marche"] = df["poids"] * montant_cible
    df["nominal"] = df["valeur_marche"] / (df["prix_pct"] / 100.0)
    # coût amorti proche du pair, avec l'historique d'achat
    df["valeur_comptable"] = df["nominal"] * rng.normal(1.0, 0.02, len(df))
    return df.drop(columns="poids")


# ---------------------------------------------------------------------------
# Générateurs par classe
# ---------------------------------------------------------------------------
def _souverains(rng, actu, hyp, montant) -> pd.DataFrame:
    h = hyp["obligations"]["souverains"]
    pays = h["pays"]
    n_total = h["nb_lignes"]
    specs = []
    for code, p in pays.items():
        n = max(1, round(p["poids"] * n_total))
        mats = _tirer_maturites(rng, n, h["maturite_moyenne"],
                                hyp["obligations"]["maturite_residuelle_min"],
                                hyp["obligations"]["maturite_residuelle_max"])
        for m in mats:
            specs.append({"sous_classe": "souverain", "emetteur": f"ETAT_{code}",
                          "groupe_emetteur": f"ETAT_{code}", "secteur": "administration_publique",
                          "pays": code, "devise": "EUR", "cqs": p["cqs"], "maturite": int(m),
                          "spread_bp": p["spread_bp"] * rng.uniform(0.9, 1.1)})
    df = _lignes_obligataires(rng, actu, specs, 0.0, hyp["obligations"])
    # mise à l'échelle par pays pour respecter les poids
    for code, p in pays.items():
        masque = df["pays"] == code
        poids = poids_normalises(int(masque.sum()), rng, 0.6)
        df.loc[masque, "valeur_marche"] = poids * p["poids"] * montant
    df["nominal"] = df["valeur_marche"] / (df["prix_pct"] / 100.0)
    df["valeur_comptable"] = df["nominal"] * rng.normal(1.0, 0.02, len(df))
    df["classe"] = "obligation"
    return df


def _corporates(rng, actu, hyp, montant, cle, prefixe, secteur) -> tuple[pd.DataFrame, list[str]]:
    h = hyp["obligations"][cle]
    ho = hyp["obligations"]
    cqs_possibles = np.array(list(h["repartition_cqs"].keys()), dtype=int)
    probas = np.array(list(h["repartition_cqs"].values()), dtype=float)
    emetteurs = [f"{prefixe}_{i:03d}" for i in range(1, h["nb_emetteurs"] + 1)]
    cqs_emetteur = dict(zip(emetteurs, rng.choice(cqs_possibles, size=len(emetteurs), p=probas)))
    taille_emetteur = rng.lognormal(0, 0.9, len(emetteurs))
    choix = rng.choice(emetteurs, size=h["nb_lignes"], p=taille_emetteur / taille_emetteur.sum())
    mats = _tirer_maturites(rng, h["nb_lignes"], h["maturite_moyenne"],
                            ho["maturite_residuelle_min"], ho["maturite_residuelle_max"])
    part_covered = h.get("part_covered_bonds", 0.0)
    specs = []
    for em, m in zip(choix, mats):
        covered = bool(rng.random() < part_covered)
        cqs = 0 if covered else int(cqs_emetteur[em])
        spread_moyen = ho["spread_par_cqs_bp"][cqs] * (0.6 if covered else 1.0)
        spread = spread_moyen * rng.lognormal(0, ho["dispersion_spread"])
        pays = rng.choice(["FR", "DE", "NL", "IT", "ES", "BE", "US", "GB"],
                          p=[0.40, 0.15, 0.10, 0.08, 0.08, 0.05, 0.09, 0.05])
        specs.append({"sous_classe": "covered_bond" if covered else "corporate",
                      "emetteur": em, "groupe_emetteur": em, "secteur": secteur,
                      "pays": pays, "devise": "EUR", "cqs": cqs, "maturite": int(m),
                      "spread_bp": spread, "covered_bond": covered})
    df = _lignes_obligataires(rng, actu, specs, montant, ho)
    df["classe"] = "obligation"
    return df, emetteurs


def _actions(rng, hyp, alloc, placements, emetteurs_corporate) -> pd.DataFrame:
    ha = hyp["actions"]
    lignes = []
    # type 1 : une partie des émetteurs sont aussi émetteurs obligataires (concentration)
    n1 = ha["type1"]["nb_lignes"]
    montant1 = alloc["actions_type1"] * placements
    w1 = poids_normalises(n1, rng, 0.8)
    zone_euro = ["FR", "DE", "NL", "IT", "ES"]
    hors_zone = {"US": "USD", "CH": "CHF", "GB": "GBP", "JP": "JPY"}
    for i in range(n1):
        if rng.random() < ha["type1"]["part_zone_euro"]:
            pays, devise = rng.choice(zone_euro), "EUR"
        else:
            pays = rng.choice(list(hors_zone))
            devise = hors_zone[pays]
        if i < n1 // 2 and devise == "EUR":
            em = rng.choice(emetteurs_corporate)
        else:
            em = f"ACT_{i + 1:03d}"
        vm = w1[i] * montant1
        lignes.append({"classe": "action", "sous_classe": "action_cotee", "emetteur": em,
                       "groupe_emetteur": em, "secteur": "entreprise", "pays": pays,
                       "devise": devise, "type_action": 1, "valeur_marche": vm,
                       "valeur_comptable": vm / (1 + rng.uniform(0.05, 0.35)),
                       "rendement_courant": ha["type1"]["rendement_dividende"]})
    # type 2 : private equity
    n2 = ha["type2"]["nb_lignes"]
    w2 = poids_normalises(n2, rng, 0.5)
    for i in range(n2):
        vm = w2[i] * alloc["actions_type2"] * placements
        lignes.append({"classe": "action", "sous_classe": "private_equity",
                       "emetteur": f"PE_FONDS_{i + 1:02d}", "groupe_emetteur": f"PE_FONDS_{i + 1:02d}",
                       "secteur": "fonds", "pays": "FR", "devise": "EUR", "type_action": 2,
                       "valeur_marche": vm, "valeur_comptable": vm / (1 + rng.uniform(0.0, 0.25)),
                       "rendement_courant": 0.0})
    # participations stratégiques
    noms = ["FILIALE_SERVICES", "PARTICIPATION_MUTUELLE"]
    w3 = poids_normalises(ha["participations"]["nb_lignes"], rng, 0.3)
    for nom, w in zip(noms, w3):
        vm = w * alloc["participations_strategiques"] * placements
        lignes.append({"classe": "action", "sous_classe": "participation_strategique",
                       "emetteur": nom, "groupe_emetteur": nom, "secteur": "participation",
                       "pays": "FR", "devise": "EUR", "type_action": 3, "valeur_marche": vm,
                       "valeur_comptable": vm / 1.1, "rendement_courant": 0.02})
    return pd.DataFrame(lignes)


def _immobilier(rng, hyp, montant) -> pd.DataFrame:
    hi = hyp["immobilier"]
    n = hi["nb_lignes"]
    w = poids_normalises(n, rng, 0.6)
    w = np.sort(w)[::-1]
    usage = np.zeros(n, dtype=bool)
    cumul = np.cumsum(w)
    usage[: max(1, np.searchsorted(cumul, hi["part_usage_propre"]))] = True
    lignes = []
    for i in range(n):
        vm = w[i] * montant
        lignes.append({"classe": "immobilier",
                       "sous_classe": "immeuble_exploitation" if usage[i] else "immeuble_placement",
                       "emetteur": f"IMMO_{i + 1:02d}", "groupe_emetteur": f"IMMO_{i + 1:02d}",
                       "secteur": "immobilier", "pays": "FR", "devise": "EUR",
                       "valeur_marche": vm, "valeur_comptable": vm / (1 + rng.uniform(0.2, 0.5)),
                       "usage_propre": bool(usage[i]), "rendement_courant": hi["rendement_locatif"]})
    return pd.DataFrame(lignes)


def _monetaire(rng, actu, hyp, montant) -> pd.DataFrame:
    hm = hyp["monetaire"]
    lignes = []
    depots = hm["part_depots_bancaires"] * montant
    for banque, p in hm["banques"].items():
        vm = p["poids"] * depots
        lignes.append({"classe": "depot", "sous_classe": "depot_bancaire", "emetteur": banque,
                       "groupe_emetteur": banque, "secteur": "banque", "pays": "FR",
                       "devise": "EUR", "cqs": p["cqs"], "maturite": 1, "nominal": vm,
                       "valeur_marche": vm, "valeur_comptable": vm, "duration_modifiee": 0.0,
                       "rendement_courant": float(actu.df(1) ** -1 - 1)})
    # OPC monétaires en transparence : titres courts
    reste = montant - depots
    specs = []
    for i in range(12):
        cqs = int(rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3]))
        specs.append({"sous_classe": "titre_court_terme_opc", "emetteur": f"CT_{i + 1:02d}",
                      "groupe_emetteur": f"CT_{i + 1:02d}", "secteur": "financier", "pays": "FR",
                      "devise": "EUR", "cqs": cqs, "maturite": 1,
                      "spread_bp": hyp["obligations"]["spread_par_cqs_bp"][cqs] * 0.4})
    opc = _lignes_obligataires(rng, actu, specs, reste, hyp["obligations"])
    opc["classe"] = "monetaire"
    return pd.concat([pd.DataFrame(lignes), opc], ignore_index=True)


def _prets_infra(rng, actu, hyp, montant) -> pd.DataFrame:
    hp = hyp["prets_infrastructure"]
    n = hp["nb_lignes"]
    specs = []
    n_infra = max(1, round(hp["part_infra_eligible"] * n))
    for i in range(n):
        infra = i < n_infra
        cqs = int(np.clip(hp["cqs_moyen"] + rng.choice([-1, 0, 0, 1]), 1, 4))
        specs.append({"sous_classe": "infrastructure_eligible" if infra else "pret_entreprise",
                      "emetteur": f"{'INFRA' if infra else 'PRET'}_{i + 1:02d}",
                      "groupe_emetteur": f"{'INFRA' if infra else 'PRET'}_{i + 1:02d}",
                      "secteur": "infrastructure" if infra else "entreprise", "pays": "FR",
                      "devise": "EUR", "cqs": cqs, "maturite": int(rng.integers(8, 21)),
                      "spread_bp": hyp["obligations"]["spread_par_cqs_bp"][cqs] * 1.2,
                      "infra_eligible": infra})
    df = _lignes_obligataires(rng, actu, specs, 0.0, hyp["obligations"])
    # respect de la part infrastructure en valeur de marché
    masque = df["infra_eligible"].astype(bool)
    for m, part in ((masque, hp["part_infra_eligible"]), (~masque, 1 - hp["part_infra_eligible"])):
        w = poids_normalises(m.sum(), rng, 0.4)
        df.loc[m, "valeur_marche"] = w * part * montant
    df["nominal"] = df["valeur_marche"] / (df["prix_pct"] / 100.0)
    df["valeur_comptable"] = df["nominal"]
    df["classe"] = "pret_infrastructure"
    return df


def _uc_transparence(rng, actu, hyp, montant) -> pd.DataFrame:
    au = hyp["uc"]["allocation"]
    lignes = []
    supports_actions = {"UC_ACTIONS_FRANCE": ("FR", "EUR", 0.30), "UC_ACTIONS_EUROPE": ("DE", "EUR", 0.30),
                        "UC_ACTIONS_MONDE": ("US", "USD", 0.30), "UC_ACTIONS_EMERGENTS": ("XX", "USD", 0.10)}
    for nom, (pays, devise, w) in supports_actions.items():
        vm = w * au["actions"] * montant
        lignes.append({"classe": "action", "sous_classe": "action_cotee", "emetteur": nom,
                       "groupe_emetteur": nom, "secteur": "fonds_uc", "pays": pays, "devise": devise,
                       "type_action": 2 if nom == "UC_ACTIONS_EMERGENTS" else 1,
                       "valeur_marche": vm, "valeur_comptable": vm, "rendement_courant": 0.025})
    specs = [
        {"sous_classe": "souverain", "emetteur": "UC_OBLIG_ETAT", "groupe_emetteur": "UC_OBLIG_ETAT",
         "secteur": "fonds_uc", "pays": "FR", "devise": "EUR", "cqs": 1, "maturite": 8, "spread_bp": 70},
        {"sous_classe": "corporate", "emetteur": "UC_OBLIG_IG", "groupe_emetteur": "UC_OBLIG_IG",
         "secteur": "fonds_uc", "pays": "FR", "devise": "EUR", "cqs": 2, "maturite": 5, "spread_bp": 90},
        {"sous_classe": "corporate", "emetteur": "UC_OBLIG_HY", "groupe_emetteur": "UC_OBLIG_HY",
         "secteur": "fonds_uc", "pays": "FR", "devise": "EUR", "cqs": 4, "maturite": 4, "spread_bp": 300},
    ]
    obl = _lignes_obligataires(rng, actu, specs, au["obligations"] * montant, hyp["obligations"])
    obl["classe"] = "obligation"
    obl["valeur_comptable"] = obl["valeur_marche"]
    lignes.append({"classe": "immobilier", "sous_classe": "scpi_opci", "emetteur": "UC_SCPI",
                   "groupe_emetteur": "UC_SCPI", "secteur": "fonds_uc", "pays": "FR", "devise": "EUR",
                   "valeur_marche": au["immobilier"] * montant, "valeur_comptable": au["immobilier"] * montant,
                   "rendement_courant": 0.045})
    lignes.append({"classe": "monetaire", "sous_classe": "opc_monetaire", "emetteur": "UC_MONETAIRE",
                   "groupe_emetteur": "UC_MONETAIRE", "secteur": "fonds_uc", "pays": "FR", "devise": "EUR",
                   "cqs": 1, "maturite": 1, "valeur_marche": au["monetaire"] * montant,
                   "valeur_comptable": au["monetaire"] * montant, "duration_modifiee": 0.25,
                   "rendement_courant": 0.02})
    df = pd.concat([pd.DataFrame(lignes), obl], ignore_index=True)
    df["portefeuille"] = "uc"
    return df


# ---------------------------------------------------------------------------
def generer_inventaire(hyp: dict[str, Any], actu: Actualisation) -> pd.DataFrame:
    rng = generateur(hyp, 100)
    alloc = hyp["allocation"]
    P = hyp["bilan_cible"]["actif"]["placements_hors_uc"]

    fin, em_fin = _corporates(rng, actu, hyp, alloc["corporate_financieres"] * P,
                              "corporate_financieres", "FIN", "financier")
    nfin, em_nfin = _corporates(rng, actu, hyp, alloc["corporate_non_financieres"] * P,
                                "corporate_non_financieres", "CORP", "non_financier")
    blocs = [
        _souverains(rng, actu, hyp, alloc["souverains"] * P),
        fin, nfin,
        _actions(rng, hyp, alloc, P, em_fin + em_nfin),
        _immobilier(rng, hyp, alloc["immobilier"] * P),
        _monetaire(rng, actu, hyp, alloc["monetaire_depots"] * P),
        _prets_infra(rng, actu, hyp, alloc["prets_infrastructure"] * P),
    ]
    euro = pd.concat(blocs, ignore_index=True)
    euro["portefeuille"] = "euro"
    uc = _uc_transparence(rng, actu, hyp, hyp["bilan_cible"]["actif"]["actifs_uc"])
    inv = pd.concat([euro, uc], ignore_index=True)

    # nettoyage et typage
    for col in COLONNES:
        if col not in inv.columns:
            inv[col] = np.nan
    for col in ("covered_bond", "infra_eligible", "usage_propre"):
        inv[col] = inv[col].fillna(False).astype(bool)
    inv["id_ligne"] = [f"A{i + 1:05d}" for i in range(len(inv))]
    inv["cqs"] = inv["cqs"].astype("Int64")
    inv["type_action"] = inv["type_action"].astype("Int64")
    inv["maturite"] = inv["maturite"].astype("Int64")
    inv.loc[inv["classe"].isin(["action", "immobilier"]), "duration_modifiee"] = np.nan
    return inv[COLONNES]
