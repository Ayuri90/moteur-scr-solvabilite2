"""Modèle ALM stochastique du fonds euro (V2).

Projection conjointe actif-passif par cohortes, sur les scénarios de l'ESG :
rendement comptable, réalisation de plus-values latentes, participation aux bénéfices,
PPE, rachats structurels et conjoncturels (lois ONC), puis actualisation par les déflateurs.

Deux jeux d'actions du management sont disponibles :
- `dynamique` : la revalorisation s'adapte au rendement financier (calcul net) ;
- `figee`    : la revalorisation reste au niveau cible quoi qu'il arrive, ce qui donne le
  calcul brut d'absorption par les prestations discrétionnaires futures ;
- `garantie` : seul le TMG est servi, ce qui isole la partie garantie du BE et donc les FDB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from scr.esg import Scenarios
from scr_data.mortalite import TableMortalite

Regime = Literal["dynamique", "figee", "garantie"]


# ---------------------------------------------------------------------------
def construire_cohortes(mp: pd.DataFrame, hyp: dict[str, Any], tables: dict[str, TableMortalite],
                        horizon: int) -> pd.DataFrame:
    """Regroupe les model points par TMG et tranche d'âge, avec une table de mortalité moyenne."""
    n_tranches = hyp["alm"]["nb_tranches_age"]
    mp = mp.copy()
    mp["tranche_age"] = pd.qcut(mp["age"], n_tranches, labels=False, duplicates="drop")
    agg = (mp.groupby(["tmg", "tranche_age"])
           .apply(lambda g: pd.Series({
               "pm": g["pm"].sum(),
               "age_moyen": int(round(np.average(g["age"], weights=g["pm"]))),
               "chargement": float(np.average(g["chargement_encours"], weights=g["pm"])),
               "rachat_structurel": float(np.average(g["taux_rachat_structurel"], weights=g["pm"])),
               "part_hommes": float((g["sexe"] == "H").mean()),
           }), include_groups=False)
           .reset_index())
    # probabilités de décès projetées, moyenne pondérée hommes / femmes
    qx = []
    for r in agg.itertuples():
        ages = np.clip(np.arange(int(r.age_moyen), int(r.age_moyen) + horizon), 0, 120)
        qh = tables["homme"].qx[ages]
        qf = tables["femme"].qx[ages]
        qx.append(r.part_hommes * qh + (1 - r.part_hommes) * qf)
    agg.attrs["qx"] = np.array(qx)            # (nb_cohortes, horizon)
    return agg


def construire_actif_euro(inventaire: pd.DataFrame, hyp: dict[str, Any]) -> dict[str, float]:
    """Agrège l'inventaire du fonds euro en quatre poches, avec leurs plus-values latentes."""
    euro = inventaire[inventaire["portefeuille"] == "euro"]
    poches = {"obligation": ["obligation", "pret_infrastructure"], "action": ["action"],
              "immobilier": ["immobilier"], "monetaire": ["monetaire", "depot"]}
    actif = {}
    for nom, classes in poches.items():
        sous = euro[euro["classe"].isin(classes)]
        actif[f"vm_{nom}"] = float(sous["valeur_marche"].sum())
        actif[f"vnc_{nom}"] = float(sous["valeur_comptable"].sum())
    oblig = euro[euro["classe"].isin(poches["obligation"])]
    poids = oblig["valeur_marche"]
    actif["taux_comptable_obligations"] = float(np.average(oblig["rendement_courant"], weights=poids))
    actif["duration_obligations"] = float(np.average(oblig["duration_modifiee"].fillna(0.0), weights=poids))
    return actif


# ---------------------------------------------------------------------------
@dataclass
class ResultatALM:
    be: float
    be_par_scenario: np.ndarray
    flux_moyens: pd.DataFrame
    taux_servi_moyen: np.ndarray
    ppe_finale: float
    rachats_moyens: np.ndarray


def _taux_rachat_conjoncturel(ecart: np.ndarray, lois: dict[str, Any]) -> np.ndarray:
    """Loi ONC : rachats conjoncturels en fonction de l'écart entre taux servi et taux attendu."""
    p = {cle: np.mean([lois["plafond"][cle], lois["plancher"][cle]])
         for cle in ("alpha", "beta", "gamma", "delta", "rc_min", "rc_max")}
    rc = np.zeros_like(ecart)
    rc = np.where(ecart < p["alpha"], p["rc_max"], rc)
    zone = (ecart >= p["alpha"]) & (ecart < p["beta"])
    rc = np.where(zone, p["rc_max"] * (ecart - p["beta"]) / (p["alpha"] - p["beta"]), rc)
    zone = (ecart > p["gamma"]) & (ecart <= p["delta"])
    rc = np.where(zone, p["rc_min"] * (ecart - p["gamma"]) / (p["delta"] - p["gamma"]), rc)
    rc = np.where(ecart > p["delta"], p["rc_min"], rc)
    return rc


class ModeleALM:
    def __init__(self, cohortes: pd.DataFrame, actif: dict[str, float], hyp: dict[str, Any]):
        self.cohortes = cohortes
        self.actif0 = actif
        self.hyp = hyp
        self.qx = cohortes.attrs["qx"]

    def projeter(self, sc: Scenarios, regime: Regime = "dynamique",
                 facteur_rachat: float = 1.0, rachat_massif: float = 0.0,
                 facteur_frais: float = 1.0, facteur_actif: dict[str, float] | None = None,
                 ppe_initiale: float | None = None) -> ResultatALM:
        h = self.hyp
        p_alm, p_epargne = h["alm"], h["epargne_euro"]
        H = min(p_alm["horizon"], sc.horizon)
        n = sc.nb_scenarios
        facteur_actif = facteur_actif or {}

        # --- état initial ---------------------------------------------------
        pm = np.tile(self.cohortes["pm"].to_numpy(float), (n, 1))          # (n, cohortes)
        tmg = self.cohortes["tmg"].to_numpy(float)[None, :]
        chargement = self.cohortes["chargement"].to_numpy(float)[None, :]
        rachat_struct = self.cohortes["rachat_structurel"].to_numpy(float)[None, :] * facteur_rachat
        # l'actif adossé au canton épargne euro est proportionnel à ses engagements
        ppe0 = p_epargne["ppe_fdb"] if ppe_initiale is None else ppe_initiale
        total_vm0 = sum(self.actif0[f"vm_{c}"] for c in
                        ("obligation", "action", "immobilier", "monetaire"))
        allocation = (self.cohortes["pm"].sum() + ppe0) / total_vm0
        vm = {c: self.actif0[f"vm_{c}"] * allocation * facteur_actif.get(c, 1.0)
              for c in ("obligation", "action", "immobilier", "monetaire")}
        vnc = {c: self.actif0[f"vnc_{c}"] * allocation for c in vm}
        vm = {c: np.full(n, v) for c, v in vm.items()}
        vnc = {c: np.full(n, v) for c, v in vnc.items()}
        ppe = np.full(n, ppe0)
        taux_book = np.full(n, self.actif0["taux_comptable_obligations"])
        duration = self.actif0["duration_obligations"]
        vitesse = 1.0 / p_alm["maturite_moyenne_obligations"]
        alpha_pb = p_epargne["taux_participation_financiere"]
        frais_pm = h["frais"]["gestion_pm_euro"] * facteur_frais
        inflation = h["frais"]["inflation"]

        # rachat massif immédiat
        prestations_immediates = pm.sum(axis=1) * rachat_massif
        pm *= (1 - rachat_massif)
        for c in vm:
            part = vm[c] / sum(vm.values())
            vm[c] = vm[c] - part * prestations_immediates
            vnc[c] = np.minimum(vnc[c], vm[c])

        flux = np.zeros((n, H))
        taux_servis, taux_rachats = np.zeros((n, H)), np.zeros((n, H))
        memoire_10ans = [float(sc.taux_10ans[:, 0].mean())] * p_alm["taux_concurrence"]["moyenne_mobile"]

        for t in range(H):
            # --- revenus financiers et valeurs de marché ---------------------
            revenus = (taux_book * vnc["obligation"]
                       + h["esg"]["action"]["rendement_dividende"] * vm["action"]
                       + h["esg"]["immobilier"]["rendement_locatif"] * vm["immobilier"]
                       + sc.taux_court[:, t] * vm["monetaire"])
            variation_taux = (sc.taux_10ans[:, min(t + 1, H - 1)] - sc.taux_10ans[:, t])
            vm["obligation"] *= (1 + taux_book - duration * variation_taux)
            vm["action"] *= (1 + sc.rendement_action[:, t] - h["esg"]["action"]["rendement_dividende"])
            vm["immobilier"] *= (1 + sc.rendement_immobilier[:, t] - h["esg"]["immobilier"]["rendement_locatif"])
            vm["monetaire"] *= (1 + sc.taux_court[:, t])
            pvl = np.maximum(vm["action"] - vnc["action"], 0.0)

            # --- taux cible servi par la concurrence -------------------------
            memoire_10ans.append(float(sc.taux_10ans[:, t].mean()))
            lissage = np.mean(memoire_10ans[-p_alm["taux_concurrence"]["moyenne_mobile"]:])
            taux_cible = (p_alm["taux_concurrence"]["coefficient_taux_10ans"] * lissage
                          + p_alm["taux_concurrence"]["constante"])

            # --- rendement distribuable et participation aux bénéfices -------
            base_comptable = sum(vnc.values()) + vm["monetaire"] - vnc["monetaire"]
            besoin = np.maximum(taux_cible * pm.sum(axis=1) - alpha_pb * revenus, 0.0)
            realisation = np.minimum(p_alm["part_pvl_actions_realisee"] * pvl, besoin)
            vnc["action"] += realisation
            rendement = alpha_pb * (revenus + realisation) / np.maximum(base_comptable, 1e-9)

            if regime == "garantie":
                servi = np.broadcast_to(tmg, pm.shape).copy()
                dotation = np.zeros(n)
            else:
                if regime == "figee":
                    cible = np.full(n, taux_cible)
                else:
                    reprise_max = ppe / np.maximum(pm.sum(axis=1), 1e-9) / 1.0
                    cible = np.minimum(taux_cible, rendement + reprise_max)
                brut = np.where(regime == "figee", cible, np.minimum(rendement, cible))
                brut = np.maximum(brut, 0.0)
                servi = np.maximum(brut[:, None] - chargement, np.broadcast_to(tmg, pm.shape))
                cout = (servi * pm).sum(axis=1)
                produit = rendement * np.maximum(base_comptable, 1e-9)
                excedent = produit - cout
                dotation = np.where(excedent > 0,
                                    np.minimum(excedent * p_alm["ppe"]["dotation_max_rendement"],
                                               excedent), excedent)
                ppe = np.maximum(ppe + dotation, 0.0)

            # --- reprise obligatoire de la PPE (8 ans au plus, Art. L.331-3) --
            if regime != "garantie":
                reprise = ppe / p_alm["ppe"]["duree_reprise_max"]
                ppe -= reprise
                servi = servi + (reprise / np.maximum(pm.sum(axis=1), 1e-9))[:, None]

            # --- rachats et décès --------------------------------------------
            servi_moyen = (servi * pm).sum(axis=1) / np.maximum(pm.sum(axis=1), 1e-9)
            ecart = servi_moyen - taux_cible
            conjoncturel = _taux_rachat_conjoncturel(ecart, p_epargne["rachat_conjoncturel"])
            taux_rachat = np.clip(rachat_struct + conjoncturel[:, None], 0.0, 1.0)
            q = self.qx[:, t][None, :]
            pm_revalorisee = pm * (1 + servi)
            sorties = pm_revalorisee * (q + taux_rachat - q * taux_rachat)
            frais = pm.sum(axis=1) * frais_pm * (1 + inflation) ** t
            flux[:, t] = sorties.sum(axis=1) + frais
            taux_servis[:, t] = servi_moyen
            taux_rachats[:, t] = (taux_rachat * pm).sum(axis=1) / np.maximum(pm.sum(axis=1), 1e-9)
            pm = pm_revalorisee - sorties

            # --- adossement de l'actif ---------------------------------------
            sortie_totale = flux[:, t]
            total_vm = sum(vm.values())
            for c in vm:
                part = vm[c] / np.maximum(total_vm, 1e-9)
                vm[c] = np.maximum(vm[c] - part * sortie_totale, 0.0)
                vnc[c] = np.minimum(vnc[c], vm[c])
            taux_book = taux_book * (1 - vitesse) + sc.taux_10ans[:, t] * vitesse

        # clôture : PM résiduelle et PPE reversées aux assurés
        flux[:, H - 1] += pm.sum(axis=1) + ppe
        be_scenario = (flux * sc.deflateur[:, :H]).sum(axis=1) + prestations_immediates
        moyens = pd.DataFrame({"annee": np.arange(1, H + 1), "flux_moyen": flux.mean(axis=0),
                               "taux_servi_moyen": taux_servis.mean(axis=0),
                               "taux_rachat_moyen": taux_rachats.mean(axis=0)})
        return ResultatALM(float(be_scenario.mean()), be_scenario, moyens,
                           taux_servis.mean(axis=0), float(ppe.mean()), taux_rachats.mean(axis=0))
