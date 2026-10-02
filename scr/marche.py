"""Module de risque de marché de la formule standard (Art. 164 à 188).

Chaque sous-module est évalué en recalculant intégralement les fonds propres de base :
les actifs sont revalorisés ligne par ligne et les provisions techniques sont reprojetées
lorsque le choc les affecte (taux, valeur des supports UC).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from functools import lru_cache

from scr_data.actifs import prix_obligation
from scr_data.config import RACINE
from scr_data.courbe import Actualisation
from scr.passif import Chocs, Passifs, be_epargne_uc

CLASSES_TAUX = ("obligation", "monetaire", "pret_infrastructure", "depot")


# ---------------------------------------------------------------------------
# Courbes choquées
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _facteurs_officiels(chemin: str) -> pd.DataFrame:
    """Facteurs de choc de taux extraits du fichier EIOPA (onglet Shocks)."""
    return pd.read_csv(RACINE / chemin).set_index("maturite")


def facteurs_choc_taux(params: dict[str, Any], sens: str, maturites: np.ndarray) -> np.ndarray:
    """Facteurs relatifs par maturité : fichier officiel s'il existe, sinon tables du YAML."""
    p = params["taux"]
    chemin = p.get("fichier_facteurs")
    if chemin and (RACINE / chemin).exists():
        table = _facteurs_officiels(chemin)
        return np.interp(maturites, table.index.to_numpy(float), table[sens].to_numpy(float),
                         left=float(table[sens].iloc[0]), right=float(table[sens].iloc[-1]))
    bruts = p["facteurs_hausse"] if sens == "hausse" else p["facteurs_baisse"]
    points = np.array(sorted(bruts), dtype=float)
    valeurs = np.array([bruts[int(m)] for m in points])
    return np.interp(maturites, points, valeurs, left=valeurs[0], right=valeurs[-1])


def courbe_choquee(courbe: pd.DataFrame, params: dict[str, Any], sens: str) -> pd.DataFrame:
    p = params["taux"]
    if p.get("methode") == "2027":
        from scr.reforme import courbe_choquee_2027
        return courbe_choquee_2027(courbe, p["reforme"], sens, p["ufr_base"])
    t = courbe["maturite"].to_numpy(float)
    f = facteurs_choc_taux(params, sens, t)
    spot = courbe["taux_spot"].to_numpy(float)
    if sens == "hausse":
        choque = np.maximum(spot * (1 + f), spot + p["plancher_hausse_absolu"])
    else:
        choque = spot * (1 - f)
        if p["neutralisation_taux_negatifs"]:
            choque = np.where(spot < 0, spot, choque)
    df = (1 + choque) ** (-t)
    forward = np.concatenate([[1.0], df[:-1]]) / df - 1.0
    return pd.DataFrame({"maturite": courbe["maturite"], "taux_spot": choque,
                         "facteur_actualisation": df, "forward_1an": forward})


# ---------------------------------------------------------------------------
# Revalorisation des actifs
# ---------------------------------------------------------------------------
def revaloriser_obligations(inv: pd.DataFrame, actu: Actualisation) -> pd.Series:
    """Valeur de marché des titres à revenu fixe sous une courbe donnée."""
    vm = inv["valeur_marche"].astype(float).copy()
    masque = inv["classe"].isin(CLASSES_TAUX) & inv["maturite"].notna() & inv["coupon"].notna()
    for idx in inv.index[masque]:
        r = inv.loc[idx]
        prix = prix_obligation(actu, float(r["coupon"]), int(r["maturite"]), float(r["spread_bp"]) / 1e4)
        vm.loc[idx] = float(r["nominal"]) * prix / 100.0
    return vm


def facteur_stress_spread(cqs, duration: float, params: dict[str, Any], covered: bool) -> float:
    p = params["spread"]
    d = max(duration, p["duration_plancher"])
    if covered and cqs is not None and int(cqs) in (0, 1) and d <= 5:
        return min(p["covered_bonds"][int(cqs)] * d, 1.0)
    if cqs is None or pd.isna(cqs):
        bandes, a, b = p["non_note"]["bandes"], p["non_note"]["a"], p["non_note"]["b"]
    else:
        table = p["notes"][int(cqs)]
        bandes, a, b = p["bandes"], table["a"], table["b"]
    i = int(np.searchsorted(bandes, d, side="left"))
    i = min(i, len(a) - 1)
    borne = 0.0 if i == 0 else bandes[i - 1]
    return float(min(a[i] + b[i] * (d - borne), 1.0))


# ---------------------------------------------------------------------------
def matrice_marche(matrice, a: float, b: float) -> np.ndarray:
    """Matrice de corrélation du module marché : A pour le taux, B pour le couple taux / spread."""
    substitution = {"A": a, "B": b}
    return np.array([[substitution.get(c, c) if isinstance(c, str) else float(c) for c in ligne]
                     for ligne in matrice], dtype=float)


@dataclass
class ResultatSousModule:
    nom: str
    scr: float
    detail: str = ""


class MoteurMarche:
    def __init__(self, inventaire: pd.DataFrame, passifs: Passifs, courbe: pd.DataFrame,
                 hyp: dict[str, Any], params: dict[str, Any], autres_passifs: float):
        self.inv = inventaire.reset_index(drop=True)
        self.passifs = passifs
        self.courbe = courbe
        self.actu = Actualisation(courbe)
        self.hyp = hyp
        self.params = params
        self.autres_passifs = autres_passifs
        self.be_base = passifs.evaluer(self.actu, courbe)
        self.actif_base = float(self.inv["valeur_marche"].sum()) + self._actifs_hors_inventaire()
        self.bof_base = self.actif_base - self.be_base["total"] - autres_passifs

    def _actifs_hors_inventaire(self) -> float:
        a = self.hyp["bilan_cible"]["actif"]
        return a["provisions_cedees"] + a["creances_tresorerie_autres"]

    # -- outil générique ---------------------------------------------------
    def _bof(self, valeurs: pd.Series, courbe: pd.DataFrame | None = None,
             chocs: Chocs | None = None) -> float:
        courbe_base = courbe is None or courbe is self.courbe
        courbe = self.courbe if courbe is None else courbe
        if courbe_base and (chocs is None or chocs == Chocs(facteur_pm_uc=chocs.facteur_pm_uc)):
            # raccourci : seuls les supports UC bougent, le reste du passif est inchangé
            be_total = self.be_base["total"] - self.be_base["epargne_uc"]
            be_total += be_epargne_uc(self.passifs.donnees["mp_epargne_uc"], self.actu, self.courbe,
                                      self.passifs.tables, self.hyp, chocs or Chocs())
        else:
            actu = Actualisation(courbe)
            be_total = self.passifs.evaluer(actu, courbe, chocs)["total"]
        return float(valeurs.sum()) + self._actifs_hors_inventaire() - be_total - self.autres_passifs

    def _perte(self, valeurs: pd.Series, courbe=None, chocs=None) -> float:
        return max(0.0, self.bof_base - self._bof(valeurs, courbe, chocs))

    def _facteur_uc(self, valeurs: pd.Series) -> float:
        """Variation relative de la valeur des supports UC induite par le choc."""
        uc = self.inv["portefeuille"] == "uc"
        base = float(self.inv.loc[uc, "valeur_marche"].sum())
        if base == 0:
            return 1.0
        return float(valeurs[uc].sum() / base)

    # -- sous-modules ------------------------------------------------------
    def taux(self) -> tuple[ResultatSousModule, str]:
        resultats = {}
        for sens in ("hausse", "baisse"):
            c = courbe_choquee(self.courbe, self.params, sens)
            valeurs = revaloriser_obligations(self.inv, Actualisation(c))
            facteur_uc = self._facteur_uc(valeurs)
            resultats[sens] = self._perte(valeurs, c, Chocs(facteur_pm_uc=facteur_uc))
        self.pertes_taux = resultats
        sens_retenu = max(resultats, key=resultats.get)
        detail = f"hausse {resultats['hausse']:.1f} / baisse {resultats['baisse']:.1f}"
        return ResultatSousModule("taux", resultats[sens_retenu], detail), sens_retenu

    def action(self) -> ResultatSousModule:
        p = self.params["action"]
        sa = p["ajustement_symetrique"]
        chocs = {1: p["choc_type1"] + sa, 2: p["choc_type2"] + sa, 3: p["choc_strategique"]}
        pertes = {}
        for type_action, choc in chocs.items():
            masque = self.inv["type_action"] == type_action
            valeurs = self.inv["valeur_marche"].copy()
            valeurs[masque] *= (1 - choc)
            pertes[type_action] = self._perte(valeurs, chocs=Chocs(facteur_pm_uc=self._facteur_uc(valeurs)))
        # agrégation type 1 / type 2 (les participations stratégiques suivent le type 1)
        t1 = pertes[1] + pertes[3]
        t2 = pertes[2]
        rho = p["correlation_type1_type2"]
        scr = float(np.sqrt(t1**2 + 2 * rho * t1 * t2 + t2**2))
        return ResultatSousModule("action", scr, f"type 1 + strat. {t1:.1f} / type 2 {t2:.1f}")

    def immobilier(self) -> ResultatSousModule:
        choc = self.params["immobilier"]["choc"]
        masque = self.inv["classe"] == "immobilier"
        valeurs = self.inv["valeur_marche"].copy()
        valeurs[masque] *= (1 - choc)
        perte = self._perte(valeurs, chocs=Chocs(facteur_pm_uc=self._facteur_uc(valeurs)))
        return ResultatSousModule("immobilier", perte, f"assiette {self.inv.loc[masque, 'valeur_marche'].sum():.0f} M€")

    def spread(self) -> ResultatSousModule:
        p = self.params["spread"]
        exempts = set(p["exposition_exemptee"]["codes_pays_eee"]) if p["exposition_exemptee"]["souverains_eee"] else set()
        valeurs = self.inv["valeur_marche"].copy()
        assiette = 0.0
        for idx, r in self.inv.iterrows():
            if r["classe"] not in ("obligation", "pret_infrastructure", "monetaire"):
                continue
            if r["sous_classe"] == "souverain" and r["pays"] in exempts:
                continue
            duration = r["duration_modifiee"]
            if pd.isna(duration):
                continue
            stress = facteur_stress_spread(r["cqs"], float(duration), self.params, bool(r["covered_bond"]))
            if r["infra_eligible"]:
                stress *= p["infrastructure_reduction"]
            valeurs[idx] = r["valeur_marche"] * (1 - stress)
            assiette += r["valeur_marche"]
        perte = self._perte(valeurs, chocs=Chocs(facteur_pm_uc=self._facteur_uc(valeurs)))
        return ResultatSousModule("spread", perte, f"assiette {assiette:.0f} M€")

    def change(self) -> ResultatSousModule:
        p = self.params["change"]
        pertes = []
        for devise in sorted(set(self.inv.loc[self.inv["devise"] != "EUR", "devise"])):
            masque = self.inv["devise"] == devise
            pire = 0.0
            for sens, choc in (("hausse", p["choc_hausse"]), ("baisse", -p["choc_baisse"])):
                valeurs = self.inv["valeur_marche"].copy()
                valeurs[masque] *= (1 - choc)
                pire = max(pire, self._perte(valeurs, chocs=Chocs(facteur_pm_uc=self._facteur_uc(valeurs))))
            pertes.append(pire)
        scr = float(np.sum(pertes))       # agrégation par simple somme entre devises
        return ResultatSousModule("change", scr, f"{len(pertes)} devises")

    def concentration(self) -> ResultatSousModule:
        p = self.params["concentration"]
        base = self.inv[self.inv["portefeuille"] == "euro"].copy()
        if p["exempter_souverains_eee"]:
            base = base[~(base["sous_classe"] == "souverain")]
        actifs = float(self.inv.loc[self.inv["portefeuille"] == "euro", "valeur_marche"].sum())
        expo = base.groupby("groupe_emetteur").agg(vm=("valeur_marche", "sum"), cqs=("cqs", "min"))
        carres = 0.0
        pires = []
        for groupe, r in expo.iterrows():
            cqs = None if pd.isna(r["cqs"]) else int(r["cqs"])
            seuil = p["seuil_non_note"] if cqs is None else p["seuil_par_cqs"][cqs]
            g = p["facteur_non_note"] if cqs is None else p["facteur_par_cqs"][cqs]
            xs = max(0.0, r["vm"] / actifs - seuil)
            conc = actifs * xs * g
            carres += conc**2
            if conc > 0:
                pires.append((groupe, conc))
        pires.sort(key=lambda x: -x[1])
        detail = ", ".join(f"{g} {c:.1f}" for g, c in pires[:3]) or "aucun dépassement"
        return ResultatSousModule("concentration", float(np.sqrt(carres)), detail)

    # -- agrégation --------------------------------------------------------
    def calculer(self) -> tuple[pd.DataFrame, float]:
        res_taux, sens = self.taux()
        sous_modules = [res_taux, self.action(), self.immobilier(), self.spread(),
                        self.change(), self.concentration()]
        ordre = self.params["correlations_marche"]["ordre"]
        v = np.array([next(s.scr for s in sous_modules if s.nom == nom) for nom in ordre])
        cm = self.params["correlations_marche"]
        a = cm["a_hausse" if sens == "hausse" else "a_baisse"]
        b = cm.get("b_hausse" if sens == "hausse" else "b_baisse", a)
        M = matrice_marche(cm["matrice"], a, b)
        scr = float(np.sqrt(v @ M @ v))
        table = pd.DataFrame({"sous_module": [s.nom for s in sous_modules],
                              "scr": [s.scr for s in sous_modules],
                              "detail": [s.detail for s in sous_modules]})
        table.loc[table["sous_module"] == "taux", "detail"] += f" — scénario retenu : {sens}"
        table = pd.concat([table, pd.DataFrame([{"sous_module": "SCR marché (agrégé)", "scr": scr,
                                                 "detail": f"A = {a}"}])], ignore_index=True)
        return table, scr
