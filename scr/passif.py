"""Valorisation déterministe des passifs (V1, sans générateur de scénarios).

Le BE est recalculé intégralement à chaque scénario choqué, ce qui permet de mesurer
la sensibilité des provisions aux chocs de la formule standard. Les options et garanties
financières (TVOG) et les rachats conjoncturels relèvent de la V2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scr_data.courbe import Actualisation
from scr_data.mortalite import TableMortalite


@dataclass(frozen=True)
class Chocs:
    """Chocs appliqués aux hypothèses de passif (1.0 / 0.0 = scénario central)."""
    facteur_mortalite: float = 1.0        # multiplicateur des q_x (mortalité, longévité)
    choc_mortalite_absolu: float = 0.0    # ajout ponctuel aux q_x la 1re année (CAT vie)
    facteur_rachat: float = 1.0
    rachat_massif: float = 0.0            # part des PM rachetée immédiatement
    facteur_frais: float = 1.0
    inflation_frais_sup: float = 0.0
    facteur_pm_uc: float = 1.0            # variation de la valeur des supports UC
    facteur_sinistres_non_vie: float = 1.0
    facteur_recuperation: float = 1.0     # taux de sortie d'incapacité (santé SLT)
    revision_rentes: float = 0.0          # revalorisation instantanée des arrérages


# ---------------------------------------------------------------------------
def _qx_projete(table: TableMortalite, age: int, horizon: int, chocs: Chocs) -> np.ndarray:
    q = table.qx[np.clip(np.arange(age, age + horizon), 0, len(table.qx) - 1)] * chocs.facteur_mortalite
    q = np.clip(q, 0.0, 1.0)
    if chocs.choc_mortalite_absolu:
        q = q.copy()
        q[0] = min(1.0, q[0] + chocs.choc_mortalite_absolu)
    return q


def _frais_unitaires(base: float, hyp: dict[str, Any], chocs: Chocs, horizon: int) -> np.ndarray:
    inflation = hyp["frais"]["inflation"] + chocs.inflation_frais_sup
    return base * chocs.facteur_frais * (1 + inflation) ** np.arange(horizon)


# ---------------------------------------------------------------------------
def be_epargne_euro(mp: pd.DataFrame, actu: Actualisation, courbe: pd.DataFrame,
                    tables: dict[str, TableMortalite], hyp: dict[str, Any], chocs: Chocs,
                    par_model_point: bool = False):
    """Projection PM par PM : revalorisation, rachats, décès, frais, puis actualisation."""
    H = hyp["projection"]["horizon"]
    t = np.arange(1, H + 1)
    df = actu.df(t)
    forward = np.interp(t, courbe["maturite"], courbe["forward_1an"])
    frais_pm = _frais_unitaires(hyp["frais"]["gestion_pm_euro"], hyp, chocs, H)
    alpha = hyp["epargne_euro"]["taux_participation_financiere"]
    detail = np.zeros(len(mp))
    for i, r in enumerate(mp.itertuples()):
        pm = r.pm * (1.0 - chocs.rachat_massif)
        detail[i] += r.pm * chocs.rachat_massif                   # rachat massif payé immédiatement
        q = _qx_projete(tables["homme" if r.sexe == "H" else "femme"], int(r.age), H, chocs)
        rachat = min(1.0, r.taux_rachat_structurel * chocs.facteur_rachat)
        flux = np.zeros(H)
        for k in range(H):
            brut = max(r.tmg + r.chargement_encours, alpha * forward[k])
            pm_revalorisee = pm * (1 + brut - r.chargement_encours)
            sortie = pm_revalorisee * (q[k] + rachat - q[k] * rachat)
            flux[k] = sortie + pm * frais_pm[k]
            pm = pm_revalorisee - sortie
            if pm <= 1e-9:
                break
        flux[H - 1] += pm                                          # clôture du run-off
        detail[i] += float(np.sum(flux * df))
    return detail if par_model_point else float(detail.sum())


def be_epargne_uc(mp: pd.DataFrame, actu: Actualisation, courbe: pd.DataFrame,
                  tables: dict[str, TableMortalite], hyp: dict[str, Any], chocs: Chocs,
                  par_model_point: bool = False):
    H = hyp["projection"]["horizon"]
    t = np.arange(1, H + 1)
    df = actu.df(t)
    forward = np.interp(t, courbe["maturite"], courbe["forward_1an"])
    frais_pm = _frais_unitaires(hyp["frais"]["gestion_pm_uc"], hyp, chocs, H)
    detail = np.zeros(len(mp))
    for i, r in enumerate(mp.itertuples()):
        pm = r.pm * chocs.facteur_pm_uc
        detail[i] += pm * chocs.rachat_massif
        pm *= (1.0 - chocs.rachat_massif)
        q = _qx_projete(tables["homme" if r.sexe == "H" else "femme"], int(r.age), H, chocs)
        rachat = min(1.0, r.taux_rachat_structurel * chocs.facteur_rachat)
        prelevement = r.chargement_encours + r.retrocession_encours
        flux = np.zeros(H)
        for k in range(H):
            pm_revalorisee = pm * (1 + forward[k] - prelevement)
            sortie = pm_revalorisee * (q[k] + rachat - q[k] * rachat)
            flux[k] = sortie + pm * frais_pm[k]
            pm = pm_revalorisee - sortie
            if pm <= 1e-9:
                break
        flux[H - 1] += pm
        detail[i] += float(np.sum(flux * df))
    return detail if par_model_point else float(detail.sum())


def be_temporaire_deces(mp: pd.DataFrame, actu: Actualisation, tables, hyp, chocs: Chocs) -> float:
    """BE = valeur actuelle des capitaux décès et des frais, nette des primes futures."""
    total = 0.0
    taux_frais = hyp["frais"]["gestion_temporaire_deces"] * chocs.facteur_frais
    for r in mp.itertuples():
        n = int(r.duree_residuelle)
        if n <= 0:
            continue
        q = _qx_projete(tables["homme" if r.sexe == "H" else "femme"], int(r.age), n, chocs)
        chute = r.taux_chute * chocs.facteur_rachat
        maintien = np.concatenate([[1.0], np.cumprod((1 - q) * (1 - chute))])[:n]
        df_mil = actu.df(np.arange(1, n + 1) - 0.5)
        df_debut = actu.df(np.arange(0, n))
        prestations = float(np.sum(maintien * q * r.capitaux_sous_risque * df_mil))
        primes = float(np.sum(maintien * r.prime_annuelle * df_debut))
        total += prestations + taux_frais * primes - primes
    return total


def be_rentes(mp: pd.DataFrame, actu: Actualisation, tables, hyp, chocs: Chocs) -> float:
    from scr_data.passif_vie import _annuite_reversion
    total = 0.0
    for r in mp.itertuples():
        tp = tables["rente_homme" if r.sexe == "H" else "rente_femme"].ajuster(chocs.facteur_mortalite)
        tc = tables["rente_femme" if r.sexe == "H" else "rente_homme"].ajuster(chocs.facteur_mortalite)
        a = tp.annuite_viagere(int(r.age), actu, terme_echu=False)
        if bool(r.reversion):
            a += r.taux_reversion * _annuite_reversion(tp, tc, int(r.age), int(r.age_conjoint), actu)
        total += (r.arrerages_annuels * (1 + chocs.revision_rentes) * a
                  * (1 + r.frais_gestion * chocs.facteur_frais))
    return total


def be_sante_slt(mp: pd.DataFrame, actu: Actualisation, tables, hyp, chocs: Chocs) -> float:
    h = hyp["sante_slt"]
    lam = (1.0 / h["duree_moyenne_incapacite"]) * chocs.facteur_recuperation
    total = 0.0
    for r in mp.itertuples():
        if r.garantie == "invalidite":
            duree = int(r.age_fin_garantie - r.age)
            if duree <= 0:
                continue
            table = tables["homme" if r.sexe == "H" else "femme"]
            q = np.clip(table.qx[int(r.age):int(r.age) + duree] * h["surmortalite_invalides"]
                        * chocs.facteur_mortalite, 0, 1)
            p = np.cumprod(1 - q)
            facteur = float(np.sum(p * actu.df(np.arange(1, duree + 1) - 0.5)))
        else:
            reste = max(h["duree_max_incapacite"] - r.anciennete_sinistre, 0.1)
            pas = np.arange(0.5, np.ceil(reste * 12)) / 12.0
            facteur = float(np.sum(np.exp(-lam * pas) * actu.df(pas)) / 12.0)
        frais = hyp["frais"]["gestion_sante_slt"] * chocs.facteur_frais
        total += r.rente_annuelle * (1 + chocs.revision_rentes) * facteur * (1 + frais)
    return total


# ---------------------------------------------------------------------------
# Non-vie : Chain Ladder puis actualisation
# ---------------------------------------------------------------------------
def chain_ladder(triangle: pd.DataFrame) -> dict[str, Any]:
    """Chain Ladder standard sur un triangle de règlements cumulés (index = survenance)."""
    tri = triangle.to_numpy(dtype=float)
    n = tri.shape[0]
    facteurs = []
    for j in range(n - 1):
        num = np.nansum(tri[: n - j - 1, j + 1])
        den = np.nansum(tri[: n - j - 1, j])
        facteurs.append(num / den if den else 1.0)
    facteurs = np.array(facteurs)
    complet = tri.copy()
    for i in range(n):
        for j in range(n - i, n):
            complet[i, j] = complet[i, j - 1] * facteurs[j - 1]
    ultimes = complet[:, -1]
    derniers = np.array([tri[i, n - 1 - i] for i in range(n)])
    return {"facteurs": facteurs, "ultimes": ultimes, "reserves": ultimes - derniers,
            "triangle_complete": complet}


def cadence_reglement(facteurs: np.ndarray) -> np.ndarray:
    """Part cumulée payée à chaque développement, déduite des facteurs de développement."""
    cumul = np.concatenate([[1.0], np.cumprod(facteurs)])
    return cumul / cumul[-1]


def be_non_vie(donnees: dict[str, pd.DataFrame], actu: Actualisation, hyp: dict[str, Any],
               chocs: Chocs, modules: tuple[str, ...] = ("non_vie", "sante_nslt")) -> dict[str, float]:
    """BE sinistres (Chain Ladder actualisé) et BE primes, par module."""
    tri = donnees["triangles_reglements"]
    volumes = donnees["volumes_primes_reserves"].set_index("lob")
    resultats = {"be_sinistres": {}, "be_primes": {}, "reserves_non_actualisees": {}}
    for lob, sous in tri.groupby("lob"):
        module = volumes.loc[lob, "module"]
        if module not in modules:
            continue
        matrice = sous.pivot(index="annee_survenance", columns="developpement", values="reglements_cumules")
        cl = chain_ladder(matrice)
        parts = cadence_reglement(cl["facteurs"])
        n = len(parts)
        reserves = cl["reserves"] * chocs.facteur_sinistres_non_vie
        be_sin = 0.0
        for i, reserve in enumerate(reserves):
            dev = n - 1 - i                                   # dernier développement observé
            restant = 1.0 - parts[dev]
            if restant <= 1e-10:
                be_sin += reserve
                continue
            incr = np.diff(np.concatenate([[parts[dev]], parts[dev + 1:], [1.0]])) / restant
            dates = np.arange(1, len(incr) + 1) - 0.5
            be_sin += float(reserve * np.sum(incr * actu.df(dates)))
        frais = hyp["frais"]["gestion_sinistres_non_vie"] * chocs.facteur_frais
        resultats["be_sinistres"][lob] = be_sin * (1 + frais)
        resultats["reserves_non_actualisees"][lob] = float(reserves.sum())

        exposition = volumes.loc[lob, "primes_acquises_n_plus_1_estimees"] * hyp["projection"]["part_primes_non_acquises"]
        primes_historiques = donnees["ultimes_vrais"].query("lob == @lob")["primes_acquises"].sum()
        sp = float(cl["ultimes"].sum() / primes_historiques)   # S/P ultime estimé par Chain Ladder
        ratio_combine = sp * chocs.facteur_sinistres_non_vie + hyp["frais"]["ratio_frais_non_vie"] * chocs.facteur_frais
        resultats["be_primes"][lob] = float(exposition * ratio_combine * actu.df(1.0))
    return resultats


# ---------------------------------------------------------------------------
@dataclass
class Passifs:
    donnees: dict[str, pd.DataFrame]
    hyp: dict[str, Any]
    tables: dict[str, TableMortalite]

    def evaluer(self, actu: Actualisation, courbe: pd.DataFrame, chocs: Chocs | None = None
                ) -> dict[str, float]:
        c = chocs or Chocs()
        d = self.donnees
        nv = be_non_vie(d, actu, self.hyp, c)
        be = {
            "epargne_euro": be_epargne_euro(d["mp_epargne_euro"], actu, courbe, self.tables, self.hyp, c),
            "epargne_uc": be_epargne_uc(d["mp_epargne_uc"], actu, courbe, self.tables, self.hyp, c),
            "temporaire_deces": be_temporaire_deces(d["mp_temporaire_deces"], actu, self.tables, self.hyp, c),
            "rentes": be_rentes(d["mp_rentes_viageres"], actu, self.tables, self.hyp, c),
            "sante_slt": be_sante_slt(d["mp_sante_slt"], actu, self.tables, self.hyp, c),
            "non_vie_sinistres": sum(v for k, v in nv["be_sinistres"].items() if k != "sante_frais_soins"),
            "non_vie_primes": sum(v for k, v in nv["be_primes"].items() if k != "sante_frais_soins"),
            "sante_nslt": nv["be_sinistres"].get("sante_frais_soins", 0.0)
                          + nv["be_primes"].get("sante_frais_soins", 0.0),
        }
        be["total"] = sum(be.values())
        be["_detail_non_vie"] = nv
        return be
