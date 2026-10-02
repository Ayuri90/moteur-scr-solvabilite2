"""Modules de souscription vie (Art. 136-144) et santé (Art. 144-163).

Chaque choc est appliqué en reprojetant intégralement les best estimates concernés :
le SCR d'un sous-module est la hausse des provisions techniques qu'il provoque,
à actif inchangé (les chocs de souscription ne touchent pas la valeur des placements).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from scr.passif import (Chocs, be_epargne_euro, be_epargne_uc, be_rentes, be_sante_slt,
                        be_temporaire_deces, be_non_vie)
from scr_data.courbe import Actualisation


def agreger(scr: dict[str, float], correlations: dict[str, Any]) -> float:
    v = np.array([scr[nom] for nom in correlations["ordre"]])
    M = np.array(correlations["matrice"], dtype=float)
    return float(np.sqrt(max(v @ M @ v, 0.0)))


# ---------------------------------------------------------------------------
class MoteurVie:
    """SCR de souscription vie : épargne euro, UC, temporaire décès, rentes."""

    def __init__(self, donnees, hyp, params, tables, actu: Actualisation, courbe: pd.DataFrame):
        self.d, self.hyp, self.p, self.tables = donnees, hyp, params["vie"], tables
        self.actu, self.courbe = actu, courbe
        self.base = self._evaluer(Chocs())
        self.base_mp = self._evaluer_mp(Chocs())

    # -- évaluations -------------------------------------------------------
    def _evaluer(self, chocs: Chocs) -> dict[str, float]:
        return {
            "epargne_euro": be_epargne_euro(self.d["mp_epargne_euro"], self.actu, self.courbe,
                                            self.tables, self.hyp, chocs),
            "epargne_uc": be_epargne_uc(self.d["mp_epargne_uc"], self.actu, self.courbe,
                                        self.tables, self.hyp, chocs),
            "temporaire_deces": be_temporaire_deces(self.d["mp_temporaire_deces"], self.actu,
                                                    self.tables, self.hyp, chocs),
            "rentes": be_rentes(self.d["mp_rentes_viageres"], self.actu, self.tables, self.hyp, chocs),
        }

    def _evaluer_mp(self, chocs: Chocs) -> dict[str, np.ndarray]:
        return {
            "epargne_euro": be_epargne_euro(self.d["mp_epargne_euro"], self.actu, self.courbe,
                                            self.tables, self.hyp, chocs, par_model_point=True),
            "epargne_uc": be_epargne_uc(self.d["mp_epargne_uc"], self.actu, self.courbe,
                                        self.tables, self.hyp, chocs, par_model_point=True),
        }

    def _perte(self, chocs: Chocs, segments: tuple[str, ...] | None = None) -> float:
        """Hausse du BE, retenue segment par segment lorsqu'elle est défavorable."""
        choque = self._evaluer(chocs)
        segments = segments or tuple(self.base)
        return float(sum(max(0.0, choque[s] - self.base[s]) for s in segments))

    # -- sous-modules ------------------------------------------------------
    def mortalite(self) -> float:
        return self._perte(Chocs(facteur_mortalite=1 + self.p["mortalite"]))

    def longevite(self) -> float:
        return self._perte(Chocs(facteur_mortalite=1 - self.p["longevite"]))

    def rachat(self) -> tuple[float, str]:
        hausse = self._perte(Chocs(facteur_rachat=1 + self.p["rachat_hausse"]))
        baisse = self._perte(Chocs(facteur_rachat=1 - self.p["rachat_baisse"]))
        massif = self._rachat_massif()
        pire = max(hausse, baisse, massif)
        return pire, f"hausse {hausse:.1f} / baisse {baisse:.1f} / massif {massif:.1f}"

    def _rachat_massif(self) -> float:
        """Le choc ne s'applique qu'aux contrats dont le rachat génère une perte (Art. 142)."""
        chocs = Chocs(rachat_massif=self.p["rachat_massif"])
        choque = self._evaluer_mp(chocs)
        perte = 0.0
        for segment, be_choc in choque.items():
            perte += float(np.sum(np.maximum(be_choc - self.base_mp[segment], 0.0)))
        return perte

    def frais(self) -> float:
        return self._perte(Chocs(facteur_frais=1 + self.p["frais_hausse"],
                                 inflation_frais_sup=self.p["frais_inflation_sup"]))

    def revision(self) -> float:
        if not self.p["revision"]:
            return 0.0
        return self._perte(Chocs(revision_rentes=self.p["revision"]), segments=("rentes",))

    def catastrophe(self) -> float:
        return self._perte(Chocs(choc_mortalite_absolu=self.p["cat_mortalite_absolu"]))

    # -- agrégation --------------------------------------------------------
    def calculer(self) -> tuple[pd.DataFrame, float]:
        rachat, detail_rachat = self.rachat()
        scr = {"mortalite": self.mortalite(), "longevite": self.longevite(), "invalidite": 0.0,
               "rachat": rachat, "frais": self.frais(), "revision": self.revision(),
               "catastrophe": self.catastrophe()}
        total = agreger(scr, self.p["correlations"])
        details = {"rachat": detail_rachat, "invalidite": "sans objet (pas de garantie incapacité en vie)",
                   "revision": "sans objet (aucune rente révisable)"}
        table = pd.DataFrame([{"sous_module": k, "scr": v, "detail": details.get(k, "")}
                              for k, v in scr.items()])
        table.loc[len(table)] = {"sous_module": "SCR souscription vie (agrégé)", "scr": total,
                                 "detail": f"diversification {1 - total / sum(scr.values()):.0%}"}
        return table, total


# ---------------------------------------------------------------------------
class MoteurSante:
    """SCR de souscription santé : SLT (rentes en service), NSLT (frais de soins), catastrophe."""

    def __init__(self, donnees, hyp, params, tables, actu: Actualisation, courbe: pd.DataFrame):
        self.d, self.hyp, self.p, self.tables = donnees, hyp, params["sante"], tables
        self.actu, self.courbe = actu, courbe
        self.base_slt = be_sante_slt(self.d["mp_sante_slt"], actu, tables, hyp, Chocs())
        self.detail_nv = be_non_vie(donnees, actu, hyp, Chocs(), modules=("sante_nslt",))

    # -- santé SLT ---------------------------------------------------------
    def _perte_slt(self, chocs: Chocs) -> float:
        be = be_sante_slt(self.d["mp_sante_slt"], self.actu, self.tables, self.hyp, chocs)
        return max(0.0, be - self.base_slt)

    def slt(self) -> tuple[pd.DataFrame, float]:
        p = self.p["slt"]
        scr = {
            "mortalite": self._perte_slt(Chocs(facteur_mortalite=1 + p["mortalite"])),
            "longevite": self._perte_slt(Chocs(facteur_mortalite=1 - p["longevite"])),
            "invalidite": self._perte_slt(Chocs(facteur_recuperation=p["recuperation_invalidite"])),
            "rachat": 0.0,
            "frais": self._perte_slt(Chocs(facteur_frais=1 + p["frais_hausse"],
                                           inflation_frais_sup=p["frais_inflation_sup"])),
            "revision": self._perte_slt(Chocs(revision_rentes=p["revision"])),
        }
        total = agreger(scr, p["correlations"])
        table = pd.DataFrame([{"sous_module": f"SLT · {k}", "scr": v} for k, v in scr.items()])
        table.loc[len(table)] = {"sous_module": "SLT (agrégé)", "scr": total}
        return table, total

    # -- santé NSLT --------------------------------------------------------
    def nslt(self) -> tuple[pd.DataFrame, float]:
        p = self.p["nslt"]
        volumes = self.d["volumes_primes_reserves"].set_index("lob")
        lignes, variances = [], []
        for lob, sigma in p["sigma"].items():
            v_primes = max(volumes.loc[lob, "primes_acquises_n_plus_1_estimees"],
                           volumes.loc[lob, "primes_acquises_n"])
            v_reserves = self.detail_nv["be_sinistres"][lob]
            v = v_primes + v_reserves
            sp, sr = sigma["primes"], sigma["reserves"]
            sigma_lob = np.sqrt((sp * v_primes) ** 2 + sp * v_primes * sr * v_reserves
                                + (sr * v_reserves) ** 2) / v
            lignes.append({"sous_module": f"NSLT · {lob}", "scr": p["facteur_ecart_type"] * sigma_lob * v})
            variances.append(p["facteur_ecart_type"] * sigma_lob * v)
        primes_reserves = float(np.sum(variances))     # une seule ligne d'activité ici
        rachat = 0.0                                   # cessation non pénalisante : ratio combiné > 1
        rho = p["correlation_primes_rachat"]
        total = float(np.sqrt(primes_reserves ** 2 + 2 * rho * primes_reserves * rachat + rachat ** 2))
        table = pd.DataFrame(lignes + [{"sous_module": "NSLT · rachat", "scr": rachat},
                                       {"sous_module": "NSLT (agrégé)", "scr": total}])
        return table, total

    # -- santé catastrophe (version simplifiée) ----------------------------
    def catastrophe(self) -> tuple[pd.DataFrame, float]:
        p = self.p["cat"]
        expo = self.d["exposition_cat"].set_index("type")["montant"]
        nb_assures = float(expo.get("nb_assures_sante", 0.0))
        masse = nb_assures * p["accident_masse"]["part_assures_touches"] * p["accident_masse"]["cout_moyen_par_sinistre"]
        conc = p["concentration"]["nb_personnes_plus_grand_groupe"] * p["concentration"]["cout_moyen_par_sinistre"]
        pandemie = nb_assures * p["pandemie"]["taux_recours_soins"] * p["pandemie"]["cout_moyen_par_sinistre"]
        total = float(np.sqrt(masse ** 2 + conc ** 2 + pandemie ** 2))
        table = pd.DataFrame([{"sous_module": "CAT · accident de masse", "scr": masse},
                              {"sous_module": "CAT · concentration d'accident", "scr": conc},
                              {"sous_module": "CAT · pandémie", "scr": pandemie},
                              {"sous_module": "CAT (agrégé)", "scr": total}])
        return table, total

    # -- agrégation --------------------------------------------------------
    def calculer(self) -> tuple[pd.DataFrame, float]:
        t_slt, slt = self.slt()
        t_nslt, nslt = self.nslt()
        t_cat, cat = self.catastrophe()
        total = agreger({"slt": slt, "nslt": nslt, "catastrophe": cat}, self.p["correlations"])
        table = pd.concat([t_slt, t_nslt, t_cat,
                           pd.DataFrame([{"sous_module": "SCR souscription santé (agrégé)", "scr": total}])],
                          ignore_index=True)
        return table, total
