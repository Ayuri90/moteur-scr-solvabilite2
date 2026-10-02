"""Module de souscription non-vie (Art. 114 à 135) : primes et réserves, rachat, catastrophe."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scr.passif import Chocs, be_non_vie
from scr.souscription import agreger
from scr_data.courbe import Actualisation


class MoteurNonVie:
    def __init__(self, donnees, hyp, params, actu: Actualisation):
        self.d, self.hyp, self.p = donnees, hyp, params["non_vie"]
        self.actu = actu
        self.detail_be = be_non_vie(donnees, actu, hyp, Chocs(), modules=("non_vie",))
        self.volumes = donnees["volumes_primes_reserves"].set_index("lob")

    # -- primes et réserves ------------------------------------------------
    def primes_reserves(self) -> tuple[pd.DataFrame, float]:
        lignes = []
        for lob, sigma in self.p["sigma"].items():
            v_primes = max(self.volumes.loc[lob, "primes_acquises_n_plus_1_estimees"],
                           self.volumes.loc[lob, "primes_acquises_n"]) \
                + self.volumes.loc[lob, "fp_existant"] + self.volumes.loc[lob, "fp_futur"]
            v_reserves = self.detail_be["be_sinistres"][lob]
            v = (v_primes + v_reserves) * self.p["facteur_diversification_geographique"]
            sp, sr = sigma["primes"], sigma["reserves"]
            sigma_lob = float(np.sqrt((sp * v_primes) ** 2 + sp * v_primes * sr * v_reserves
                                      + (sr * v_reserves) ** 2) / (v_primes + v_reserves))
            lignes.append({"lob": lob, "v_primes": v_primes, "v_reserves": v_reserves,
                           "volume": v, "sigma": sigma_lob})
        table = pd.DataFrame(lignes).set_index("lob").loc[self.p["correlations_lob"]["ordre"]]
        sv = (table["sigma"] * table["volume"]).to_numpy()
        M = np.array(self.p["correlations_lob"]["matrice"], dtype=float)
        v_total = float(table["volume"].sum())
        sigma_global = float(np.sqrt(sv @ M @ sv) / v_total)
        scr = self.p["facteur_ecart_type"] * sigma_global * v_total
        table["scr_isole"] = self.p["facteur_ecart_type"] * table["sigma"] * table["volume"]
        return table.reset_index(), float(scr)

    # -- rachat ------------------------------------------------------------
    def rachat(self) -> tuple[float, str]:
        """40 % de cessations : la perte n'existe que si la provision pour primes est bénéficiaire."""
        benefice = 0.0
        for lob, be_primes in self.detail_be["be_primes"].items():
            exposition = (self.volumes.loc[lob, "primes_acquises_n_plus_1_estimees"]
                          * self.hyp["projection"]["part_primes_non_acquises"])
            benefice += max(0.0, exposition - be_primes)      # marge attendue sur primes non acquises
        return self.p["rachat"] * benefice, f"marge sur primes non acquises {benefice:.1f} M€"

    # -- catastrophe -------------------------------------------------------
    def catastrophe(self) -> tuple[pd.DataFrame, float]:
        p = self.p["catastrophe"]
        expo = self.d["exposition_cat"]
        capitaux = expo.loc[expo["type"] == "capitaux_assures_mrh", "montant"].sum()
        lignes = []
        for peril, par in p["perils_naturels"].items():
            lignes.append({"composante": f"nat · {peril}", "scr_brut": capitaux * par["facteur_capitaux"]})
        nat = float(np.sqrt(sum(l["scr_brut"] ** 2 for l in lignes)))

        vehicules = float(expo.loc[expo["type"] == "nb_vehicules_assures", "montant"].iloc[0])
        auto = max(vehicules * p["auto"]["facteur_par_vehicule"], p["auto"]["plancher"])
        incendie = float(expo.loc[expo["type"] == "concentration_incendie_200m", "montant"].max())
        rc = p["responsabilite"]["facteur_primes"] * self.volumes.loc["rc_generale", "primes_acquises_n"]
        homme = float(np.sqrt(auto ** 2 + incendie ** 2 + rc ** 2))

        lignes += [{"composante": "nat (agrégé)", "scr_brut": nat},
                   {"composante": "homme · auto", "scr_brut": auto},
                   {"composante": "homme · incendie", "scr_brut": incendie},
                   {"composante": "homme · responsabilité", "scr_brut": rc},
                   {"composante": "homme (agrégé)", "scr_brut": homme}]
        brut = float(np.sqrt(nat ** 2 + homme ** 2))
        net = self._appliquer_xl(brut) if p["appliquer_xl_cat"] else brut
        lignes += [{"composante": "CAT brut de réassurance", "scr_brut": brut},
                   {"composante": "CAT net de réassurance", "scr_brut": net}]
        return pd.DataFrame(lignes), net

    def _appliquer_xl(self, brut: float) -> float:
        r = self.hyp["non_vie"]["reassurance"]
        priorite, portee = r["xl_cat_priorite"], r["xl_cat_portee"]
        return float(min(brut, priorite) + max(0.0, brut - priorite - portee))

    # -- agrégation --------------------------------------------------------
    def calculer(self) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        table_pr, pr = self.primes_reserves()
        rachat, detail_rachat = self.rachat()
        table_cat, cat = self.catastrophe()
        total = agreger({"primes_reserves": pr, "rachat": rachat, "catastrophe": cat},
                        self.p["correlations"])
        table = pd.DataFrame([
            {"sous_module": "primes et réserves", "scr": pr,
             "detail": f"volume {table_pr['volume'].sum():.0f} M€"},
            {"sous_module": "rachat", "scr": rachat, "detail": detail_rachat},
            {"sous_module": "catastrophe", "scr": cat, "detail": "net de l'XL CAT"},
            {"sous_module": "SCR souscription non-vie (agrégé)", "scr": total, "detail": ""},
        ])
        return table, pd.concat([table_pr.assign(bloc="primes_reserves"),
                                 table_cat.assign(bloc="catastrophe")], ignore_index=True), total
