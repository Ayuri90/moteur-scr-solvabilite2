"""Générateur de scénarios économiques risque-neutre (V2).

- Taux : Hull-White à un facteur, calibré pour reproduire exactement la courbe initiale.
- Action et immobilier : mouvements browniens géométriques de dérive r_t, corrélés au taux court.
- Déflateurs : exp(-∫ r) approché par la règle du trapèze.

Le caractère market-consistent se vérifie par les tests de martingale : l'espérance du
déflateur doit redonner le prix zéro-coupon initial, et l'espérance de l'actif déflaté
doit rester égale à sa valeur initiale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Scenarios:
    """Trajectoires économiques simulées. Dimensions : (nb_scenarios, horizon)."""
    taux_court: np.ndarray          # taux court annualisé sur [t, t+1[
    deflateur: np.ndarray           # facteur d'actualisation stochastique en fin d'année t
    taux_10ans: np.ndarray          # taux zéro-coupon 10 ans observé en début d'année t
    rendement_action: np.ndarray    # performance totale de l'action sur l'année
    rendement_immobilier: np.ndarray
    courbe_initiale: pd.DataFrame

    @property
    def nb_scenarios(self) -> int:
        return self.taux_court.shape[0]

    @property
    def horizon(self) -> int:
        return self.taux_court.shape[1]


class HullWhite:
    """Modèle à un facteur dr = (θ(t) − a r) dt + σ dW, ajusté sur la courbe initiale."""

    def __init__(self, courbe: pd.DataFrame, a: float, sigma: float):
        self.a, self.sigma = a, sigma
        t = courbe["maturite"].to_numpy(float)
        df = courbe["facteur_actualisation"].to_numpy(float)
        self.t = np.concatenate([[0.0], t])
        self.log_df = np.concatenate([[0.0], np.log(df)])
        # forward instantané f(0,t) par différences finies sur -ln P
        self.forward = np.gradient(-self.log_df, self.t)
        self.r0 = float(self.forward[0])

    def p0(self, t: np.ndarray | float) -> np.ndarray:
        return np.exp(np.interp(t, self.t, self.log_df))

    def f0(self, t: np.ndarray | float) -> np.ndarray:
        return np.interp(t, self.t, self.forward)

    def alpha(self, t: np.ndarray | float) -> np.ndarray:
        return self.f0(t) + (self.sigma ** 2) / (2 * self.a ** 2) * (1 - np.exp(-self.a * np.asarray(t))) ** 2

    def prix_zc(self, t: float, maturite: float, r: np.ndarray) -> np.ndarray:
        """P(t, t+maturité) conditionnellement au taux court r."""
        T = t + maturite
        B = (1 - np.exp(-self.a * maturite)) / self.a
        lnA = (np.log(self.p0(T) / self.p0(t)) + B * self.f0(t)
               - (self.sigma ** 2) / (4 * self.a) * (1 - np.exp(-2 * self.a * t)) * B ** 2)
        return np.exp(lnA - B * r)

    def simuler(self, nb_scenarios: int, horizon: int, rng: np.random.Generator,
                bruits: np.ndarray | None = None) -> np.ndarray:
        """Taux court simulé aux dates 0, 1, ..., horizon (pas annuel exact)."""
        a, s = self.a, self.sigma
        e = np.exp(-a)
        ecart_type = s * np.sqrt((1 - np.exp(-2 * a)) / (2 * a))
        z = rng.standard_normal((nb_scenarios, horizon)) if bruits is None else bruits
        r = np.zeros((nb_scenarios, horizon + 1))
        r[:, 0] = self.r0
        alphas = self.alpha(np.arange(horizon + 1))
        for k in range(horizon):
            r[:, k + 1] = r[:, k] * e + alphas[k + 1] - alphas[k] * e + ecart_type * z[:, k]
        return r


def _bruits_correles(rng, nb_scenarios: int, horizon: int, correlations: np.ndarray,
                     antithetiques: bool) -> np.ndarray:
    """Tirages gaussiens corrélés, éventuellement antithétiques. Sortie : (3, n, H)."""
    n = nb_scenarios // 2 if antithetiques else nb_scenarios
    L = np.linalg.cholesky(correlations)
    z = rng.standard_normal((3, n, horizon))
    z = np.einsum("ij,jnh->inh", L, z)
    if antithetiques:
        z = np.concatenate([z, -z], axis=1)
        if z.shape[1] < nb_scenarios:                       # nombre impair de scénarios
            z = np.concatenate([z, z[:, :1]], axis=1)
    return z


def generer_scenarios(courbe: pd.DataFrame, hyp: dict[str, Any]) -> Scenarios:
    p = hyp["esg"]
    n, H = p["nb_scenarios"], p["horizon"]
    rng = np.random.default_rng(p["graine"])
    rho_a = p["action"]["correlation_taux"]
    rho_i = p["immobilier"]["correlation_taux"]
    correlations = np.array([[1.0, rho_a, rho_i], [rho_a, 1.0, 0.5], [rho_i, 0.5, 1.0]])
    z = _bruits_correles(rng, n, H, correlations, p["antithetiques"])

    hw = HullWhite(courbe, p["hull_white"]["a"], p["hull_white"]["sigma"])
    r = hw.simuler(n, H, rng, bruits=z[0])

    # déflateurs : intégrale du taux court approchée par la règle du trapèze
    integrale = np.cumsum(0.5 * (r[:, :-1] + r[:, 1:]), axis=1)
    deflateur = np.exp(-integrale)

    taux_10ans = np.zeros((n, H))
    for k in range(H):
        taux_10ans[:, k] = hw.prix_zc(float(k), 10.0, r[:, k]) ** (-0.1) - 1.0

    sa, si = p["action"]["volatilite"], p["immobilier"]["volatilite"]
    derive = 0.5 * (r[:, :-1] + r[:, 1:])
    rendement_action = np.exp(derive - 0.5 * sa ** 2 + sa * z[1]) - 1.0
    rendement_immobilier = np.exp(derive - 0.5 * si ** 2 + si * z[2]) - 1.0
    return Scenarios(r[:, :-1], deflateur, taux_10ans, rendement_action, rendement_immobilier, courbe)


def scenario_central(courbe: pd.DataFrame, hyp: dict[str, Any]) -> Scenarios:
    """Scénario unique déterministe (taux forward, aucune volatilité).

    Il sert de référence pour isoler la valeur temps des options et garanties :
    TVOG = BE stochastique − BE du scénario central.
    """
    H = hyp["esg"]["horizon"]
    forward = np.interp(np.arange(1, H + 1), courbe["maturite"], courbe["forward_1an"])[None, :]
    df = np.interp(np.arange(1, H + 1), courbe["maturite"],
                   courbe["facteur_actualisation"])[None, :]
    hw = HullWhite(courbe, hyp["esg"]["hull_white"]["a"], 0.0)
    taux_10ans = np.array([[float(hw.prix_zc(float(k), 10.0, np.array([hw.f0(float(k))]))[0]) ** -0.1 - 1
                            for k in range(H)]])
    return Scenarios(forward, df, taux_10ans, forward.copy(), forward.copy(), courbe)


# ---------------------------------------------------------------------------
def tests_martingale(sc: Scenarios) -> pd.DataFrame:
    """Compare les prix simulés aux prix de marché initiaux (en points de base)."""
    courbe = sc.courbe_initiale.set_index("maturite")["facteur_actualisation"]
    lignes = []
    for maturite in (1, 5, 10, 20, 30, 40):
        if maturite > sc.horizon:
            continue
        simule = float(sc.deflateur[:, maturite - 1].mean())
        marche = float(courbe.loc[maturite])
        lignes.append({"test": f"E[déflateur {maturite} ans]", "simule": simule, "reference": marche,
                       "ecart_bp": 1e4 * (simule / marche - 1)})
    for nom, rendement in (("action", sc.rendement_action), ("immobilier", sc.rendement_immobilier)):
        for maturite in (10, 20):
            if maturite > sc.horizon:
                continue
            valeur = np.prod(1 + rendement[:, :maturite], axis=1) * sc.deflateur[:, maturite - 1]
            simule = float(valeur.mean())
            lignes.append({"test": f"E[{nom} déflaté {maturite} ans]", "simule": simule,
                           "reference": 1.0, "ecart_bp": 1e4 * (simule - 1.0)})
    return pd.DataFrame(lignes)
