"""Modèle interne partiel non-vie (V4).

Trois blocs de risque simulés séparément puis agrégés par copule :
- réserves : bootstrap ODP sur le triangle (rééchantillonnage des résidus de Pearson,
  puis bruit de processus gamma), qui donne la distribution complète du boni-mali ;
- primes : loi log-normale ajustée sur l'historique des S/P ultimes estimés par Chain Ladder,
  majorée d'une incertitude de paramètre ;
- catastrophe : fréquence de Poisson et sévérité GPD au-delà d'un seuil, nette du traité XL.

Le SCR interne est la VaR à 99,5 % de la perte agrégée, mesurée en écart à la moyenne.
Le module produit aussi les écarts types spécifiques (USP) du risque de réserve.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scr.passif import chain_ladder


# ---------------------------------------------------------------------------
# Réserves : bootstrap ODP
# ---------------------------------------------------------------------------
def _increments(triangle: np.ndarray) -> np.ndarray:
    incr = triangle.copy()
    incr[:, 1:] = triangle[:, 1:] - triangle[:, :-1]
    return incr


def bootstrap_odp(triangle: pd.DataFrame, nb_simulations: int, rng: np.random.Generator,
                  ajustement_biais: bool = True) -> dict[str, Any]:
    """Distribution des réserves par bootstrap ODP (England-Verrall)."""
    tri = triangle.to_numpy(dtype=float)
    n = tri.shape[0]
    observe = ~np.isnan(tri)
    cl = chain_ladder(triangle)
    complet = cl["triangle_complete"]

    # triangle ajusté : retro-projection des ultimes par les facteurs de développement
    ajuste = np.full_like(tri, np.nan)
    ajuste[:, n - 1] = complet[:, n - 1]
    for j in range(n - 2, -1, -1):
        ajuste[:, j] = ajuste[:, j + 1] / cl["facteurs"][j]
    incr_obs, incr_aj = _increments(tri), _increments(ajuste)

    nb_donnees = int(observe.sum())
    nb_parametres = 2 * n - 1
    correction = np.sqrt(nb_donnees / (nb_donnees - nb_parametres)) if ajustement_biais else 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        residus = (incr_obs - incr_aj) / np.sqrt(np.abs(incr_aj)) * correction
    residus = residus[observe & np.isfinite(residus)]
    residus = residus - residus.mean()
    phi = float(np.sum(residus ** 2) / max(nb_donnees - nb_parametres, 1))

    reserves = np.empty(nb_simulations)
    for k in range(nb_simulations):
        tirage = rng.choice(residus, size=(n, n), replace=True)
        pseudo = incr_aj + tirage * np.sqrt(np.abs(incr_aj))
        pseudo = np.where(observe, pseudo, np.nan)
        cumule = np.nancumsum(np.nan_to_num(pseudo), axis=1)
        cumule = np.where(observe, cumule, np.nan)
        facteurs = []
        for j in range(n - 1):
            num = np.nansum(cumule[: n - j - 1, j + 1])
            den = np.nansum(cumule[: n - j - 1, j])
            facteurs.append(num / den if den else 1.0)
        projete = cumule.copy()
        for i in range(n):
            for j in range(n - i, n):
                projete[i, j] = projete[i, j - 1] * facteurs[j - 1]
        futur = _increments(projete)
        masque_futur = ~observe
        attendu = np.clip(futur[masque_futur], 1e-6, None)
        # bruit de processus : loi gamma de moyenne l'incrément et de variance phi x incrément
        simule = rng.gamma(shape=attendu / phi, scale=phi)
        reserves[k] = simule.sum()
    return {"reserves": reserves, "phi": phi, "reserve_centrale": float(cl["reserves"].sum()),
            "nb_residus": len(residus)}


# ---------------------------------------------------------------------------
# Primes
# ---------------------------------------------------------------------------
def ajuster_sp(triangle: pd.DataFrame, primes: pd.Series) -> tuple[float, float]:
    """Moyenne et écart type logarithmiques des S/P ultimes estimés par Chain Ladder."""
    cl = chain_ladder(triangle)
    sp = cl["ultimes"] / primes.to_numpy(float)
    logs = np.log(sp[sp > 0])
    return float(logs.mean()), float(logs.std(ddof=1))


def simuler_primes(mu: float, sigma: float, volume: float, nb_simulations: int,
                   rng: np.random.Generator, incertitude: float = 0.0) -> np.ndarray:
    sigma_total = sigma * (1 + incertitude)
    return volume * rng.lognormal(mu, sigma_total, nb_simulations)


# ---------------------------------------------------------------------------
# Catastrophe
# ---------------------------------------------------------------------------
def simuler_catastrophe(params: dict[str, Any], nb_simulations: int, rng: np.random.Generator,
                        priorite: float | None = None, portee: float | None = None) -> np.ndarray:
    """Fréquence de Poisson et sévérités GPD ; application éventuelle du traité XL par événement."""
    nb = rng.poisson(params["frequence_poisson"], nb_simulations)
    pertes = np.zeros(nb_simulations)
    total = int(nb.sum())
    if total:
        u = rng.random(total)
        xi, beta, seuil = params["xi"], params["beta"], params["seuil_gpd"]
        severites = seuil + beta / xi * ((1 - u) ** (-xi) - 1)
        if priorite is not None:
            severites = np.minimum(severites, priorite) + np.maximum(severites - priorite - portee, 0.0)
        indices = np.repeat(np.arange(nb_simulations), nb)
        np.add.at(pertes, indices, severites)
    return pertes


# ---------------------------------------------------------------------------
# Agrégation par copule
# ---------------------------------------------------------------------------
def copule(matrice: np.ndarray, nb_simulations: int, rng: np.random.Generator,
           famille: str = "gaussienne", degres_liberte: int = 6) -> np.ndarray:
    """Tirages uniformes corrélés (dimension = taille de la matrice)."""
    L = np.linalg.cholesky(matrice)
    z = rng.standard_normal((nb_simulations, matrice.shape[0])) @ L.T
    if famille == "student":
        w = rng.chisquare(degres_liberte, nb_simulations)[:, None]
        z = z * np.sqrt(degres_liberte / w)
        from scipy.stats import t
        return t.cdf(z, degres_liberte)
    from scipy.stats import norm
    return norm.cdf(z)


def coupler(marginales: list[np.ndarray], uniformes: np.ndarray) -> np.ndarray:
    """Relie des marginales simulées indépendamment par les rangs de la copule."""
    couple = np.empty((len(uniformes), len(marginales)))
    for k, echantillon in enumerate(marginales):
        tri = np.sort(echantillon)
        positions = np.clip((uniformes[:, k] * len(tri)).astype(int), 0, len(tri) - 1)
        couple[:, k] = tri[positions]
    return couple


# ---------------------------------------------------------------------------
def mesures_risque(pertes: np.ndarray, quantile: float = 0.995) -> dict[str, float]:
    moyenne = float(pertes.mean())
    var = float(np.quantile(pertes, quantile))
    queue = pertes[pertes >= var]
    return {"moyenne": moyenne, "ecart_type": float(pertes.std(ddof=1)), "var": var,
            "tvar": float(queue.mean()) if len(queue) else var,
            "scr": var - moyenne, "scr_tvar": (float(queue.mean()) if len(queue) else var) - moyenne,
            "coefficient_variation": float(pertes.std(ddof=1) / moyenne) if moyenne else np.nan}


# ---------------------------------------------------------------------------
# USP réserve
# ---------------------------------------------------------------------------
def usp_reserve(distribution: np.ndarray, sigma_standard: float, nb_annees: int,
                credibilite: dict[int, float]) -> dict[str, float]:
    """Écart type spécifique du risque de réserve, mélangé au paramètre standard par crédibilité."""
    sigma_usp = float(distribution.std(ddof=1) / distribution.mean())
    c = credibilite.get(nb_annees, max(credibilite.values()))
    return {"sigma_usp": sigma_usp, "credibilite": c, "sigma_standard": sigma_standard,
            "sigma_retenu": c * sigma_usp + (1 - c) * sigma_standard}
