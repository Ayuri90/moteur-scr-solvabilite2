"""Chargement et accès aux hypothèses du projet."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

RACINE = Path(__file__).resolve().parents[1]


def charger_hypotheses(chemin: str | Path | None = None) -> dict[str, Any]:
    """Lit le fichier YAML d'hypothèses et vérifie les contraintes de base."""
    chemin = Path(chemin) if chemin else RACINE / "config" / "hypotheses.yaml"
    with open(chemin, encoding="utf-8") as f:
        hyp = yaml.safe_load(f)
    _verifier(hyp)
    return hyp


def _verifier(hyp: dict[str, Any]) -> None:
    alloc = hyp["allocation"]
    total = sum(alloc.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"L'allocation d'actifs somme à {total:.4f} au lieu de 1.")
    for cle in ("corporate_financieres", "corporate_non_financieres"):
        rep = hyp["obligations"][cle]["repartition_cqs"]
        if abs(sum(rep.values()) - 1.0) > 1e-9:
            raise ValueError(f"La répartition CQS de {cle} ne somme pas à 1.")
    pays = hyp["obligations"]["souverains"]["pays"]
    if abs(sum(p["poids"] for p in pays.values()) - 1.0) > 1e-9:
        raise ValueError("Les poids pays des souverains ne somment pas à 1.")


def date_evaluation(hyp: dict[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(hyp["general"]["date_evaluation"])


def generateur(hyp: dict[str, Any], decalage: int = 0) -> np.random.Generator:
    """Générateur aléatoire reproductible ; `decalage` isole chaque module."""
    return np.random.default_rng(hyp["general"]["graine_aleatoire"] + decalage)


def poids_normalises(n: int, rng: np.random.Generator, sigma: float = 0.6) -> np.ndarray:
    """Poids log-normaux normalisés (tailles de lignes hétérogènes mais réalistes)."""
    w = rng.lognormal(mean=0.0, sigma=sigma, size=n)
    return w / w.sum()
