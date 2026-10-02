"""Extraction des paramètres officiels du fichier EIOPA « Risk-free interest rate term structures ».

Le classeur mensuel de l'EIOPA contient, outre les courbes :
  - onglet `Shocks`  : les facteurs relatifs de choc de taux des articles 166 et 167,
    pour toutes les maturités de 1 à 150 ans ;
  - onglets de courbes : LLP, point de convergence, UFR, alpha, CRA et ajustement pour
    volatilité, par devise.

Sorties, dans `sources/` :
  - facteurs_choc_taux.csv   maturite;hausse;baisse
  - parametres_eiopa.csv     paramètre;valeur, pour la devise retenue

Usage :
    python scripts/extraire_parametres_eiopa.py sources/EIOPA_RFR_..._Term_Structures.xlsx [Euro]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
SORTIE = RACINE / "sources"
ETIQUETTES = ["identifiant", "coupon_freq", "llp", "convergence", "ufr", "alpha", "cra", "va"]


def facteurs_choc(chemin: Path) -> pd.DataFrame:
    """Facteurs relatifs de l'onglet `Shocks` : colonnes maturité, baisse, hausse."""
    brut = pd.read_excel(chemin, sheet_name="Shocks", header=None)
    numerique = brut.apply(pd.to_numeric, errors="coerce")
    lignes = numerique[1].notna() & numerique[3].notna() & numerique[4].notna()
    table = numerique.loc[lignes, [1, 3, 4]]
    table.columns = ["maturite", "baisse", "hausse"]
    return table.astype({"maturite": int}).reset_index(drop=True)


def parametres(chemin: Path, devise: str) -> pd.Series:
    """Paramètres de construction de la courbe pour une devise (LLP, UFR, alpha, CRA, VA)."""
    brut = pd.read_excel(chemin, sheet_name="RFR_spot_no_VA", header=None)
    positions = np.argwhere(brut.astype(str).apply(lambda s: s.str.strip()).values == devise)
    if len(positions) == 0:
        raise ValueError(f"Devise « {devise} » absente de l'onglet RFR_spot_no_VA.")
    ligne, colonne = positions[0]
    valeurs = brut.iloc[ligne + 1:ligne + 1 + len(ETIQUETTES), colonne]
    return pd.Series(valeurs.to_numpy(), index=ETIQUETTES, name="valeur")


def principal(chemin: Path, devise: str = "Euro") -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    table = facteurs_choc(chemin)
    table.to_csv(SORTIE / "facteurs_choc_taux.csv", index=False)

    p = parametres(chemin, devise)
    p.rename_axis("parametre").to_frame().to_csv(SORTIE / "parametres_eiopa.csv")

    print(f"  facteurs_choc_taux.csv   {len(table)} maturités "
          f"(1 an : hausse {table.iloc[0]['hausse']:.2f}, baisse {table.iloc[0]['baisse']:.2f})")
    print(f"  parametres_eiopa.csv     {p['identifiant']}")
    for cle in ("llp", "convergence", "ufr", "alpha", "cra"):
        print(f"    {cle:12s} {p[cle]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    principal(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "Euro")
