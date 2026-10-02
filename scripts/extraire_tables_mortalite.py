"""Extraction des tables de mortalité réglementaires françaises vers des CSV exploitables.

Entrées : les deux classeurs officiels
  - TH-TF 00-02 : un onglet, quatre tables (TF décès, TF vie, TH décès, TH vie), en Lx et qx ;
  - TGF05-TGH05 : deux onglets, tables générationnelles en Lx par année de naissance.

Sorties, dans `sources/` :
  - TH0002.csv, TF0002.csv                  âge;qx — tables en cas de décès
  - TH0002_vie.csv, TF0002_vie.csv          âge;qx — tables en cas de vie
  - TGH05.csv, TGF05.csv                    generation;age;qx — format générationnel complet
  - TGH05_<annee>.csv, TGF05_<annee>.csv    âge;qx — diagonale à la date d'évaluation,
                                            pour les outils qui n'acceptent qu'une table à une
                                            dimension

Usage :
    python scripts/extraire_tables_mortalite.py chemin/TH-TF.xls chemin/TGF05-TGH05.xls [annee]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
SORTIE = RACINE / "sources"
AGE_MAX = 120


def _qx_depuis_lx(lx: pd.Series) -> pd.DataFrame:
    """q_x = 1 - l_{x+1} / l_x, calculé là où les effectifs sont renseignés."""
    lx = lx.astype(float)
    suivant = lx.shift(-1)
    qx = 1.0 - suivant / lx
    qx = qx.where(lx > 0)
    return pd.DataFrame({"age": lx.index.astype(int), "qx": qx.to_numpy()}).dropna()


def extraire_th_tf(chemin: Path) -> dict[str, pd.DataFrame]:
    """Les quatre tables du classeur TH-TF 00-02 (colonnes Lx et qx par table)."""
    brut = pd.read_excel(chemin, sheet_name=0, header=None)
    ages = pd.to_numeric(brut[0], errors="coerce")
    lignes = ages.notna() & (ages <= AGE_MAX)
    ages = ages[lignes].astype(int)
    colonnes = {"TF0002": 1, "TF0002_vie": 3, "TH0002": 5, "TH0002_vie": 7}
    tables = {}
    for nom, colonne in colonnes.items():
        lx = pd.to_numeric(brut.loc[lignes, colonne], errors="coerce")
        lx.index = ages
        tables[nom] = _qx_depuis_lx(lx.dropna())
    return tables


def extraire_generationnelle(chemin: Path, onglet: str) -> pd.DataFrame:
    """Table générationnelle en format long : generation, age, qx."""
    brut = pd.read_excel(chemin, sheet_name=onglet, header=None)
    generations = pd.to_numeric(brut.iloc[1, 1:], errors="coerce").dropna().astype(int)
    ages = pd.to_numeric(brut.iloc[2:, 0], errors="coerce")
    valides = ages.notna() & (ages <= AGE_MAX)
    lx = brut.iloc[2:, 1:1 + len(generations)][valides.to_numpy()]
    lx.index = ages[valides].astype(int)
    lx.columns = generations.to_numpy()

    lignes = []
    for generation in lx.columns:
        colonne = pd.to_numeric(lx[generation], errors="coerce")
        table = _qx_depuis_lx(colonne[colonne > 0])
        table.insert(0, "generation", generation)
        lignes.append(table)
    return pd.concat(lignes, ignore_index=True)


def diagonale(table_longue: pd.DataFrame, annee: int) -> pd.DataFrame:
    """Table à une dimension : pour chaque âge x, le q_x de la génération vivante en `annee`."""
    table = table_longue.copy()
    table["generation_cible"] = annee - table["age"]
    extrait = table[table["generation"] == table["generation_cible"]]
    if extrait.empty:                                   # repli sur la génération la plus proche
        lignes = []
        for age, sous in table.groupby("age"):
            cible = annee - age
            proche = sous.iloc[(sous["generation"] - cible).abs().argsort().iloc[0]]
            lignes.append({"age": age, "qx": proche["qx"]})
        return pd.DataFrame(lignes)
    return extrait[["age", "qx"]].sort_values("age").reset_index(drop=True)


def principal(chemin_th_tf: Path, chemin_tg: Path, annee: int = 2025) -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    ecrites = []

    for nom, table in extraire_th_tf(chemin_th_tf).items():
        chemin = SORTIE / f"{nom}.csv"
        table.to_csv(chemin, index=False)
        ecrites.append((chemin, len(table), f"âges {table['age'].min()}–{table['age'].max()}"))

    for onglet in ("TGH05", "TGF05"):
        longue = extraire_generationnelle(chemin_tg, onglet)
        chemin = SORTIE / f"{onglet}.csv"
        longue.to_csv(chemin, index=False)
        ecrites.append((chemin, len(longue),
                        f"générations {longue['generation'].min()}–{longue['generation'].max()}"))
        plate = diagonale(longue, annee)
        chemin = SORTIE / f"{onglet}_{annee}.csv"
        plate.to_csv(chemin, index=False)
        ecrites.append((chemin, len(plate), f"diagonale {annee}"))

    for chemin, nb, detail in ecrites:
        print(f"  {chemin.name:22s} {nb:6d} lignes   {detail}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    annee = int(sys.argv[3]) if len(sys.argv) > 3 else 2025
    principal(Path(sys.argv[1]), Path(sys.argv[2]), annee)
