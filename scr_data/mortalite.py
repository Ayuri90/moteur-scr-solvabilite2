"""Tables de mortalité : fichiers réglementaires (TH/TF 00-02, TGH/TGF 05) ou Makeham."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import RACINE
from .courbe import Actualisation

AGE_MAX = 120


class TableMortalite:
    def __init__(self, qx: np.ndarray, nom: str):
        qx = np.clip(np.asarray(qx, dtype=float), 0.0, 1.0)
        qx[-1] = 1.0
        self.qx = qx
        self.nom = nom

    @classmethod
    def makeham(cls, A: float, B: float, c: float, nom: str) -> "TableMortalite":
        ages = np.arange(AGE_MAX + 1)
        # q_x = 1 - exp(-∫_x^{x+1} (A + B c^s) ds)
        integrale = A + B * (c ** ages) * (c - 1.0) / np.log(c)
        return cls(1.0 - np.exp(-integrale), nom)

    @classmethod
    def depuis_csv(cls, chemin: str, nom: str) -> "TableMortalite":
        df = pd.read_csv(RACINE / chemin)
        qx = np.ones(AGE_MAX + 1)
        ages = df["age"].to_numpy(int)
        qx[ages[ages <= AGE_MAX]] = df["qx"].to_numpy(float)[ages <= AGE_MAX]
        return cls(qx, nom)

    def ajuster(self, facteur: float, nom: str | None = None) -> "TableMortalite":
        return TableMortalite(self.qx * facteur, nom or f"{self.nom} x{facteur}")

    def survie(self, age: int, n: int) -> np.ndarray:
        """Probabilités de survie k_p_x pour k = 0..n (vecteur de longueur n+1)."""
        age = int(min(max(age, 0), AGE_MAX))
        q = self.qx[age:age + n]
        if len(q) < n:
            q = np.concatenate([q, np.ones(n - len(q))])
        return np.concatenate([[1.0], np.cumprod(1.0 - q)])

    def annuite_viagere(self, age: int, actu: Actualisation, terme_echu: bool = True,
                        duree_max: int | None = None) -> float:
        """Valeur actuelle d'une rente de 1 par an payable tant que l'assuré est en vie."""
        n = duree_max if duree_max is not None else AGE_MAX - int(age) + 1
        p = self.survie(age, n)
        t = np.arange(1, n + 1) if terme_echu else np.arange(0, n)
        probas = p[1:] if terme_echu else p[:-1]
        return float(np.sum(probas * actu.df(t)))


class TableGenerationnelle(TableMortalite):
    """Table par génération (TGH05, TGF05).

    Une personne d'âge `x` à la date d'évaluation appartient à la génération
    `annee_reference - x` : c'est cette colonne qui est utilisée pour toute sa projection,
    ce qui intègre les gains de longévité futurs. L'attribut `qx` contient la diagonale,
    c'est-à-dire la mortalité de chaque âge à la date d'évaluation, pour les traitements
    qui n'acceptent qu'une table à une dimension.
    """

    def __init__(self, colonnes: dict[int, np.ndarray], annee_reference: int, nom: str,
                 facteur: float = 1.0):
        self.colonnes = colonnes
        self.annee_reference = annee_reference
        self.facteur = facteur
        self.nom = nom
        self.generations = np.array(sorted(colonnes))
        diagonale = np.ones(AGE_MAX + 1)
        for age in range(AGE_MAX + 1):
            colonne = self._colonne(annee_reference - age)
            if not np.isnan(colonne[age]):
                diagonale[age] = colonne[age]
        self.qx = np.clip(diagonale * facteur, 0.0, 1.0)

    @classmethod
    def depuis_csv_long(cls, chemin: str, annee_reference: int, nom: str) -> "TableGenerationnelle":
        """Lit un fichier `generation,age,qx` (sortie de scripts/extraire_tables_mortalite.py)."""
        df = pd.read_csv(RACINE / chemin)
        colonnes = {}
        for generation, sous in df.groupby("generation"):
            colonne = np.full(AGE_MAX + 1, np.nan)
            ages = sous["age"].to_numpy(int)
            garde = ages <= AGE_MAX
            colonne[ages[garde]] = sous["qx"].to_numpy(float)[garde]
            colonnes[int(generation)] = colonne
        return cls(colonnes, annee_reference, nom)

    def _colonne(self, generation: int) -> np.ndarray:
        """Colonne de la génération demandée, à défaut la génération disponible la plus proche."""
        if generation not in self.colonnes:
            generation = int(self.generations[np.argmin(np.abs(self.generations - generation))])
        return self.colonnes[generation]

    def ajuster(self, facteur: float, nom: str | None = None) -> "TableGenerationnelle":
        return TableGenerationnelle(self.colonnes, self.annee_reference,
                                    nom or f"{self.nom} x{facteur}", self.facteur * facteur)

    def survie(self, age: int, n: int) -> np.ndarray:
        age = int(min(max(age, 0), AGE_MAX))
        colonne = self._colonne(self.annee_reference - age)
        indices = np.clip(np.arange(age, age + n), 0, AGE_MAX)
        q = colonne[indices]
        if np.isnan(q).any():                       # prolongement par la dernière valeur connue
            connus = np.where(~np.isnan(q))[0]
            if len(connus) == 0:
                q = self.qx[indices]
            else:
                q = pd.Series(q).ffill().bfill().to_numpy()
        q = np.clip(q * self.facteur, 0.0, 1.0)
        return np.concatenate([[1.0], np.cumprod(1.0 - q)])


def charger_tables(hyp: dict[str, Any]) -> dict[str, TableMortalite]:
    hm = hyp["mortalite"]
    if hm["source"] == "fichier":
        f = hm["fichiers"]
        annee = int(str(hyp["general"]["date_evaluation"])[:4])
        return {
            "homme": TableMortalite.depuis_csv(f["homme"], "TH 00-02"),
            "femme": TableMortalite.depuis_csv(f["femme"], "TF 00-02"),
            "rente_homme": _charger_rente(f["rente_homme"], annee, "TGH05"),
            "rente_femme": _charger_rente(f["rente_femme"], annee, "TGF05"),
        }
    mk = hm["makeham"]
    h = TableMortalite.makeham(**mk["homme"], nom="Makeham homme")
    fe = TableMortalite.makeham(**mk["femme"], nom="Makeham femme")
    ab = hm["abattement_rentes"]
    return {"homme": h, "femme": fe,
            "rente_homme": h.ajuster(ab, "Makeham homme rentiers"),
            "rente_femme": fe.ajuster(ab, "Makeham femme rentiers")}


def _charger_rente(chemin: str, annee_reference: int, nom: str) -> TableMortalite:
    """Table de rente : format générationnel si le fichier porte une colonne `generation`."""
    entetes = pd.read_csv(RACINE / chemin, nrows=0).columns
    if "generation" in entetes:
        return TableGenerationnelle.depuis_csv_long(chemin, annee_reference, nom)
    return TableMortalite.depuis_csv(chemin, nom)
