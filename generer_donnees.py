"""Génère l'ensemble des données de la compagnie composite fictive.

Usage :
    python generer_donnees.py                      # hypothèses par défaut
    python generer_donnees.py --config mon_scenario.yaml
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from scr_data.actifs import generer_inventaire
from scr_data.bilan import construire_bilan, controles, generer_contreparties
from scr_data.config import RACINE, charger_hypotheses
from scr_data.courbe import Actualisation, construire_courbe
from scr_data.export import exporter_csv, exporter_excel
from scr_data.mortalite import charger_tables
from scr_data.non_vie_sante import generer_non_vie, generer_sante_slt, triangle_en_matrice
from scr_data.passif_vie import generer_passif_vie


def generer(chemin_config: str | None = None, ecrire: bool = True) -> dict[str, pd.DataFrame]:
    hyp = charger_hypotheses(chemin_config)

    courbe, info_courbe = construire_courbe(hyp)
    actu = Actualisation(courbe)
    tables = charger_tables(hyp)

    inventaire = generer_inventaire(hyp, actu)
    passif_vie = generer_passif_vie(hyp, tables, actu)
    with warnings.catch_warnings(record=True) as alertes:
        warnings.simplefilter("always")
        non_vie = generer_non_vie(hyp)
    sante_slt = generer_sante_slt(hyp, tables, actu)
    contreparties = generer_contreparties(hyp)

    bilan = construire_bilan(hyp, inventaire, passif_vie, non_vie, sante_slt)
    ctrl = controles(hyp, inventaire, passif_vie, non_vie, sante_slt)

    tables_sortie = {
        "courbe_taux_sans_risque": courbe,
        "inventaire_actifs": inventaire,
        **passif_vie,
        "mp_sante_slt": sante_slt,
        "triangles_reglements": non_vie["triangles_reglements"],
        "ultimes_vrais": non_vie["ultimes_vrais"],
        "volumes_primes_reserves": non_vie["volumes_primes_reserves"],
        "exposition_cat": non_vie["exposition_cat"],
        "contreparties": contreparties,
        "bilan_prudentiel_indicatif": bilan,
        "controles": ctrl,
    }

    if ecrire:
        dossier = RACINE / hyp["general"]["dossier_sortie"]
        exporter_csv(tables_sortie, dossier)
        tri = non_vie["triangles_reglements"]
        with pd.ExcelWriter(dossier / "triangles_matrices.xlsx") as xw:
            for lob, sous in tri.groupby("lob"):
                triangle_en_matrice(sous).to_excel(xw, sheet_name=lob[:31])
        exporter_excel(dossier / "synthese_compagnie.xlsx", hyp, bilan, ctrl, inventaire, courbe, info_courbe)
        _resume(hyp, bilan, ctrl, info_courbe, dossier, alertes)
    return tables_sortie


def _resume(hyp, bilan, ctrl, info_courbe, dossier: Path, alertes) -> None:
    actif = bilan.loc[bilan["cote"] == "Actif", "montant"].sum()
    passif = bilan.loc[bilan["cote"] == "Passif", "montant"].sum()
    print(f"\n{hyp['general']['nom_compagnie']} — évaluation au {hyp['general']['date_evaluation']}")
    alpha = f", alpha = {info_courbe['alpha']:.4f}" if info_courbe["alpha"] else ""
    print(f"Courbe : {info_courbe['source']}{alpha}")
    print(f"Total actif  : {actif:10,.1f} M€")
    print(f"Total passif : {passif:10,.1f} M€")
    print(f"Fonds propres: {actif - passif:10,.1f} M€  ({(actif - passif) / actif:.1%} de l'actif)")
    nb_alertes = (ctrl["statut"] == "ALERTE").sum()
    print(f"Contrôles    : {(ctrl['statut'] == 'OK').sum()} OK, {nb_alertes} ALERTE(S)")
    for a in alertes:
        print(f"  ! {a.message}")
    print(f"Sorties      : {dossier}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="chemin vers un fichier d'hypothèses YAML")
    generer(parser.parse_args().config)
