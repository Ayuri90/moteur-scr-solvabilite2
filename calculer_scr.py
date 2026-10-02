"""Calcul du SCR — étape 1 de la V1 : valorisation des passifs, risque de marché, contrepartie.

Usage :
    python calculer_scr.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yaml

from scr import agregation, contrepartie
from scr.non_vie import MoteurNonVie
from scr.marche import MoteurMarche
from scr.passif import Passifs
from scr.rapport import ecrire_rapport
from scr.souscription import MoteurSante, MoteurVie
from scr_data.config import RACINE, charger_hypotheses
from scr_data.courbe import Actualisation, construire_courbe
from scr_data.mortalite import charger_tables

TABLES = ["inventaire_actifs", "mp_epargne_euro", "mp_epargne_uc", "mp_temporaire_deces",
          "mp_rentes_viageres", "mp_sante_slt", "triangles_reglements", "ultimes_vrais",
          "volumes_primes_reserves", "contreparties", "exposition_cat"]


def charger_donnees(dossier: Path) -> dict[str, pd.DataFrame]:
    manquantes = [t for t in TABLES if not (dossier / f"{t}.csv").exists()]
    if manquantes:
        raise FileNotFoundError(f"Tables absentes de {dossier} : {manquantes}. "
                                "Lancez d'abord `python generer_donnees.py`.")
    return {t: pd.read_csv(dossier / f"{t}.csv", sep=";", decimal=",") for t in TABLES}


def charger_params(chemin: Path | None = None) -> dict:
    chemin = chemin or RACINE / "config" / "params_reglementaires.yaml"
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculer(ecrire: bool = True, hyp: dict | None = None, params: dict | None = None,
             courbe: pd.DataFrame | None = None) -> dict:
    debut = time.time()
    hyp = hyp or charger_hypotheses()
    params = params or charger_params()
    donnees = charger_donnees(RACINE / hyp["general"]["dossier_sortie"])
    if courbe is None:
        courbe, info_courbe = construire_courbe(hyp)
    else:
        info_courbe = {"source": "courbe fournie", "alpha": None}
    actu = Actualisation(courbe)
    tables = charger_tables(hyp)

    passifs = Passifs(donnees, hyp, tables)
    be = passifs.evaluer(actu, courbe)
    bp = hyp["bilan_cible"]["passif"]
    autres_passifs = bp["marge_risque"] + bp["impots_differes_passifs"] + bp["autres_passifs"]

    moteur = MoteurMarche(donnees["inventaire_actifs"], passifs, courbe, hyp, params, autres_passifs)
    table_marche, scr_marche = moteur.calculer()
    table_cp, scr_cp = contrepartie.calculer(donnees["contreparties"], params)
    table_vie, scr_vie = MoteurVie(donnees, hyp, params, tables, actu, courbe).calculer()
    table_sante, scr_sante = MoteurSante(donnees, hyp, params, tables, actu, courbe).calculer()
    moteur_nv = MoteurNonVie(donnees, hyp, params, actu)
    table_nv, detail_nv, scr_nv = moteur_nv.calculer()

    modules = {"marche": scr_marche, "defaut": scr_cp, "vie": scr_vie,
               "sante": scr_sante, "non_vie": scr_nv}
    valeur_bscr, table_bscr = agregation.bscr(modules, params)
    scr_op, detail_op = agregation.operationnel(hyp, be, valeur_bscr, params, donnees)
    lac_tp, lac_dt, detail_lac = agregation.ajustements(valeur_bscr, scr_op, hyp, params)
    scr_total = valeur_bscr + scr_op + lac_tp + lac_dt
    valeur_mcr, table_mcr = agregation.mcr(scr_total, be, donnees, hyp, params,
                                           be["_detail_non_vie"])
    table_s25 = agregation.tableau_s25(modules, valeur_bscr, scr_op, lac_tp, lac_dt, scr_total)
    ratio_scr = moteur.bof_base / scr_total
    ratio_mcr = moteur.bof_base / valeur_mcr

    table_be = pd.DataFrame([{"segment": k, "be": v} for k, v in be.items() if not k.startswith("_")])
    resume = pd.DataFrame([
        {"poste": "Actif total", "montant": moteur.actif_base},
        {"poste": "Best estimate total", "montant": be["total"]},
        {"poste": "Autres passifs (MR, IDP, divers)", "montant": autres_passifs},
        {"poste": "Fonds propres de base (BOF)", "montant": moteur.bof_base},
        {"poste": "SCR marché", "montant": scr_marche},
        {"poste": "SCR contrepartie", "montant": scr_cp},
        {"poste": "SCR souscription vie", "montant": scr_vie},
        {"poste": "SCR souscription santé", "montant": scr_sante},
        {"poste": "SCR souscription non-vie", "montant": scr_nv},
        {"poste": "BSCR", "montant": valeur_bscr},
        {"poste": "Risque opérationnel", "montant": scr_op},
        {"poste": "LAC TP", "montant": lac_tp},
        {"poste": "LAC DT", "montant": lac_dt},
        {"poste": "SCR", "montant": scr_total},
        {"poste": "MCR", "montant": valeur_mcr},
        {"poste": "Ratio de couverture du SCR", "montant": ratio_scr},
        {"poste": "Ratio de couverture du MCR", "montant": ratio_mcr},
    ])

    if ecrire:
        dossier = RACINE / "resultats"
        dossier.mkdir(exist_ok=True)
        for nom, df in (("be_par_segment", table_be), ("scr_marche", table_marche),
                        ("scr_contrepartie", table_cp), ("scr_vie", table_vie),
                        ("scr_sante", table_sante), ("scr_non_vie", table_nv),
                        ("scr_non_vie_detail", detail_nv), ("bscr", table_bscr),
                        ("mcr", table_mcr), ("s25_01", table_s25), ("resume", resume)):
            df.to_csv(dossier / f"{nom}.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
        resultats_partiels = {"s25": table_s25, "marche": table_marche, "vie": table_vie,
                              "sante": table_sante, "non_vie": table_nv, "contrepartie": table_cp,
                              "bscr": table_bscr, "mcr": table_mcr, "be": table_be,
                              "moteur": moteur, "scr": scr_total, "mcr_valeur": valeur_mcr}
        ecrire_rapport(dossier / "rapport_scr.xlsx", hyp, resultats_partiels)
        _afficher(hyp, info_courbe, table_be, table_marche, table_cp, table_vie, table_sante,
                  table_nv, table_s25, table_mcr, moteur, detail_op, detail_lac,
                  ratio_scr, ratio_mcr, time.time() - debut)
    return {"be": table_be, "marche": table_marche, "contrepartie": table_cp, "vie": table_vie,
            "sante": table_sante, "non_vie": table_nv, "non_vie_detail": detail_nv,
            "bscr": table_bscr, "mcr": table_mcr, "s25": table_s25, "resume": resume,
            "moteur": moteur, "modules": modules, "scr": scr_total, "mcr_valeur": valeur_mcr}


def _afficher(hyp, info_courbe, table_be, table_marche, table_cp, table_vie, table_sante,
              table_nv, table_s25, table_mcr, moteur, detail_op, detail_lac,
              ratio_scr, ratio_mcr, duree):
    print(f"\n{hyp['general']['nom_compagnie']} — {hyp['general']['date_evaluation']}")
    print(f"Courbe : {info_courbe['source']}\n")
    print("Best estimate par segment (M€)")
    for r in table_be.itertuples():
        if r.segment != "total":
            print(f"  {r.segment:24s} {r.be:10,.1f}")
    print(f"  {'TOTAL':24s} {table_be.loc[table_be.segment == 'total', 'be'].iloc[0]:10,.1f}\n")
    print(f"Actif total   : {moteur.actif_base:10,.1f} M€")
    print(f"Fonds propres : {moteur.bof_base:10,.1f} M€\n")
    print("SCR marché (M€)")
    for r in table_marche.itertuples():
        print(f"  {r.sous_module:24s} {r.scr:10,.1f}   {r.detail}")
    print("\nSCR contrepartie (M€)")
    for r in table_cp.itertuples():
        print(f"  {r.composante:24s} {r.scr:10,.1f}   {r.detail}")
    for titre, table, colonne in (("SCR souscription vie (M€)", table_vie, "sous_module"),
                                  ("SCR souscription santé (M€)", table_sante, "sous_module"),
                                  ("SCR souscription non-vie (M€)", table_nv, "sous_module")):
        print(f"\n{titre}")
        for r in table.itertuples():
            detail = getattr(r, "detail", "") or ""
            print(f"  {getattr(r, colonne):32s} {r.scr:10,.1f}   {detail}")
    print("\nAgrégation — tableau S.25.01 (M€)")
    for r in table_s25.itertuples():
        print(f"  {r.code}  {r.poste:52s} {r.montant:10,.1f}")
    print(f"\n  dont risque opérationnel : {detail_op}")
    print(f"  dont LAC DT : {detail_lac}")
    print("\nMCR (M€)")
    for r in table_mcr.itertuples():
        print(f"  {r.poste:40s} {r.montant:10,.1f}")
    print(f"\nFonds propres de base : {moteur.bof_base:,.1f} M€")
    print(f"Ratio de couverture du SCR : {ratio_scr:.0%}")
    print(f"Ratio de couverture du MCR : {ratio_mcr:.0%}")
    print(f"Calcul effectué en {duree:.1f} s\n")


if __name__ == "__main__":
    calculer()
