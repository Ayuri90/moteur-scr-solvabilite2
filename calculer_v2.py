"""V2 — modèle ALM stochastique du fonds euro, TVOG, rachats dynamiques et LAC TP explicite.

Le canton épargne euro est revalorisé par le modèle ALM sur les scénarios de l'ESG.
Chaque choc de la formule standard est rejoué deux fois :
- actions du management dynamiques  → SCR net d'absorption ;
- revalorisation figée au taux cible → SCR brut.
L'écart, plafonné par les prestations discrétionnaires futures, donne la LAC TP.

Usage :
    python calculer_v2.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import calculer_scr
from scr import agregation
from scr.alm import ModeleALM, construire_actif_euro, construire_cohortes
from scr.esg import generer_scenarios, scenario_central, tests_martingale
from scr.marche import MoteurMarche, courbe_choquee, facteur_stress_spread, matrice_marche
from scr.passif import Chocs, Passifs, be_epargne_euro
from scr_data.config import RACINE, charger_hypotheses
from scr_data.courbe import Actualisation, construire_courbe
from scr_data.mortalite import charger_tables

CHOCS_MARCHE = ("taux_hausse", "taux_baisse", "action", "immobilier", "spread")
CHOCS_VIE = ("rachat_hausse", "rachat_baisse", "rachat_massif", "frais")


def _stress_spread_moyen(inventaire: pd.DataFrame, params) -> float:
    """Perte relative moyenne de la poche obligataire sous le choc de spread."""
    euro = inventaire[(inventaire["portefeuille"] == "euro")
                      & inventaire["classe"].isin(["obligation", "pret_infrastructure", "monetaire"])]
    exempts = set(params["spread"]["exposition_exemptee"]["codes_pays_eee"])
    perte = vm = 0.0
    for r in euro.itertuples():
        vm += r.valeur_marche
        if r.sous_classe == "souverain" and r.pays in exempts or pd.isna(r.duration_modifiee):
            continue
        stress = facteur_stress_spread(r.cqs, float(r.duration_modifiee), params, bool(r.covered_bond))
        if r.infra_eligible:
            stress *= params["spread"]["infrastructure_reduction"]
        perte += stress * r.valeur_marche
    return perte / vm


def calculer(ecrire: bool = True, hyp: dict | None = None, params: dict | None = None,
             courbe: pd.DataFrame | None = None) -> dict:
    debut = time.time()
    hyp = hyp or charger_hypotheses()
    params = params or calculer_scr.charger_params()
    donnees = calculer_scr.charger_donnees(RACINE / hyp["general"]["dossier_sortie"])
    if courbe is None:
        courbe, _ = construire_courbe(hyp)
    actu = Actualisation(courbe)
    tables = charger_tables(hyp)

    # ---- V1 de référence -------------------------------------------------
    v1 = calculer_scr.calculer(ecrire=False, hyp=hyp, params=params, courbe=courbe)
    moteur_v1: MoteurMarche = v1["moteur"]
    moteur_v1.taux()                                   # alimente moteur_v1.pertes_taux
    be_v1 = v1["be"].set_index("segment")["be"]

    # ---- ESG et modèle ALM ----------------------------------------------
    scenarios = generer_scenarios(courbe, hyp)
    central = scenario_central(courbe, hyp)
    martingales = tests_martingale(scenarios)
    cohortes = construire_cohortes(donnees["mp_epargne_euro"], hyp, tables, hyp["esg"]["horizon"])
    actif = construire_actif_euro(donnees["inventaire_actifs"], hyp)
    alm = ModeleALM(cohortes, actif, hyp)

    be_central = alm.projeter(central, "dynamique").be
    resultat_base = alm.projeter(scenarios, "dynamique")
    be_stoch = resultat_base.be
    be_garanti = alm.projeter(scenarios, "garantie").be
    be_brut_base = alm.projeter(scenarios, "figee").be     # référence du calcul brut
    fdb = be_stoch - be_garanti
    be_det_base = float(be_v1["epargne_euro"])

    # ---- chocs sur le canton euro ---------------------------------------
    stress_oblig = _stress_spread_moyen(donnees["inventaire_actifs"], params)
    scenarios_choques = {
        "taux_hausse": generer_scenarios(courbe_choquee(courbe, params, "hausse"), hyp),
        "taux_baisse": generer_scenarios(courbe_choquee(courbe, params, "baisse"), hyp),
    }
    action = params["action"]["choc_type1"] + params["action"]["ajustement_symetrique"]
    definitions = {
        "taux_hausse": {"scenarios": "taux_hausse"},
        "taux_baisse": {"scenarios": "taux_baisse"},
        "action": {"facteur_actif": {"action": 1 - action}},
        "immobilier": {"facteur_actif": {"immobilier": 1 - params["immobilier"]["choc"]}},
        "spread": {"facteur_actif": {"obligation": 1 - stress_oblig}},
        "rachat_hausse": {"facteur_rachat": 1 + params["vie"]["rachat_hausse"]},
        "rachat_baisse": {"facteur_rachat": 1 - params["vie"]["rachat_baisse"]},
        "rachat_massif": {"rachat_massif": params["vie"]["rachat_massif"]},
        "frais": {"facteur_frais": 1 + params["vie"]["frais_hausse"]},
    }
    massif = _rachat_massif_selectif(cohortes, actif, hyp, scenarios,
                                     params["vie"]["rachat_massif"])
    lignes = []
    for nom, d in definitions.items():
        sc = scenarios_choques.get(d.pop("scenarios", None), scenarios)
        if nom == "rachat_massif":
            net, brut = massif["net"], massif["brut"]
        else:
            net = alm.projeter(sc, "dynamique", **d).be
            brut = alm.projeter(sc, "figee", **d).be
        lignes.append({"choc": nom, "be_net": net, "delta_net": net - be_stoch,
                       "be_brut": brut, "delta_brut": brut - (massif["brut_base"] if nom == "rachat_massif"
                                                                  else be_brut_base),
                       "absorption": (brut - (massif["brut_base"] if nom == "rachat_massif" else be_brut_base))
                                     - (net - (massif["net_base"] if nom == "rachat_massif" else be_stoch))})
    chocs_euro = pd.DataFrame(lignes).set_index("choc")

    # ---- deltas déterministes correspondants (V1) ------------------------
    def be_det(chocs: Chocs, courbe_alt=None) -> float:
        c = courbe if courbe_alt is None else courbe_alt
        return be_epargne_euro(donnees["mp_epargne_euro"], Actualisation(c), c, tables, hyp, chocs)

    delta_det = {
        "taux_hausse": be_det(Chocs(), courbe_choquee(courbe, params, "hausse")) - be_det_base,
        "taux_baisse": be_det(Chocs(), courbe_choquee(courbe, params, "baisse")) - be_det_base,
        "action": 0.0, "immobilier": 0.0, "spread": 0.0,
        "rachat_hausse": be_det(Chocs(facteur_rachat=1 + params["vie"]["rachat_hausse"])) - be_det_base,
        "rachat_baisse": be_det(Chocs(facteur_rachat=1 - params["vie"]["rachat_baisse"])) - be_det_base,
        "rachat_massif": be_det(Chocs(rachat_massif=params["vie"]["rachat_massif"])) - be_det_base,
        "frais": be_det(Chocs(facteur_frais=1 + params["vie"]["frais_hausse"],
                              inflation_frais_sup=params["vie"]["frais_inflation_sup"])) - be_det_base,
    }

    # ---- reconstruction des sous-modules ---------------------------------
    marche_v1 = v1["marche"].set_index("sous_module")["scr"]
    vie_v1 = v1["vie"].set_index("sous_module")["scr"]
    pertes_taux = moteur_v1.pertes_taux

    def substituer(perte_v1: float, choc: str, regime: str) -> float:
        colonne = "delta_net" if regime == "net" else "delta_brut"
        return max(0.0, perte_v1 - delta_det[choc] + chocs_euro.loc[choc, colonne])

    sous_modules = {}
    for regime in ("net", "brut"):
        taux = {s: substituer(pertes_taux[s], f"taux_{s}", regime) for s in ("hausse", "baisse")}
        sens = max(taux, key=taux.get)
        marche = {"taux": taux[sens],
                  "action": substituer(marche_v1["action"], "action", regime),
                  "immobilier": substituer(marche_v1["immobilier"], "immobilier", regime),
                  "spread": substituer(marche_v1["spread"], "spread", regime),
                  "change": marche_v1["change"], "concentration": marche_v1["concentration"]}
        cm = params["correlations_marche"]
        a = cm["a_hausse" if sens == "hausse" else "a_baisse"]
        b = cm.get("b_hausse" if sens == "hausse" else "b_baisse", a)
        M = matrice_marche(cm["matrice"], a, b)
        v = np.array([marche[nom] for nom in params["correlations_marche"]["ordre"]])
        scr_marche = float(np.sqrt(v @ M @ v))

        rachat_v1 = max(0.0, vie_v1["rachat"])
        rachats = [substituer(rachat_v1, f"rachat_{t}", regime) for t in ("hausse", "baisse", "massif")]
        vie = {"mortalite": vie_v1["mortalite"], "longevite": vie_v1["longevite"], "invalidite": 0.0,
               "rachat": max(rachats), "frais": substituer(vie_v1["frais"], "frais", regime),
               "revision": 0.0, "catastrophe": vie_v1["catastrophe"]}
        Mv = np.array(params["vie"]["correlations"]["matrice"], dtype=float)
        vv = np.array([vie[nom] for nom in params["vie"]["correlations"]["ordre"]])
        scr_vie = float(np.sqrt(vv @ Mv @ vv))
        sous_modules[regime] = {"marche_detail": marche, "vie_detail": vie, "sens_taux": sens,
                                "marche": scr_marche, "vie": scr_vie}

    # ---- agrégation ------------------------------------------------------
    modules_v1 = v1["modules"]
    resultats = {}
    for regime in ("net", "brut"):
        modules = dict(modules_v1)
        modules["marche"] = sous_modules[regime]["marche"]
        modules["vie"] = sous_modules[regime]["vie"]
        resultats[regime] = agregation.bscr(modules, params)

    bscr_net, table_bscr_net = resultats["net"]
    bscr_brut, table_bscr_brut = resultats["brut"]
    be_v2 = dict(be_v1)
    be_v2["epargne_euro"] = be_stoch
    be_v2["total"] = float(be_v1["total"]) - be_det_base + be_stoch
    scr_op, detail_op = agregation.operationnel(hyp, be_v2, bscr_brut, params, donnees)
    lac_tp = -min(fdb, bscr_brut - bscr_net)
    plafond_dt = params["fiscalite"]["taux_impot"] * (bscr_brut + scr_op + lac_tp)
    idp = hyp["bilan_cible"]["passif"]["impots_differes_passifs"]
    capacite = idp + params["lac"]["dt_part_benefices_futurs"] * max(plafond_dt - idp, 0.0)
    lac_dt = -min(plafond_dt, capacite)
    scr_v2 = bscr_brut + scr_op + lac_tp + lac_dt

    bof_v2 = moteur_v1.bof_base - (be_stoch - be_det_base)
    valeur_mcr, table_mcr = agregation.mcr(scr_v2, be_v2, donnees, hyp, params, be_v1["_detail_non_vie"]
                                           if "_detail_non_vie" in be_v1 else v1["moteur"].be_base["_detail_non_vie"])
    table_s25 = agregation.tableau_s25({**modules_v1, "marche": sous_modules["brut"]["marche"],
                                        "vie": sous_modules["brut"]["vie"]},
                                       bscr_brut, scr_op, lac_tp, lac_dt, scr_v2)

    comparaison = pd.DataFrame([
        {"indicateur": "BE épargne euro", "v1": be_det_base, "v2": be_stoch},
        {"indicateur": "BE total", "v1": float(be_v1["total"]), "v2": be_v2["total"]},
        {"indicateur": "Fonds propres de base", "v1": moteur_v1.bof_base, "v2": bof_v2},
        {"indicateur": "SCR marché", "v1": marche_v1["SCR marché (agrégé)"], "v2": sous_modules["net"]["marche"]},
        {"indicateur": "SCR vie", "v1": vie_v1["SCR souscription vie (agrégé)"], "v2": sous_modules["net"]["vie"]},
        {"indicateur": "BSCR", "v1": v1["bscr"].set_index("module").loc["BSCR", "scr"], "v2": bscr_net},
        {"indicateur": "SCR", "v1": v1["scr"], "v2": scr_v2},
        {"indicateur": "MCR", "v1": v1["mcr_valeur"], "v2": valeur_mcr},
        {"indicateur": "Ratio de couverture", "v1": moteur_v1.bof_base / v1["scr"], "v2": bof_v2 / scr_v2},
    ])

    sensibilite = _sensibilite_tmg(alm, cohortes, actif, hyp, scenarios, central)
    sortie = {"martingales": martingales, "chocs_euro": chocs_euro.reset_index(),
              "comparaison": comparaison, "s25_v2": table_s25, "bscr_net": table_bscr_net,
              "bscr_brut": table_bscr_brut, "mcr": table_mcr, "sensibilite_tmg": sensibilite,
              "flux": resultat_base.flux_moyens,
              "be": {"central": be_central, "stochastique": be_stoch, "garanti": be_garanti,
                     "tvog": be_stoch - be_central, "fdb": fdb},
              "scr": scr_v2, "lac_tp": lac_tp, "lac_dt": lac_dt, "ratio": bof_v2 / scr_v2}

    if ecrire:
        dossier = RACINE / "resultats"
        dossier.mkdir(exist_ok=True)
        pd.DataFrame([sortie["be"]]).to_csv(dossier / "v2_be.csv", index=False, sep=";",
                                            decimal=",", encoding="utf-8-sig")
        for nom in ("martingales", "chocs_euro", "comparaison", "s25_v2", "mcr", "sensibilite_tmg", "flux"):
            sortie[nom].to_csv(dossier / f"v2_{nom}.csv", index=False, sep=";", decimal=",",
                               encoding="utf-8-sig")
        _afficher(hyp, sortie, detail_op, time.time() - debut)
    return sortie


def _rachat_massif_selectif(cohortes, actif, hyp, scenarios, part: float) -> dict[str, float]:
    """Rachat massif appliqué cohorte par cohorte, seulement là où il est défavorable (Art. 142)."""
    pm_total = cohortes["pm"].sum()
    ppe_total = hyp["epargne_euro"]["ppe_fdb"]
    resultats = {"net": 0.0, "brut": 0.0, "net_base": 0.0, "brut_base": 0.0}
    for i in range(len(cohortes)):
        coh = cohortes.iloc[[i]].reset_index(drop=True)
        coh.attrs["qx"] = cohortes.attrs["qx"][[i]]
        modele = ModeleALM(coh, actif, hyp)
        ppe = ppe_total * float(coh["pm"].iloc[0]) / pm_total
        for regime in ("dynamique", "figee"):
            cle = "net" if regime == "dynamique" else "brut"
            base = modele.projeter(scenarios, regime, ppe_initiale=ppe).be
            choque = modele.projeter(scenarios, regime, rachat_massif=part, ppe_initiale=ppe).be
            resultats[f"{cle}_base"] += base
            resultats[cle] += max(base, choque)
    return resultats


def _sensibilite_tmg(alm, cohortes, actif, hyp, scenarios, central) -> pd.DataFrame:
    """Valeur temps des options en fonction du niveau des taux garantis."""
    lignes = []
    for bump in (0.0, 0.01, 0.02, 0.03):
        coh = cohortes.copy()
        coh["tmg"] = coh["tmg"] + bump
        coh.attrs["qx"] = cohortes.attrs["qx"]
        modele = ModeleALM(coh, actif, hyp)
        stoch = modele.projeter(scenarios, "dynamique").be
        centr = modele.projeter(central, "dynamique").be
        lignes.append({"decalage_tmg": bump, "be_central": centr, "be_stochastique": stoch,
                       "tvog": stoch - centr})
    return pd.DataFrame(lignes)


def _afficher(hyp, s, detail_op, duree):
    be = s["be"]
    print(f"\n{hyp['general']['nom_compagnie']} — V2 (ALM stochastique, "
          f"{hyp['esg']['nb_scenarios']} scénarios)\n")
    print("Tests de martingale (écarts en points de base)")
    for r in s["martingales"].itertuples():
        print(f"  {r.test:34s} {r.ecart_bp:+8.1f}")
    print(f"\nBE épargne euro — scénario central {be['central']:,.1f} · stochastique "
          f"{be['stochastique']:,.1f} · garanti seul {be['garanti']:,.1f}")
    print(f"TVOG {be['tvog']:+,.1f} M€   ·   prestations discrétionnaires futures (FDB) {be['fdb']:,.1f} M€\n")
    print("Chocs sur le canton euro (M€)")
    print(f"  {'choc':16s} {'Δ BE net':>10s} {'Δ BE brut':>10s} {'absorption':>12s}")
    for r in s["chocs_euro"].itertuples():
        print(f"  {r.choc:16s} {r.delta_net:10,.1f} {r.delta_brut:10,.1f} {r.absorption:12,.1f}")
    print("\nTableau S.25.01 — V2 (M€)")
    for r in s["s25_v2"].itertuples():
        print(f"  {r.code}  {r.poste:52s} {r.montant:10,.1f}")
    print(f"\n  dont risque opérationnel : {detail_op}")
    print("\nComparaison V1 / V2")
    for r in s["comparaison"].itertuples():
        if "Ratio" in r.indicateur:
            print(f"  {r.indicateur:26s} {r.v1:10.0%} {r.v2:10.0%}")
        else:
            print(f"  {r.indicateur:26s} {r.v1:10,.1f} {r.v2:10,.1f}")
    print("\nSensibilité de la TVOG au niveau des TMG")
    for r in s["sensibilite_tmg"].itertuples():
        print(f"  TMG +{r.decalage_tmg:.0%} : central {r.be_central:9,.1f} · stochastique "
              f"{r.be_stochastique:9,.1f} · TVOG {r.tvog:+8.1f}")
    print(f"\nCalcul effectué en {duree:.1f} s\n")


if __name__ == "__main__":
    calculer()
