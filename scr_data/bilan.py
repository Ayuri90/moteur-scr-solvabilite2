"""Contreparties, bilan prudentiel indicatif (format proche du S.02.01) et contrôles de calibrage."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
def generer_contreparties(hyp: dict[str, Any]) -> pd.DataFrame:
    hc = hyp["contreparties"]
    cedees = hyp["bilan_cible"]["actif"]["provisions_cedees"]
    lignes = [{"contrepartie": nom, "type_exposition": "type1_reassurance", "cqs": p["cqs"],
               "montant": p["part"] * cedees, "commentaire": "part des provisions cédées"}
              for nom, p in hc["reassureurs"].items()]
    for nom, p in hyp["monetaire"]["banques"].items():
        lignes.append({"contrepartie": nom, "type_exposition": "type1_tresorerie", "cqs": p["cqs"],
                       "montant": p["poids"] * hc["creances"]["tresorerie_banques"],
                       "commentaire": "comptes courants (hors dépôts de placement)"})
    cr = hc["creances"]
    total = cr["creances_assures_intermediaires"]
    lignes += [
        {"contrepartie": "ASSURES_INTERMEDIAIRES", "type_exposition": "type2_creances", "cqs": None,
         "montant": total * (1 - cr["part_echues_plus_3_mois"]), "commentaire": "échues depuis moins de 3 mois"},
        {"contrepartie": "ASSURES_INTERMEDIAIRES", "type_exposition": "type2_creances_echues_3m", "cqs": None,
         "montant": total * cr["part_echues_plus_3_mois"], "commentaire": "échues depuis plus de 3 mois"},
    ]
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
def construire_bilan(hyp, inventaire, passif_vie, non_vie, sante_slt) -> pd.DataFrame:
    euro = inventaire[inventaire["portefeuille"] == "euro"]
    vm = lambda masque: float(euro.loc[masque, "valeur_marche"].sum())  # noqa: E731
    cr = hyp["contreparties"]["creances"]
    bc = hyp["bilan_cible"]

    actif = [
        ("R0060", "Immobilisations corporelles pour usage propre", vm(euro["usage_propre"]), "inventaire"),
        ("R0080", "Immobilier (autre que pour usage propre)",
         vm((euro["classe"] == "immobilier") & ~euro["usage_propre"]), "inventaire"),
        ("R0090", "Participations", vm(euro["sous_classe"] == "participation_strategique"), "inventaire"),
        ("R0100", "Actions (cotées et non cotées)",
         vm(euro["sous_classe"].isin(["action_cotee", "private_equity"])), "inventaire"),
        ("R0140", "Obligations d'État", vm(euro["sous_classe"] == "souverain"), "inventaire"),
        ("R0150", "Obligations d'entreprises (y c. covered bonds)",
         vm(euro["sous_classe"].isin(["corporate", "covered_bond"])), "inventaire"),
        ("R0180", "OPC monétaires (mis en transparence)", vm(euro["classe"] == "monetaire"), "inventaire"),
        ("R0200", "Dépôts autres que les équivalents de trésorerie", vm(euro["classe"] == "depot"), "inventaire"),
        ("R0230", "Prêts et prêts hypothécaires / infrastructure", vm(euro["classe"] == "pret_infrastructure"),
         "inventaire"),
        ("R0220", "Actifs en représentation de contrats en UC",
         float(inventaire.loc[inventaire["portefeuille"] == "uc", "valeur_marche"].sum()), "inventaire"),
        ("R0270", "Montants recouvrables au titre de la réassurance", bc["actif"]["provisions_cedees"], "hypothèse"),
        ("R0360", "Créances nées d'opérations d'assurance", cr["creances_assures_intermediaires"], "hypothèse"),
        ("R0410", "Trésorerie et équivalents de trésorerie", cr["tresorerie_banques"], "hypothèse"),
        ("R0420", "Autres actifs", cr["autres_actifs"], "hypothèse"),
    ]

    reserves = non_vie["volumes_primes_reserves"].set_index("lob")["reserve_non_actualisee_vraie"]
    be_nv = float(reserves.drop("sante_frais_soins").sum())
    be_nslt = float(reserves["sante_frais_soins"])
    be_slt = float(sante_slt["be_indicatif"].sum())
    be_rentes = float(passif_vie["mp_rentes_viageres"]["be_indicatif"].sum())
    rm = bc["passif"]["marge_risque"]
    passif = [
        ("R0520", "PT non-vie (hors santé) — BE (réserves non actualisées, indicatif)", be_nv, "indicatif"),
        ("R0560", "PT santé NSLT — BE (indicatif)", be_nslt, "indicatif"),
        ("R0640", "PT santé SLT — BE (calé sur model points)", be_slt, "indicatif"),
        ("R0690a", "PT vie hors UC — BE épargne euro", bc["passif"]["be_epargne_euro"], "cible à recalculer"),
        ("R0690b", "PT vie hors UC — BE rentes viagères (calé sur model points)", be_rentes, "indicatif"),
        ("R0690c", "PT vie hors UC — BE temporaire décès",
         bc["passif"]["be_prevoyance_rentes"] - be_rentes, "cible à recalculer"),
        ("R0700", "PT UC — BE", bc["passif"]["be_uc"], "cible à recalculer"),
        ("R0590", "Marge de risque (toutes lignes)", rm, "cible à recalculer"),
        ("R0780", "Passifs d'impôts différés", bc["passif"]["impots_differes_passifs"], "cible à recalculer"),
        ("R0900", "Autres passifs", bc["passif"]["autres_passifs"], "hypothèse"),
    ]
    df_a = pd.DataFrame(actif, columns=["code", "poste", "montant", "statut"]).assign(cote="Actif")
    df_p = pd.DataFrame(passif, columns=["code", "poste", "montant", "statut"]).assign(cote="Passif")
    return pd.concat([df_a, df_p], ignore_index=True)[["cote", "code", "poste", "montant", "statut"]]


# ---------------------------------------------------------------------------
def controles(hyp, inventaire, passif_vie, non_vie, sante_slt) -> pd.DataFrame:
    res = []

    def ajouter(nom, valeur, cible=None, tolerance=0.005, info=False):
        if info or cible is None:
            statut, ecart = "INFO", np.nan
        else:
            ecart = valeur / cible - 1 if cible else np.nan
            statut = "OK" if abs(ecart) <= tolerance else "ALERTE"
        res.append({"controle": nom, "valeur": valeur, "cible": cible, "ecart_relatif": ecart, "statut": statut})

    euro = inventaire[inventaire["portefeuille"] == "euro"]
    P = hyp["bilan_cible"]["actif"]["placements_hors_uc"]
    ajouter("Placements hors UC (M€)", euro["valeur_marche"].sum(), P)
    ajouter("Actifs UC (M€)", inventaire.loc[inventaire["portefeuille"] == "uc", "valeur_marche"].sum(),
            hyp["bilan_cible"]["actif"]["actifs_uc"])

    correspondance = {
        "souverains": euro["sous_classe"] == "souverain",
        "corporate_financieres": euro["secteur"].eq("financier") & euro["classe"].eq("obligation"),
        "corporate_non_financieres": euro["secteur"].eq("non_financier") & euro["classe"].eq("obligation"),
        "actions_type1": euro["sous_classe"] == "action_cotee",
        "actions_type2": euro["sous_classe"] == "private_equity",
        "participations_strategiques": euro["sous_classe"] == "participation_strategique",
        "immobilier": euro["classe"] == "immobilier",
        "monetaire_depots": euro["classe"].isin(["monetaire", "depot"]),
        "prets_infrastructure": euro["classe"] == "pret_infrastructure",
    }
    for cle, masque in correspondance.items():
        ajouter(f"Poids {cle}", euro.loc[masque, "valeur_marche"].sum() / P, hyp["allocation"][cle], 0.001)

    ajouter("PM épargne euro (M€)", passif_vie["mp_epargne_euro"]["pm"].sum(), hyp["epargne_euro"]["pm_totale"])
    ajouter("PM UC = actifs UC (M€)", passif_vie["mp_epargne_uc"]["pm"].sum(), hyp["epargne_uc"]["pm_totale"])
    ajouter("Primes temporaire décès (M€)", passif_vie["mp_temporaire_deces"]["prime_annuelle"].sum(),
            hyp["temporaire_deces"]["primes_annuelles"])
    ajouter("BE rentes viagères (M€)", passif_vie["mp_rentes_viageres"]["be_indicatif"].sum(),
            hyp["rentes_viageres"]["be_cible"])
    ajouter("BE santé SLT (M€)", sante_slt["be_indicatif"].sum(), hyp["sante_slt"]["be_cible"])

    for nom, f in non_vie["calages_triangles"].items():
        ajouter(f"Facteur de calage ultimes {nom}", f, 1.0, 0.15)

    obl = euro[euro["duration_modifiee"].notna()]
    ajouter("Duration modifiée moyenne actif euro", np.average(obl["duration_modifiee"], weights=obl["valeur_marche"]),
            info=True)
    notes = euro.dropna(subset=["cqs"])
    ajouter("CQS moyen pondéré (titres notés)", np.average(notes["cqs"].astype(float), weights=notes["valeur_marche"]),
            info=True)
    hors_etats = euro[~euro["groupe_emetteur"].str.startswith("ETAT_")]
    expo = hors_etats.groupby("groupe_emetteur")["valeur_marche"].sum()
    ajouter(f"Plus grande exposition émetteur hors États ({expo.idxmax()}) en % des placements",
            expo.max() / P, info=True)
    ajouter("Part devises hors EUR (actif euro)", euro.loc[euro["devise"] != "EUR", "valeur_marche"].sum() / P,
            info=True)
    pvl = euro["valeur_marche"].sum() - euro["valeur_comptable"].sum()
    ajouter("Plus ou moins-values latentes actif euro (M€)", pvl, info=True)
    return pd.DataFrame(res)
