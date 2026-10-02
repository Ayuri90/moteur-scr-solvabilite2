"""Chiffres du cahier de calculs manuels.

Reprend les données et les paramètres du projet, refait à la main chaque étape intermédiaire
(actualisation d'une obligation, facteur de stress, forme quadratique, formule de variance…)
et écrit `chiffres_pedago.tex`, un fichier de macros utilisé par `cahier_calculs.tex`.

Aucune valeur du cahier n'est saisie à la main : elles proviennent toutes d'ici.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calculer_scr
from scr.marche import courbe_choquee, facteur_stress_spread, prix_obligation
from scr_data.config import RACINE, charger_hypotheses
from scr_data.courbe import Actualisation, construire_courbe

DOSSIER = Path(__file__).resolve().parent
LIGNE_OBLIGATION = "A00045"          # corporate CQS 2, coupon élevé, maturité 8 ans


def pc(valeur: float, decimales: int = 1) -> str:
    """Pourcentage formaté pour LaTeX : le signe % doit être échappé."""
    return f"{valeur:.{decimales}%}".replace("%", "\\%")


def lire(nom: str) -> pd.DataFrame:
    return pd.read_csv(RACINE / "resultats" / nom, sep=";", decimal=",")


def donnees() -> dict[str, pd.DataFrame]:
    hyp = charger_hypotheses()
    return calculer_scr.charger_donnees(RACINE / hyp["general"]["dossier_sortie"])


# ---------------------------------------------------------------------------
def bloc_obligation(inv, actu, courbe, params, macros, tableaux):
    """Prix, duration et chocs de taux et de spread sur une obligation précise."""
    o = inv.set_index("id_ligne").loc[LIGNE_OBLIGATION]
    c, m, s = float(o["coupon"]), int(o["maturite"]), float(o["spread_bp"]) / 1e4
    t = np.arange(1, m + 1)
    df = actu.df(t, s)
    flux = np.full(m, c * 100.0)
    flux[-1] += 100.0
    macros |= {
        "OblId": LIGNE_OBLIGATION, "OblEmetteur": str(o["emetteur"]).replace("_", "\\_"),
        "OblCqs": int(o["cqs"]), "OblCoupon": c * 100, "OblMaturite": m,
        "OblSpread": s * 1e4, "OblNominal": float(o["nominal"]),
        "OblPrix": float(o["prix_pct"]), "OblVM": float(o["valeur_marche"]),
        "OblDuration": float(o["duration_modifiee"]),
        "OblPVCoupons": float(np.sum(c * 100.0 * df)), "OblPVPrincipal": float(100.0 * df[-1]),
        "OblSpotHuit": float(actu.df(m) ** (-1 / m) - 1) * 100,
        "OblTauxActu": float((actu.df(m) ** (-1 / m) - 1) + s) * 100,
    }
    lignes = []
    for k in list(range(1, min(3, m) + 1)) + [m]:
        lignes.append(f"{k} & {flux[k-1]:.3f} & {actu.df(k) ** (-1/k) - 1:.5f} & "
                      f"{df[k-1]:.5f} & {flux[k-1] * df[k-1]:.3f} \\\\")
    tableaux["tableauFlux"] = "\n".join(lignes)

    # choc de spread
    stress = facteur_stress_spread(o["cqs"], float(o["duration_modifiee"]), params, False)
    macros |= {"OblStress": stress * 100, "OblPerteSpread": stress * float(o["valeur_marche"]),
               "OblBandeA": params["spread"]["notes"][int(o["cqs"])]["a"][1] * 100,
               "OblBandeB": params["spread"]["notes"][int(o["cqs"])]["b"][1] * 100}

    # choc de taux à la hausse
    haut = courbe_choquee(courbe, params, "hausse")
    actu_haut = Actualisation(haut)
    prix_haut = prix_obligation(actu_haut, c, m, s)
    macros |= {
        "OblSpotHuitChoc": float(haut.set_index("maturite").loc[m, "taux_spot"]) * 100,
        "OblFacteurChoc": float(haut.set_index("maturite").loc[m, "taux_spot"]
                                / courbe.set_index("maturite").loc[m, "taux_spot"] - 1) * 100,
        "OblPrixChoc": prix_haut,
        "OblVMChoc": float(o["nominal"]) * prix_haut / 100,
        "OblPerteTaux": float(o["valeur_marche"]) - float(o["nominal"]) * prix_haut / 100,
        "OblPerteApprochee": float(o["valeur_marche"]) * float(o["duration_modifiee"])
                             * (float(haut.set_index("maturite").loc[m, "taux_spot"])
                                - float(courbe.set_index("maturite").loc[m, "taux_spot"])),
    }


def bloc_action_immobilier_change(inv, params, macros, tableaux):
    euro = inv[inv["portefeuille"] == "euro"]
    p = params["action"]
    expositions = {
        "type 1 (actions cotées)": (float(euro.loc[euro["type_action"] == 1, "valeur_marche"].sum()),
                                    p["choc_type1"] + p["ajustement_symetrique"]),
        "type 2 (non coté)": (float(euro.loc[euro["type_action"] == 2, "valeur_marche"].sum()),
                              p["choc_type2"] + p["ajustement_symetrique"]),
        "participations stratégiques": (float(euro.loc[euro["type_action"] == 3, "valeur_marche"].sum()),
                                        p["choc_strategique"]),
    }
    lignes = [f"{nom} & {vm:,.1f} & {pc(choc, 0)} & {vm * choc:,.1f} \\\\".replace(",", "\\,")
              for nom, (vm, choc) in expositions.items()]
    tableaux["tableauAction"] = "\n".join(lignes)
    perte_un = expositions["type 1 (actions cotées)"][0] * expositions["type 1 (actions cotées)"][1] \
        + expositions["participations stratégiques"][0] * expositions["participations stratégiques"][1]
    perte_deux = expositions["type 2 (non coté)"][0] * expositions["type 2 (non coté)"][1]
    rho = p["correlation_type1_type2"]
    macros |= {
        "ActionUn": perte_un, "ActionDeux": perte_deux, "ActionRho": rho,
        "ActionAgrege": float(np.sqrt(perte_un**2 + 2 * rho * perte_un * perte_deux + perte_deux**2)),
        "ImmoAssiette": float(euro.loc[euro["classe"] == "immobilier", "valeur_marche"].sum()),
        "ImmoChoc": params["immobilier"]["choc"] * 100,
    }
    macros["ImmoPerte"] = macros["ImmoAssiette"] * params["immobilier"]["choc"]

    devises = (inv[inv["devise"] != "EUR"].groupby("devise")["valeur_marche"].sum()
               .sort_values(ascending=False))
    choc = params["change"]["choc_hausse"]
    lignes = [f"{d} & {vm:,.1f} & {vm * choc:,.1f} \\\\".replace(",", "\\,")
              for d, vm in devises.items()]
    tableaux["tableauChange"] = "\n".join(lignes)
    macros |= {"ChangeTotal": float(devises.sum()), "ChangeChoc": choc * 100,
               "ChangePerte": float(devises.sum() * choc)}


def bloc_concentration(inv, params, macros, tableaux):
    euro = inv[inv["portefeuille"] == "euro"]
    actifs = float(euro["valeur_marche"].sum())
    base = euro[euro["sous_classe"] != "souverain"]
    expo = base.groupby("groupe_emetteur").agg(vm=("valeur_marche", "sum"), cqs=("cqs", "min"))
    p = params["concentration"]
    def parametres_groupe(cqs):
        """Seuil et facteur : repli sur le traitement des non notés si l'échelon manque."""
        if pd.isna(cqs):
            return None, p["seuil_non_note"], p["facteur_non_note"]
        return int(cqs), p["seuil_par_cqs"][int(cqs)], p["facteur_par_cqs"][int(cqs)]

    lignes, carres = [], 0.0
    for groupe, r in expo.nlargest(4, "vm").iterrows():
        cqs, seuil, g = parametres_groupe(r["cqs"])
        xs = max(0.0, r["vm"] / actifs - seuil)
        conc = actifs * xs * g
        lignes.append(f"{groupe.replace('_', '\\_')} & {r['vm']:,.1f} & {pc(r['vm'] / actifs, 2)} & "
                      f"{'non noté' if cqs is None else cqs} & {pc(seuil, 1)} & {xs:.4f} & "
                      f"{conc:,.1f} \\\\".replace(",", "\\,"))
    for groupe, r in expo.iterrows():
        _, seuil, g = parametres_groupe(r["cqs"])
        xs = max(0.0, r["vm"] / actifs - seuil)
        carres += (actifs * xs * g) ** 2
    tableaux["tableauConcentration"] = "\n".join(lignes)
    premier = expo.nlargest(1, "vm").iloc[0]
    _, seuil_premier, g_premier = parametres_groupe(premier["cqs"])
    macros |= {
        "ConcActifs": actifs, "ConcEmetteur": expo.nlargest(1, "vm").index[0].replace("_", "\\_"),
        "ConcExpo": float(premier["vm"]), "ConcPart": float(premier["vm"]) / actifs * 100,
        "ConcSeuil": seuil_premier * 100, "ConcFacteur": g_premier * 100,
        "ConcXS": float(premier["vm"]) / actifs - seuil_premier,
        "ConcPremier": actifs * (float(premier["vm"]) / actifs - seuil_premier) * g_premier,
        "ConcTotal": float(np.sqrt(carres)),
    }


def bloc_agregation_marche(params, macros, tableaux):
    table = lire("scr_marche.csv").set_index("sous_module")["scr"]
    ordre = params["correlations_marche"]["ordre"]
    a = params["correlations_marche"]["a_hausse"]
    M = np.array([[a if c == "A" else float(c) for c in ligne]
                  for ligne in params["correlations_marche"]["matrice"]])
    v = np.array([float(table[nom]) for nom in ordre])
    lignes = [" & ".join([ordre[i].replace("_", " ")] + [f"{M[i, j]:.2f}" for j in range(len(ordre))]
                         + [f"{v[i]:,.1f}"]).replace(",", "\\,") + " \\\\"
              for i in range(len(ordre))]
    tableaux["tableauMatriceMarche"] = "\n".join(lignes)
    macros |= {"MarcheSomme": float(v.sum()), "MarcheQuadratique": float(v @ M @ v),
               "MarcheAgrege": float(np.sqrt(v @ M @ v)),
               "MarcheDiversification": float(v.sum() - np.sqrt(v @ M @ v))}


def bloc_contrepartie(cp, params, macros, tableaux):
    p = params["contrepartie"]
    type1 = cp[cp["type_exposition"].str.startswith("type1")].copy()
    type1["lgd"] = np.where(type1["type_exposition"] == "type1_reassurance",
                            p["lgd_reassurance"] * type1["montant"], type1["montant"])
    type1["pd"] = type1["cqs"].map(lambda c: p["pd_par_cqs"][int(c)])
    lignes = [f"{r.contrepartie.replace('_', '\\_')} & {int(r.cqs)} & {r.montant:,.1f} & "
              f"{r.lgd:,.1f} & {pc(r.pd, 4)} \\\\".replace(",", "\\,") for r in type1.itertuples()]
    tableaux["tableauContrepartie"] = "\n".join(lignes)

    groupes = type1.groupby("pd")["lgd"].agg(["sum", lambda s: float((s**2).sum())])
    groupes.columns = ["tlgd", "lgd2"]
    V = 0.0
    for pd_j, rj in groupes.iterrows():
        for pd_k, rk in groupes.iterrows():
            V += (pd_j * pd_k) / (1.25 * (pd_j + pd_k) - pd_j * pd_k) * rj["tlgd"] * rk["tlgd"]
    diagonale = sum((1.5 * pd_j * (1 - pd_j)) / (2.5 - pd_j) * rj["lgd2"]
                    for pd_j, rj in groupes.iterrows())
    V += diagonale
    total = float(type1["lgd"].sum())
    racine = float(np.sqrt(V))
    scr1 = 3 * racine if racine <= 0.05 * total else (5 * racine if racine <= 0.20 * total else total)
    echues = float(cp.loc[cp["type_exposition"] == "type2_creances_echues_3m", "montant"].sum())
    autres = float(cp.loc[cp["type_exposition"] == "type2_creances", "montant"].sum())
    scr2 = p["type2_facteur_standard"] * autres + p["type2_facteur_echues_3m"] * echues
    macros |= {
        "CpTLGDun": float(groupes["tlgd"].iloc[0]), "CpTLGDdeux": float(groupes["tlgd"].iloc[1]),
        "CpPDun": float(groupes.index[0]) * 100, "CpPDdeux": float(groupes.index[1]) * 100,
        "CpDiagonale": diagonale, "CpVariance": V, "CpRacine": racine, "CpLGDtotal": total,
        "CpSeuil": 0.05 * total, "CpUn": scr1, "CpCreances": autres, "CpEchues": echues,
        "CpDeux": scr2,
        "CpAgrege": float(np.sqrt(scr1**2 + 1.5 * scr1 * scr2 + scr2**2)),
    }


def bloc_vie(donnees_, hyp, params, macros, tableaux):
    td = donnees_["mp_temporaire_deces"]
    capitaux = float(td["capitaux_sous_risque"].sum())
    macros |= {
        "VieCapitaux": capitaux, "VieCatChoc": params["vie"]["cat_mortalite_absolu"] * 100,
        "VieCatPerte": capitaux * params["vie"]["cat_mortalite_absolu"],
        "VieCatModele": float(lire("scr_vie.csv").set_index("sous_module").loc["catastrophe", "scr"]),
        "VieMortaliteChoc": params["vie"]["mortalite"] * 100,
        "VieMortaliteModele": float(lire("scr_vie.csv").set_index("sous_module")
                                    .loc["mortalite", "scr"]),
        "VieMassifTaux": params["vie"]["rachat_massif"] * 100,
        "VieMassifModele": float(lire("scr_vie.csv").set_index("sous_module").loc["rachat", "scr"]),
    }
    table = lire("scr_vie.csv").set_index("sous_module")["scr"]
    ordre = params["vie"]["correlations"]["ordre"]
    M = np.array(params["vie"]["correlations"]["matrice"], dtype=float)
    v = np.array([float(table[nom]) for nom in ordre])
    lignes = [" & ".join([ordre[i].replace("_", " ")] + [f"{M[i, j]:.2f}" for j in range(len(ordre))]
                         + [f"{v[i]:,.1f}"]).replace(",", "\\,") + " \\\\" for i in range(len(ordre))]
    tableaux["tableauMatriceVie"] = "\n".join(lignes)
    macros |= {"VieSomme": float(v.sum()), "VieAgrege": float(np.sqrt(v @ M @ v))}


def bloc_non_vie(donnees_, hyp, params, macros, tableaux):
    detail = lire("scr_non_vie_detail.csv").query("bloc == 'primes_reserves'").set_index("lob")
    sigma = params["non_vie"]["sigma"]
    lignes = []
    for lob in params["non_vie"]["correlations_lob"]["ordre"]:
        r = detail.loc[lob]
        lignes.append(f"{lob.replace('_', ' ')} & {r['v_primes']:,.1f} & {r['v_reserves']:,.1f} & "
                      f"{pc(sigma[lob]['primes'], 1)} & {pc(sigma[lob]['reserves'], 1)} & "
                      f"{pc(r['sigma'], 2)} \\\\".replace(",", "\\,"))
    tableaux["tableauNonVie"] = "\n".join(lignes)
    lob = "auto_rc"
    r = detail.loc[lob]
    sp, sr = sigma[lob]["primes"], sigma[lob]["reserves"]
    vp, vr = float(r["v_primes"]), float(r["v_reserves"])
    macros |= {
        "NvLob": lob.replace("_", " "), "NvVp": vp, "NvVr": vr, "NvSigmaP": sp * 100,
        "NvSigmaR": sr * 100, "NvNumerateur": float(np.sqrt((sp * vp) ** 2 + sp * vp * sr * vr
                                                            + (sr * vr) ** 2)),
        "NvSigmaLob": float(r["sigma"]) * 100, "NvVolumeTotal": float(detail["volume"].sum()),
        "NvSigmaGlobal": float(lire("scr_non_vie.csv").set_index("sous_module")
                               .loc["primes et réserves", "scr"]) / 3 / float(detail["volume"].sum()) * 100,
        "NvPrimesReserves": float(lire("scr_non_vie.csv").set_index("sous_module")
                                  .loc["primes et réserves", "scr"]),
        "NvCatBrut": float(lire("scr_non_vie_detail.csv").set_index("composante")
                           .loc["CAT brut de réassurance", "scr_brut"]),
        "NvCatNet": float(lire("scr_non_vie_detail.csv").set_index("composante")
                          .loc["CAT net de réassurance", "scr_brut"]),
        "NvPriorite": hyp["non_vie"]["reassurance"]["xl_cat_priorite"],
        "NvPortee": hyp["non_vie"]["reassurance"]["xl_cat_portee"],
        "NvAgrege": float(lire("scr_non_vie.csv").set_index("sous_module")
                          .loc["SCR souscription non-vie (agrégé)", "scr"]),
    }


def bloc_final(donnees_, hyp, params, macros, tableaux):
    be = lire("be_par_segment.csv").set_index("segment")["be"]
    s25 = lire("s25_01.csv").set_index("poste")["montant"]
    modules = {"marche": "Risque de marché", "defaut": "Risque de défaut de la contrepartie",
               "vie": "Risque de souscription vie", "sante": "Risque de souscription santé",
               "non_vie": "Risque de souscription non-vie"}
    ordre = params["bscr"]["ordre"]
    M = np.array(params["bscr"]["matrice"], dtype=float)
    v = np.array([float(s25[modules[m]]) for m in ordre])
    lignes = [" & ".join([ordre[i].replace("_", " ")] + [f"{M[i, j]:.2f}" for j in range(len(ordre))]
                         + [f"{v[i]:,.1f}"]).replace(",", "\\,") + " \\\\" for i in range(len(ordre))]
    tableaux["tableauMatriceBscr"] = "\n".join(lignes)

    op = params["operationnel"]
    primes_vie = hyp["epargne_euro"]["collecte_annuelle"] + float(
        donnees_["mp_temporaire_deces"]["prime_annuelle"].sum())
    primes_non_vie = (sum(l["primes_n"] for l in hyp["non_vie"]["lignes"].values())
                      + hyp["sante_nslt"]["primes_n"] + hyp["sante_slt"]["primes_annuelles"])
    pt_vie = float(be["epargne_euro"] + be["temporaire_deces"] + be["rentes"] + be["sante_slt"])
    pt_non_vie = float(be["non_vie_sinistres"] + be["non_vie_primes"] + be["sante_nslt"])
    op_primes = op["facteur_primes_vie"] * primes_vie + op["facteur_primes_non_vie"] * primes_non_vie
    op_prov = op["facteur_provisions_vie"] * pt_vie + op["facteur_provisions_non_vie"] * pt_non_vie
    bscr = float(s25["Capital de solvabilité requis de base (BSCR)"])
    frais_uc = hyp["frais"]["gestion_pm_uc"] * hyp["epargne_uc"]["pm_totale"]
    macros |= {
        "BscrSomme": float(v.sum()), "BscrAgrege": bscr,
        "BscrDiversification": float(v.sum()) - bscr,
        "OpPrimesVie": primes_vie, "OpPrimesNonVie": primes_non_vie,
        "OpPrimes": op_primes, "OpProvVie": pt_vie, "OpProvNonVie": pt_non_vie,
        "OpProvisions": op_prov, "OpPlafond": op["plafond_bscr"] * bscr,
        "OpFraisUC": frais_uc, "OpTotal": float(s25["Risque opérationnel"]),
        "LacPlafond": params["fiscalite"]["taux_impot"]
                      * (bscr + float(s25["Risque opérationnel"])),
        "LacIdp": hyp["bilan_cible"]["passif"]["impots_differes_passifs"],
        "LacPart": params["lac"]["dt_part_benefices_futurs"] * 100,
        "LacDt": float(s25["Capacité d'absorption des impôts différés"]),
        "ScrFinal": float(s25["Capital de solvabilité requis (SCR)"]),
    }
    mcr = lire("mcr.csv").set_index("poste")["montant"]
    resume = lire("resume.csv").set_index("poste")["montant"]
    macros |= {
        "McrVie": float(mcr["MCR linéaire vie"]), "McrNonVie": float(mcr["MCR linéaire non-vie et santé"]),
        "McrLineaire": float(mcr["MCR linéaire total"]), "McrPlancher": float(mcr["Plancher (25 % du SCR)"]),
        "McrPlafond": float(mcr["Plafond (45 % du SCR)"]), "McrRetenu": float(mcr["MCR retenu"]),
        "FondsPropres": float(resume["Fonds propres de base (BOF)"]),
        "RatioScr": float(resume["Ratio de couverture du SCR"]) * 100,
        "RatioMcr": float(resume["Ratio de couverture du MCR"]) * 100,
    }


# ---------------------------------------------------------------------------
def ecrire(macros: dict, tableaux: dict) -> None:
    lignes = ["% Fichier engendré par pedagogie/generer_chiffres.py — ne pas éditer"]
    for nom, valeur in macros.items():
        if isinstance(valeur, str):
            lignes.append(f"\\newcommand{{\\{nom}}}{{{valeur}}}")
        elif isinstance(valeur, (int, np.integer)):
            lignes.append(f"\\newcommand{{\\{nom}}}{{\\num{{{valeur}}}}}")
        else:
            decimales = 4 if abs(valeur) < 0.01 else (3 if abs(valeur) < 1 else 2)
            lignes.append(f"\\newcommand{{\\{nom}}}{{\\num{{{valeur:.{decimales}f}}}}}")
    for nom, corps in tableaux.items():
        lignes.append(f"\\newcommand{{\\{nom}}}{{%\n{corps}\n}}")
    (DOSSIER / "chiffres_pedago.tex").write_text("\n".join(lignes) + "\n", encoding="utf-8")


def principal() -> None:
    hyp = charger_hypotheses()
    params = calculer_scr.charger_params()
    d = donnees()
    courbe, _ = construire_courbe(hyp)
    actu = Actualisation(courbe)
    macros, tableaux = {}, {}
    bloc_obligation(d["inventaire_actifs"], actu, courbe, params, macros, tableaux)
    bloc_action_immobilier_change(d["inventaire_actifs"], params, macros, tableaux)
    bloc_concentration(d["inventaire_actifs"], params, macros, tableaux)
    bloc_agregation_marche(params, macros, tableaux)
    bloc_contrepartie(d["contreparties"], params, macros, tableaux)
    bloc_vie(d, hyp, params, macros, tableaux)
    bloc_non_vie(d, hyp, params, macros, tableaux)
    bloc_final(d, hyp, params, macros, tableaux)
    ecrire(macros, tableaux)
    print(f"{len(macros)} macros et {len(tableaux)} tableaux écrits dans chiffres_pedago.tex")


if __name__ == "__main__":
    principal()
