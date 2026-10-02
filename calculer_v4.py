"""V4 — modèle interne partiel non-vie, USP et comparaisons.

Compare quatre mesures du risque de souscription non-vie :
1. formule standard (écarts types de l'annexe II) ;
2. formule standard avec paramètres propres à l'entreprise (USP) sur le risque de réserve ;
3. modèle interne partiel, copule gaussienne ;
4. modèle interne partiel, copule de Student (queues plus épaisses).

Puis mesure l'effet de chaque variante sur le SCR global et le ratio de couverture,
et rapproche le résultat de la marge de solvabilité du code CIMA.

Usage :
    python calculer_v4.py
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import calculer_scr
import calculer_v2
from scr import agregation
from scr.modele_interne import (ajuster_sp, bootstrap_odp, copule, coupler, mesures_risque,
                                simuler_catastrophe, simuler_primes, usp_reserve)
from scr.souscription import agreger
from scr_data.config import RACINE, charger_hypotheses


def _triangles(donnees) -> dict[str, pd.DataFrame]:
    tri = donnees["triangles_reglements"]
    return {lob: sous.pivot(index="annee_survenance", columns="developpement",
                            values="reglements_cumules")
            for lob, sous in tri.groupby("lob")}


def calculer(ecrire: bool = True) -> dict:
    debut = time.time()
    hyp = charger_hypotheses()
    params = calculer_scr.charger_params()
    donnees = calculer_scr.charger_donnees(RACINE / hyp["general"]["dossier_sortie"])
    p_mi = hyp["modele_interne"]
    rng = np.random.default_rng(p_mi["graine"])
    n_sim = p_mi["nb_simulations"]

    triangles = _triangles(donnees)
    volumes = donnees["volumes_primes_reserves"].set_index("lob")
    lobs = hyp["modele_interne"]["correlations_lob"]["ordre"]

    # ---- marginales par ligne d'activité ---------------------------------
    reserves, primes, diagnostics = {}, {}, []
    for lob in lobs:
        historique = donnees["ultimes_vrais"].query("lob == @lob").set_index("annee_survenance")
        boot = bootstrap_odp(triangles[lob], p_mi["bootstrap"]["nb_reechantillons"], rng,
                             p_mi["bootstrap"]["ajustement_biais"])
        reserves[lob] = boot["reserves"]
        mu, sigma = ajuster_sp(triangles[lob], historique["primes_acquises"])
        primes[lob] = simuler_primes(mu, sigma, volumes.loc[lob, "primes_acquises_n_plus_1_estimees"],
                                     n_sim, rng, p_mi["primes"]["incertitude_parametre"])
        diagnostics.append({
            "lob": lob, "reserve_chain_ladder": boot["reserve_centrale"],
            "reserve_moyenne_bootstrap": float(boot["reserves"].mean()),
            "cv_reserves": float(boot["reserves"].std(ddof=1) / boot["reserves"].mean()),
            "sp_median": float(np.exp(mu)), "sigma_log_sp": sigma,
            "cv_primes": float(primes[lob].std(ddof=1) / primes[lob].mean()),
            "phi_odp": boot["phi"]})
    diagnostics = pd.DataFrame(diagnostics)

    # ---- agrégation entre lignes d'activité ------------------------------
    M_lob = np.array(hyp["modele_interne"]["correlations_lob"]["matrice"], dtype=float)
    u_lob = copule(M_lob, n_sim, rng, p_mi["copule"], p_mi["degres_liberte"])
    total_reserves = coupler([reserves[l] for l in lobs], u_lob).sum(axis=1)
    u_lob2 = copule(M_lob, n_sim, rng, p_mi["copule"], p_mi["degres_liberte"])
    total_primes = coupler([primes[l] for l in lobs], u_lob2).sum(axis=1)

    # ---- catastrophe ------------------------------------------------------
    reass = hyp["non_vie"]["reassurance"]
    cat_brut = simuler_catastrophe(p_mi["catastrophe"], n_sim, rng)
    cat_net = simuler_catastrophe(p_mi["catastrophe"], n_sim, rng,
                                  priorite=reass["xl_cat_priorite"], portee=reass["xl_cat_portee"])

    # ---- agrégation des trois blocs --------------------------------------
    M_blocs = np.array(p_mi["correlations"]["matrice"], dtype=float)
    resultats_copules = {}
    for famille in ("gaussienne", "student"):
        u = copule(M_blocs, n_sim, rng, famille, p_mi["degres_liberte"])
        total = coupler([total_primes, total_reserves, cat_net], u).sum(axis=1)
        resultats_copules[famille] = total

    mesures = {f"modele_interne_{f}": mesures_risque(t, p_mi["quantile"])
               for f, t in resultats_copules.items()}
    mesures["primes"] = mesures_risque(total_primes, p_mi["quantile"])
    mesures["reserves"] = mesures_risque(total_reserves, p_mi["quantile"])
    mesures["catastrophe_brute"] = mesures_risque(cat_brut, p_mi["quantile"])
    mesures["catastrophe_nette"] = mesures_risque(cat_net, p_mi["quantile"])

    # ---- USP sur le risque de réserve ------------------------------------
    credibilite = {int(k): v for k, v in hyp["usp"]["credibilite"].items()}
    lignes_usp = []
    params_usp = calculer_scr.charger_params()
    for lob in lobs:
        standard = params["non_vie"]["sigma"][lob]["reserves"]
        u = usp_reserve(reserves[lob], standard, hyp["non_vie"]["nb_annees"], credibilite)
        params_usp["non_vie"]["sigma"][lob]["reserves"] = u["sigma_retenu"]
        lignes_usp.append({"lob": lob, **u})
    table_usp = pd.DataFrame(lignes_usp)

    # ---- comparaison des mesures du module non-vie -----------------------
    v1 = calculer_scr.calculer(ecrire=False)
    v1_usp = calculer_scr.calculer(ecrire=False, params=params_usp)
    v2 = calculer_v2.calculer(ecrire=False)
    non_vie_fs = float(v1["non_vie"].set_index("sous_module")
                       .loc["SCR souscription non-vie (agrégé)", "scr"])
    non_vie_usp = float(v1_usp["non_vie"].set_index("sous_module")
                        .loc["SCR souscription non-vie (agrégé)", "scr"])
    variantes = {
        "Formule standard": non_vie_fs,
        "Formule standard avec USP": non_vie_usp,
        "Modèle interne (copule gaussienne)": mesures["modele_interne_gaussienne"]["scr"],
        "Modèle interne (copule de Student)": mesures["modele_interne_student"]["scr"],
    }
    comparaison = pd.DataFrame([
        {"mesure": nom, "scr_non_vie": valeur, **_recalculer_global(v2, valeur, hyp, params)}
        for nom, valeur in variantes.items()])

    cima = _marge_cima(hyp, donnees, v1, v2)
    figure = _tracer(resultats_copules, variantes, p_mi["quantile"]) if ecrire else None

    sortie = {"diagnostics": diagnostics, "usp": table_usp, "comparaison": comparaison,
              "mesures": pd.DataFrame(mesures).T.reset_index(names="bloc"), "cima": cima}
    if ecrire:
        dossier = RACINE / "resultats"
        dossier.mkdir(exist_ok=True)
        for nom in ("diagnostics", "usp", "comparaison", "mesures", "cima"):
            sortie[nom].to_csv(dossier / f"v4_{nom}.csv", index=False, sep=";", decimal=",",
                               encoding="utf-8-sig")
        figure.savefig(dossier / "v4_distribution_non_vie.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
        _afficher(sortie, mesures, time.time() - debut)
    return sortie


def _recalculer_global(v2, scr_non_vie: float, hyp, params) -> dict[str, float]:
    """Remplace le module non-vie dans l'agrégation et recalcule SCR, ratio et MCR."""
    s25 = v2["s25_v2"].set_index("poste")["montant"]
    op = float(s25["Risque opérationnel"])
    modules_net = v2["bscr_net"].set_index("module")["scr"].to_dict()
    modules_brut = v2["bscr_brut"].set_index("module")["scr"].to_dict()
    for d in (modules_net, modules_brut):
        d.pop("BSCR", None)
        d["non_vie"] = scr_non_vie
    bscr_net, _ = agregation.bscr(modules_net, params)
    bscr_brut, _ = agregation.bscr(modules_brut, params)
    lac_tp = -min(v2["be"]["fdb"], bscr_brut - bscr_net)
    plafond = params["fiscalite"]["taux_impot"] * (bscr_brut + op + lac_tp)
    idp = hyp["bilan_cible"]["passif"]["impots_differes_passifs"]
    capacite = idp + params["lac"]["dt_part_benefices_futurs"] * max(plafond - idp, 0.0)
    lac_dt = -min(plafond, capacite)
    scr = bscr_brut + op + lac_tp + lac_dt
    bof = v2["ratio"] * v2["scr"]
    return {"bscr": bscr_net, "scr_total": scr, "ratio": bof / scr}


def _marge_cima(hyp, donnees, v1, v2) -> pd.DataFrame:
    """Marge de solvabilité du code CIMA, à titre de comparaison de référentiels."""
    p = hyp["cima"]
    primes_non_vie = (sum(l["primes_n"] for l in hyp["non_vie"]["lignes"].values())
                      + hyp["sante_nslt"]["primes_n"])
    sinistres = float(donnees["ultimes_vrais"].query("lob != 'sante_frais_soins'")
                      .groupby("annee_survenance")["ultime_vrai"].sum().tail(3).mean())
    marge_non_vie = max(p["non_vie"]["taux_primes"] * primes_non_vie,
                        p["non_vie"]["taux_sinistres"] * sinistres)
    pm_vie = hyp["epargne_euro"]["pm_totale"] + hyp["epargne_uc"]["pm_totale"]
    capitaux = float(donnees["mp_temporaire_deces"]["capitaux_sous_risque"].sum())
    marge_vie = (p["vie"]["taux_provisions"] * pm_vie
                 + p["vie"]["taux_capitaux_sous_risque"] * capitaux)
    return pd.DataFrame([
        {"referentiel": "CIMA — marge non-vie", "exigence": marge_non_vie},
        {"referentiel": "CIMA — marge vie", "exigence": marge_vie},
        {"referentiel": "CIMA — total", "exigence": marge_non_vie + marge_vie},
        {"referentiel": "Solvabilité II — MCR", "exigence": v1["mcr_valeur"]},
        {"referentiel": "Solvabilité II — SCR (V2)", "exigence": v2["scr"]},
    ])


def _tracer(distributions: dict[str, np.ndarray], variantes: dict[str, float], quantile: float):
    figure, axes = plt.subplots(figsize=(10, 5.5))
    couleurs = {"gaussienne": "#1f77b4", "student": "#d62728"}
    for famille, pertes in distributions.items():
        axes.hist(pertes, bins=200, density=True, histtype="step", linewidth=1.4,
                  color=couleurs[famille], label=f"copule {famille}")
        var = np.quantile(pertes, quantile)
        axes.axvline(var, color=couleurs[famille], linestyle="--", linewidth=1.0)
    moyenne = float(np.mean(list(distributions.values())[0]))
    styles = {"Formule standard": ("#2ca02c", ":"), "Formule standard avec USP": ("#ff7f0e", "-.")}
    for nom, scr in variantes.items():
        if nom in styles:
            couleur, style = styles[nom]
            axes.axvline(moyenne + scr, color=couleur, linestyle=style, linewidth=1.6,
                         label=f"{nom} (moyenne + SCR)")
    axes.set_xlim(np.quantile(list(distributions.values())[0], 0.001),
                  np.quantile(distributions["student"], 0.9995))
    axes.set_xlabel("Charge de sinistres non-vie sur un an (M€)")
    axes.set_ylabel("Densité")
    axes.set_title("Distribution agrégée primes + réserves + catastrophe, "
                   f"VaR {quantile:.1%} en pointillés")
    axes.legend(frameon=False, fontsize=9)
    axes.spines[["top", "right"]].set_visible(False)
    return figure


def _afficher(sortie, mesures, duree):
    print("\nDiagnostics par ligne d'activité")
    print(sortie["diagnostics"].round(3).to_string(index=False))
    print("\nBlocs de risque (M€)")
    for bloc in ("primes", "reserves", "catastrophe_brute", "catastrophe_nette",
                 "modele_interne_gaussienne", "modele_interne_student"):
        m = mesures[bloc]
        print(f"  {bloc:28s} moyenne {m['moyenne']:8,.1f}  VaR 99,5 % {m['var']:8,.1f}  "
              f"SCR {m['scr']:8,.1f}  TVaR−moyenne {m['scr_tvar']:8,.1f}")
    print("\nParamètres propres à l'entreprise (risque de réserve)")
    print(sortie["usp"].round(4).to_string(index=False))
    print("\nComparaison des mesures")
    for r in sortie["comparaison"].itertuples():
        print(f"  {r.mesure:38s} SCR non-vie {r.scr_non_vie:7,.1f}  "
              f"SCR total {r.scr_total:8,.1f}  ratio {r.ratio:5.0%}")
    print("\nComparaison de référentiels")
    for r in sortie["cima"].itertuples():
        print(f"  {r.referentiel:34s} {r.exigence:9,.1f} M€")
    print(f"\nCalcul effectué en {duree:.0f} s\n")


if __name__ == "__main__":
    calculer()
