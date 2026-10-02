"""V5 — ORSA : projection du ratio sur l'horizon du plan et scénarios adverses.

Produit :
- la trajectoire centrale du bilan, du SCR et du ratio de couverture ;
- six scénarios adverses (krach actions, hausse et baisse des taux, catastrophe,
  inflation des frais, crise combinée) avec ratio minimal et année de franchissement
  des seuils d'appétence ;
- le besoin global de solvabilité, qui complète le SCR par les risques mal captés
  par la formule standard ;
- un graphique des trajectoires.

Usage :
    python calculer_v5.py
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
from scr.orsa import ProjectionORSA, besoin_global_solvabilite, etat_initial
from scr_data.config import RACINE, charger_hypotheses
from scr_data.courbe import construire_courbe


def calculer(ecrire: bool = True) -> dict:
    debut = time.time()
    hyp = charger_hypotheses()
    params = calculer_scr.charger_params()
    donnees = calculer_scr.charger_donnees(RACINE / hyp["general"]["dossier_sortie"])
    courbe, _ = construire_courbe(hyp)

    v1 = calculer_scr.calculer(ecrire=False)
    v2 = calculer_v2.calculer(ecrire=False)
    be_segments = v1["be"].set_index("segment")["be"].to_dict()
    be_segments["epargne_euro"] = v2["be"]["stochastique"]

    etat = etat_initial(be_segments, hyp, donnees["inventaire_actifs"])
    # modules bruts d'absorption : le SCR projeté = BSCR brut + opérationnel + LAC TP + LAC DT
    modules = v2["bscr_brut"].set_index("module")["scr"].to_dict()
    modules.pop("BSCR", None)
    s25 = v2["s25_v2"].set_index("poste")["montant"]
    projection = ProjectionORSA(
        etat, modules, float(s25["Risque opérationnel"]),
        float(s25["Capacité d'absorption des provisions techniques"]),
        float(s25["Capacité d'absorption des impôts différés"]),
        hyp, params, float(courbe.loc[courbe["maturite"] == 10, "taux_spot"].iloc[0]))

    ecart = abs(projection.scr_initial() / v2["scr"] - 1)
    if ecart > 0.01:
        raise AssertionError(f"Le SCR reconstitué en t=0 s'écarte de {ecart:.1%} du SCR de la V2.")

    trajectoires = {"central": projection.projeter()}
    for cle, scenario in hyp["orsa"]["scenarios"].items():
        trajectoires[cle] = projection.projeter(scenario)

    synthese = _synthese(trajectoires, hyp, v2)
    inverses = _stress_inverses(projection, hyp)
    bgs = besoin_global_solvabilite(v2["scr"], hyp)
    detail = pd.concat([t.assign(scenario=nom) for nom, t in trajectoires.items()],
                       ignore_index=True)
    figure = _tracer(trajectoires, hyp, v2) if ecrire else None

    sortie = {"trajectoires": detail, "synthese": synthese, "besoin_global": bgs,
              "stress_inverses": inverses, "ratio_initial": v2["ratio"]}
    if ecrire:
        dossier = RACINE / "resultats"
        dossier.mkdir(exist_ok=True)
        for nom in ("trajectoires", "synthese", "besoin_global", "stress_inverses"):
            sortie[nom].to_csv(dossier / f"v5_{nom}.csv", index=False, sep=";", decimal=",",
                               encoding="utf-8-sig")
        figure.savefig(dossier / "v5_trajectoires_orsa.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
        _afficher(sortie, hyp, v2, time.time() - debut)
    return sortie


def _stress_inverses(projection, hyp) -> pd.DataFrame:
    """Stress tests inversés : amplitude du choc qui amène le ratio à la limite d'appétence."""
    limite = hyp["orsa"]["appetence"]["limite"]
    lignes = []
    definitions = {
        "Choc action instantané": ("choc_action", -0.05, -0.95, {"annee": 1}),
        "Hausse des taux": ("choc_taux", 0.005, 0.10, {"annee": 1, "rachats_supplementaires": 0.05,
                                                       "duree_rachats": 3}),
        "Sinistre exceptionnel net": ("sinistre_net", 10.0, 1500.0, {"annee": 1}),
    }
    for libelle, (cle, depart, extreme, base) in definitions.items():
        bas, haut = depart, extreme
        for _ in range(28):
            milieu = 0.5 * (bas + haut)
            ratio = projection.projeter({**base, cle: milieu})["ratio"].min()
            if ratio > limite:
                bas = milieu
            else:
                haut = milieu
        lignes.append({"stress": libelle, "amplitude_critique": 0.5 * (bas + haut),
                       "ratio_cible": limite})
    return pd.DataFrame(lignes)


def _libelle(cle: str, hyp) -> str:
    if cle == "central":
        return "Scénario central"
    return hyp["orsa"]["scenarios"][cle]["libelle"]


def _synthese(trajectoires: dict[str, pd.DataFrame], hyp, v2) -> pd.DataFrame:
    a = hyp["orsa"]["appetence"]
    lignes = []
    for cle, t in trajectoires.items():
        ratio_min = float(t["ratio"].min())
        annee_min = int(t.loc[t["ratio"].idxmin(), "annee"])
        sous_alerte = t.loc[t["ratio"] < a["alerte"], "annee"]
        sous_limite = t.loc[t["ratio"] < a["limite"], "annee"]
        besoin = max(0.0, float((a["limite"] * t["scr"] - t["fonds_propres"]).max()))
        lignes.append({
            "scenario": _libelle(cle, hyp), "ratio_final": float(t["ratio"].iloc[-1]),
            "ratio_minimum": ratio_min, "annee_du_minimum": annee_min,
            "annee_franchissement_alerte": int(sous_alerte.iloc[0]) if len(sous_alerte) else 0,
            "annee_franchissement_limite": int(sous_limite.iloc[0]) if len(sous_limite) else 0,
            "resultat_net_cumule": float(t["resultat_net"].sum()),
            "capital_a_injecter": besoin,
            "statut": ("ROUGE" if len(sous_limite) else "ORANGE" if len(sous_alerte)
                       else "JAUNE" if ratio_min < a["cible"] else "VERT")})
    return pd.DataFrame(lignes)


def _tracer(trajectoires: dict[str, pd.DataFrame], hyp, v2):
    a = hyp["orsa"]["appetence"]
    figure, axes = plt.subplots(figsize=(10, 5.5))
    couleurs = plt.get_cmap("tab10")
    for k, (cle, t) in enumerate(trajectoires.items()):
        annees = np.concatenate([[0], t["annee"].to_numpy()])
        ratios = np.concatenate([[v2["ratio"]], t["ratio"].to_numpy()])
        axes.plot(annees, ratios, marker="o", markersize=3.5,
                  linewidth=2.2 if cle == "central" else 1.3,
                  color="#111111" if cle == "central" else couleurs(k),
                  label=_libelle(cle, hyp))
    axes.axhline(a["cible"], color="#2ca02c", linestyle=":", linewidth=1.2)
    axes.axhline(a["alerte"], color="#ff7f0e", linestyle=":", linewidth=1.2)
    axes.axhline(a["limite"], color="#d62728", linestyle="--", linewidth=1.4)
    axes.text(hyp["orsa"]["horizon"] - 0.9, a["limite"] + 0.015, "limite d'appétence",
              color="#d62728", fontsize=8)
    axes.text(hyp["orsa"]["horizon"] - 0.9, a["alerte"] + 0.015, "seuil d'alerte",
              color="#ff7f0e", fontsize=8)
    axes.text(hyp["orsa"]["horizon"] - 0.9, a["cible"] + 0.015, "cible", color="#2ca02c", fontsize=8)
    axes.set_ylim(a["limite"] - 0.12, None)
    axes.set_xlabel("Année de projection")
    axes.set_ylabel("Ratio de couverture du SCR")
    axes.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes.set_title("ORSA — trajectoires du ratio de solvabilité")
    axes.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    axes.spines[["top", "right"]].set_visible(False)
    return figure


def _afficher(sortie, hyp, v2, duree):
    print(f"\nORSA — ratio de départ {v2['ratio']:.0%}, horizon {hyp['orsa']['horizon']} ans\n")
    centrale = sortie["trajectoires"].query("scenario == 'central'")
    print("Trajectoire centrale")
    print(f"  {'an':>3s} {'actif':>10s} {'fonds propres':>14s} {'SCR':>9s} {'ratio':>7s} "
          f"{'résultat':>9s} {'taux servi':>11s}")
    for r in centrale.itertuples():
        print(f"  {r.annee:3d} {r.actif:10,.0f} {r.fonds_propres:14,.0f} {r.scr:9,.0f} "
              f"{r.ratio:7.0%} {r.resultat_net:9,.1f} {r.taux_servi:11.2%}")
    print("\nScénarios")
    print(f"  {'scénario':44s} {'ratio final':>11s} {'ratio min':>10s} {'année':>6s} "
          f"{'capital':>9s}  statut")
    for r in sortie["synthese"].itertuples():
        print(f"  {r.scenario:44s} {r.ratio_final:11.0%} {r.ratio_minimum:10.0%} "
              f"{r.annee_du_minimum:6d} {r.capital_a_injecter:9,.1f}  {r.statut}")
    print("\nStress tests inversés (amplitude amenant le ratio à la limite d'appétence)")
    for r in sortie["stress_inverses"].itertuples():
        print(f"  {r.stress:34s} {r.amplitude_critique:+10.3f}")
    print("\nBesoin global de solvabilité (M€)")
    for r in sortie["besoin_global"].itertuples():
        print(f"  {r.composante:40s} {r.montant:9,.1f}")
    print(f"\nCalcul effectué en {duree:.0f} s\n")


if __name__ == "__main__":
    calculer()
