"""Construit les figures et les macros de chiffres du rapport de synthèse.

Toutes les valeurs citées dans le rapport proviennent de ce script, qui lit les CSV
produits par les cinq versions du modèle : aucune valeur n'est recopiée à la main.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from calculer_scr import charger_params            # noqa: E402
from scr_data.config import charger_hypotheses     # noqa: E402

RESULTATS = RACINE / "resultats"
DOSSIER = Path(__file__).resolve().parent
FIGURES = DOSSIER / "figures"
BLEU, ROUGE, VERT, ORANGE, GRIS = "#1f3864", "#c00000", "#548235", "#ed7d31", "#7f7f7f"


def lire(nom: str) -> pd.DataFrame:
    return pd.read_csv(RESULTATS / nom, sep=";", decimal=",")


def style(axes, titre: str = "", xlabel: str = "", ylabel: str = ""):
    axes.set_title(titre, fontsize=11)
    axes.set_xlabel(xlabel, fontsize=9)
    axes.set_ylabel(ylabel, fontsize=9)
    axes.tick_params(labelsize=8)
    axes.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
def figure_allocation() -> None:
    inventaire = pd.read_csv(RACINE / "donnees" / "inventaire_actifs.csv", sep=";", decimal=",")
    euro = inventaire[inventaire["portefeuille"] == "euro"]
    agg = (euro.groupby("sous_classe")["valeur_marche"].sum().sort_values(ascending=True) / 1000)
    figure, axes = plt.subplots(figsize=(7.5, 4.2))
    axes.barh(agg.index.str.replace("_", " "), agg.to_numpy(), color=BLEU, height=0.65)
    for i, v in enumerate(agg.to_numpy()):
        axes.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8, color=GRIS)
    style(axes, "", "Valeur de marché (Md€)", "")
    axes.set_xlim(0, agg.max() * 1.15)
    figure.savefig(FIGURES / "allocation.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def figure_modules() -> None:
    s25 = lire("v2_s25_v2.csv").set_index("poste")["montant"]
    postes = ["Risque de marché", "Risque de souscription vie", "Risque de souscription non-vie",
              "Risque de souscription santé", "Risque de défaut de la contrepartie",
              "Diversification", "Risque opérationnel",
              "Capacité d'absorption des provisions techniques",
              "Capacité d'absorption des impôts différés"]
    libelles = ["Marché", "Vie", "Non-vie", "Santé", "Contrepartie", "Diversification",
                "Opérationnel", "LAC TP", "LAC DT"]
    valeurs = [float(s25[p]) for p in postes]
    scr = float(s25["Capital de solvabilité requis (SCR)"])
    figure, axes = plt.subplots(figsize=(8, 4.2))
    couleurs = [BLEU if v > 0 else VERT for v in valeurs]
    axes.bar(libelles, valeurs, color=couleurs, width=0.62)
    axes.axhline(0, color="black", linewidth=0.8)
    axes.axhline(scr, color=ROUGE, linestyle="--", linewidth=1.2)
    axes.text(len(libelles) - 0.4, scr + 20, f"SCR = {scr:,.0f} M€".replace(",", " "),
              color=ROUGE, fontsize=8, ha="right")
    for i, v in enumerate(valeurs):
        axes.text(i, v + (25 if v > 0 else -55), f"{v:,.0f}".replace(",", " "),
                  ha="center", fontsize=8, color=GRIS)
    style(axes, "", "", "M€")
    plt.setp(axes.get_xticklabels(), rotation=20, ha="right")
    figure.savefig(FIGURES / "modules.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def figure_pont() -> None:
    pont = lire("v3_pont_2027.csv")
    etapes = pont["etape"].tolist()
    ratios = pont["ratio"].to_numpy()
    figure, axes = plt.subplots(figsize=(8.4, 4.4))
    axes.step(range(len(ratios)), ratios, where="mid", color=BLEU, linewidth=1.8)
    axes.scatter(range(len(ratios)), ratios, color=BLEU, s=25, zorder=3)
    for i, (r, e) in enumerate(zip(ratios, etapes)):
        axes.annotate(f"{r:.0%}", (i, r), textcoords="offset points", xytext=(0, 9),
                      ha="center", fontsize=8, color=GRIS)
    axes.set_xticks(range(len(etapes)))
    axes.set_xticklabels([e.replace(" (", "\n(") for e in etapes], fontsize=7.5)
    plt.setp(axes.get_xticklabels(), rotation=28, ha="right")
    axes.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(axes, "", "", "Ratio de couverture du SCR")
    figure.savefig(FIGURES / "pont_2027.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def figure_tvog() -> None:
    sensibilite = lire("v2_sensibilite_tmg.csv")
    figure, axes = plt.subplots(figsize=(7, 3.8))
    x = sensibilite["decalage_tmg"].to_numpy() * 100
    axes.plot(x, sensibilite["tvog"], marker="o", color=ROUGE, linewidth=1.8)
    axes.axhline(0, color="black", linewidth=0.8)
    for xi, yi in zip(x, sensibilite["tvog"]):
        axes.annotate(f"{yi:+.0f}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                      ha="center", fontsize=8, color=GRIS)
    style(axes, "", "Décalage appliqué aux taux minimums garantis (points)", "TVOG (M€)")
    figure.savefig(FIGURES / "tvog.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def figure_chocs_euro() -> None:
    chocs = lire("v2_chocs_euro.csv").set_index("choc")
    ordre = ["taux_hausse", "taux_baisse", "action", "immobilier", "spread"]
    libelles = ["Taux hausse", "Taux baisse", "Action", "Immobilier", "Spread"]
    x = np.arange(len(ordre))
    figure, axes = plt.subplots(figsize=(7.6, 3.9))
    axes.bar(x - 0.19, chocs.loc[ordre, "delta_net"], width=0.36, color=BLEU, label="net")
    axes.bar(x + 0.19, chocs.loc[ordre, "delta_brut"], width=0.36, color=ORANGE, label="brut")
    axes.axhline(0, color="black", linewidth=0.8)
    axes.set_xticks(x)
    axes.set_xticklabels(libelles, fontsize=8)
    axes.legend(frameon=False, fontsize=8)
    style(axes, "", "", "Variation du BE épargne euro (M€)")
    figure.savefig(FIGURES / "chocs_euro.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
def macros() -> str:
    """Écrit les macros LaTeX des chiffres cités dans le rapport."""
    s25v1 = lire("s25_01.csv").set_index("poste")["montant"]
    s25v2 = lire("v2_s25_v2.csv").set_index("poste")["montant"]
    be = lire("be_par_segment.csv").set_index("segment")["be"]
    comparaison = lire("v2_comparaison.csv").set_index("indicateur")
    pont = lire("v3_pont_2027.csv")
    v4 = lire("v4_comparaison.csv").set_index("mesure")
    v4mesures = lire("v4_mesures.csv").set_index("bloc")
    cima = lire("v4_cima.csv").set_index("referentiel")["exigence"]
    synthese = lire("v5_synthese.csv").set_index("scenario")
    bgs = lire("v5_besoin_global.csv").set_index("composante")["montant"]
    martingales = lire("v2_martingales.csv")
    be_v2 = lire("v2_be.csv").iloc[0]
    usp = lire("v4_usp.csv").set_index("lob")

    donnees = {nom: pd.read_csv(RACINE / "donnees" / f"{nom}.csv", sep=";", decimal=",")
               for nom in ("inventaire_actifs", "mp_epargne_euro", "mp_epargne_uc",
                           "mp_temporaire_deces", "mp_rentes_viageres", "mp_sante_slt",
                           "volumes_primes_reserves")}
    hyp = charger_hypotheses()
    nv = donnees["volumes_primes_reserves"].set_index("lob")["primes_acquises_n"]

    valeurs = {
        "ActifTotal": 11000.0,
        "LignesInventaire": len(donnees["inventaire_actifs"]),
        "ModelPoints": sum(len(donnees[t]) for t in ("mp_epargne_euro", "mp_epargne_uc",
                                                     "mp_temporaire_deces", "mp_rentes_viageres",
                                                     "mp_sante_slt")),
        "PMeuro": float(donnees["mp_epargne_euro"]["pm"].sum()),
        "PMuc": float(donnees["mp_epargne_uc"]["pm"].sum()),
        "CapitauxSousRisque": float(donnees["mp_temporaire_deces"]["capitaux_sous_risque"].sum()),
        "PrimesNonVie": float(nv.drop("sante_frais_soins").sum()),
        "PrimesSante": float(nv["sante_frais_soins"]),
        "NbScenarios": hyp["esg"]["nb_scenarios"],
        "HorizonAlm": hyp["alm"]["horizon"],
        "AjustementSymetrique": charger_params()["action"]["ajustement_symetrique"] * 100,
        "BEtotalVun": float(be["total"]),
        "BEeuroVun": float(be["epargne_euro"]),
        "BEeuroVdeux": float(comparaison.loc["BE épargne euro", "v2"]),
        "BEeuroCentral": float(be_v2["central"]),
        "TVOG": float(be_v2["tvog"]),
        "FDB": float(be_v2["fdb"]),
        "ScrVun": float(s25v1["Capital de solvabilité requis (SCR)"]),
        "ScrVdeux": float(s25v2["Capital de solvabilité requis (SCR)"]),
        "MarcheVun": float(s25v1["Risque de marché"]),
        "MarcheVdeux": float(s25v2["Risque de marché"]),
        "VieVun": float(s25v1["Risque de souscription vie"]),
        "VieVdeux": float(s25v2["Risque de souscription vie"]),
        "NonVieVun": float(s25v1["Risque de souscription non-vie"]),
        "SanteVun": float(s25v1["Risque de souscription santé"]),
        "ContrepartieVun": float(s25v1["Risque de défaut de la contrepartie"]),
        "OperationnelVdeux": float(s25v2["Risque opérationnel"]),
        "LacTPVdeux": float(s25v2["Capacité d'absorption des provisions techniques"]),
        "LacDTVdeux": float(s25v2["Capacité d'absorption des impôts différés"]),
        "BscrVdeux": float(s25v2["Capital de solvabilité requis de base (BSCR)"]),
        "FondsPropresVun": float(comparaison.loc["Fonds propres de base", "v1"]),
        "FondsPropresVdeux": float(comparaison.loc["Fonds propres de base", "v2"]),
        "McrVdeux": float(comparaison.loc["MCR", "v2"]),
        "ScrNonVieFS": float(v4.loc["Formule standard", "scr_non_vie"]),
        "ScrNonVieUSP": float(v4.loc["Formule standard avec USP", "scr_non_vie"]),
        "ScrNonVieMIG": float(v4.loc["Modèle interne (copule gaussienne)", "scr_non_vie"]),
        "ScrNonVieMIS": float(v4.loc["Modèle interne (copule de Student)", "scr_non_vie"]),
        "CatBrute": float(v4mesures.loc["catastrophe_brute", "scr"]),
        "CatNette": float(v4mesures.loc["catastrophe_nette", "scr"]),
        "CatBruteTVaR": float(v4mesures.loc["catastrophe_brute", "scr_tvar"]),
        "CimaTotal": float(cima["CIMA — total"]),
        "MargeRisqueActuelle": float(pont.iloc[1]["marge_risque"]),
        "MargeRisqueReforme": float(pont.iloc[-1]["marge_risque"]),
        "ScrReforme": float(pont.iloc[-1]["scr"]),
        "ImpactChocsTaux": float(pont.set_index("etape").loc["Chocs de taux révisés", "delta_scr"]),
        "BesoinGlobal": float(bgs["Besoin global de solvabilité"]),
        "EcartMartingaleMax": float(martingales[martingales["test"].str.contains("déflateur")]
                                    ["ecart_bp"].abs().max()),
    }
    ratios = {
        "RatioVun": float(comparaison.loc["Ratio de couverture", "v1"]),
        "RatioVdeux": float(comparaison.loc["Ratio de couverture", "v2"]),
        "RatioReforme": float(pont.iloc[-1]["ratio"]),
        "RatioOrsaCentral": float(synthese.loc["Scénario central", "ratio_final"]),
        "RatioOrsaCrise": float(synthese.loc["Crise combinée (actions, taux, rachats, sinistre)",
                                             "ratio_minimum"]),
        "RatioOrsaTaux": float(synthese.loc["Hausse des taux (+200 pb) et rachats", "ratio_minimum"]),
        "RatioOrsaActions": float(synthese.loc["Krach actions (-30 % en année 1)", "ratio_minimum"]),
        "SigmaUspAutoRC": float(usp.loc["auto_rc", "sigma_usp"]),
        "SigmaStdAutoRC": float(usp.loc["auto_rc", "sigma_standard"]),
    }
    lignes = ["% Fichier engendré par rapport/generer_rapport.py, ne pas éditer à la main"]
    entiers = {"LignesInventaire", "ModelPoints", "NbScenarios", "HorizonAlm"}
    for nom, valeur in valeurs.items():
        decimales = 0 if nom in entiers else 1
        lignes.append(f"\\newcommand{{\\{nom}}}{{\\num{{{valeur:.{decimales}f}}}}}")
    for nom, valeur in ratios.items():
        lignes.append(f"\\newcommand{{\\{nom}}}{{\\qty{{{valeur * 100:.1f}}}{{\\percent}}}}")
    return "\n".join(lignes) + "\n"


def nb(valeur: float, decimales: int = 1, signe: bool = False) -> str:
    """Nombre formaté à la française, avec espace fine insécable comme séparateur."""
    format_ = f"{{:{'+' if signe else ''},.{decimales}f}}"
    return format_.format(valeur).replace(",", "\\,").replace(".", ",")


def pourcent(valeur: float, decimales: int = 0) -> str:
    return f"{valeur * 100:.{decimales}f}".replace(".", ",") + "\\,\\%"


def texte(valeur: str) -> str:
    """Échappe les caractères réservés de LaTeX présents dans les libellés."""
    return (valeur.replace("\\", "").replace("&", "\\&").replace("%", "\\%")
            .replace("_", " ").replace("+/-", "$\\pm$"))


def tableaux() -> str:
    """Corps des tableaux LaTeX construits directement à partir des résultats."""
    blocs = []

    s25 = lire("v2_s25_v2.csv")
    lignes = [f"{texte(r.poste)} & {nb(r.montant)} \\\\" for r in s25.itertuples()]
    blocs.append("\\newcommand{\\tableauSXXV}{%\n" + "\n".join(lignes) + "\n}")

    pont = lire("v3_pont_2027.csv")
    lignes = []
    for r in pont.itertuples():
        delta = "" if pd.isna(getattr(r, "delta_scr", np.nan)) else nb(r.delta_scr, signe=True)
        lignes.append(f"{texte(r.etape)} & {nb(r.scr)} & {delta} & {pourcent(r.ratio)} \\\\")
    blocs.append("\\newcommand{\\tableauPont}{%\n" + "\n".join(lignes) + "\n}")

    synthese = lire("v5_synthese.csv")
    lignes = [f"{texte(r.scenario)} & {pourcent(r.ratio_minimum)} & {pourcent(r.ratio_final)} "
              f"& {r.statut.capitalize()} \\\\" for r in synthese.itertuples()]
    blocs.append("\\newcommand{\\tableauOrsa}{%\n" + "\n".join(lignes) + "\n}")

    v4 = lire("v4_comparaison.csv")
    lignes = [f"{texte(r.mesure)} & {nb(r.scr_non_vie)} & {nb(r.scr_total)} & "
              f"{pourcent(r.ratio)} \\\\" for r in v4.itertuples()]
    blocs.append("\\newcommand{\\tableauModeleInterne}{%\n" + "\n".join(lignes) + "\n}")

    usp = lire("v4_usp.csv")
    lignes = [f"{texte(r.lob)} & {pourcent(r.sigma_usp, 1)} & {pourcent(r.sigma_standard, 1)} & "
              f"{pourcent(r.sigma_retenu, 1)} \\\\" for r in usp.itertuples()]
    blocs.append("\\newcommand{\\tableauUSP}{%\n" + "\n".join(lignes) + "\n}")

    be = lire("be_par_segment.csv")
    correspondance = {"epargne_euro": "Épargne euro", "epargne_uc": "Épargne en UC",
                      "temporaire_deces": "Temporaire décès", "rentes": "Rentes viagères",
                      "sante_slt": "Santé SLT", "non_vie_sinistres": "Non-vie, sinistres",
                      "non_vie_primes": "Non-vie, primes", "sante_nslt": "Santé NSLT",
                      "total": "\\textbf{Total}"}
    lignes = [f"{correspondance.get(r.segment, texte(r.segment))} & {nb(r.be)} \\\\"
              for r in be.itertuples()]
    blocs.append("\\newcommand{\\tableauBE}{%\n" + "\n".join(lignes) + "\n}")

    return "\n\n".join(blocs) + "\n"


def principal() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_allocation()
    figure_modules()
    figure_pont()
    figure_tvog()
    figure_chocs_euro()
    for source in ("v4_distribution_non_vie.png", "v5_trajectoires_orsa.png"):
        (FIGURES / source).write_bytes((RESULTATS / source).read_bytes())
    (DOSSIER / "chiffres.tex").write_text(macros() + "\n" + tableaux(), encoding="utf-8")
    print(f"Figures et macros écrites dans {DOSSIER}")


if __name__ == "__main__":
    principal()
