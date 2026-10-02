"""Tests du moteur SCR (étape 1 : marché et contrepartie)."""
import numpy as np
import pandas as pd
import pytest

from calculer_scr import calculer, charger_params
from scr.contrepartie import scr_type2
from scr.marche import courbe_choquee, facteur_stress_spread
from scr_data.config import charger_hypotheses
from scr_data.courbe import construire_courbe


@pytest.fixture(scope="module")
def params():
    return charger_params()


@pytest.fixture(scope="module")
def resultats():
    return calculer(ecrire=False)


def test_choc_taux_hausse_respecte_le_plancher(params):
    courbe, _ = construire_courbe(charger_hypotheses())
    haut = courbe_choquee(courbe, params, "hausse")
    ecart = haut["taux_spot"] - courbe["taux_spot"]
    assert (ecart >= params["taux"]["plancher_hausse_absolu"] - 1e-12).all()


def test_choc_taux_baisse_diminue_les_taux_positifs(params):
    courbe, _ = construire_courbe(charger_hypotheses())
    bas = courbe_choquee(courbe, params, "baisse")
    positifs = courbe["taux_spot"] > 0
    assert (bas.loc[positifs, "taux_spot"] < courbe.loc[positifs, "taux_spot"]).all()


@pytest.mark.parametrize("cqs, duration, attendu", [
    (0, 3, 0.027),      # 0,9 % x 3
    (2, 5, 0.070),      # 1,4 % x 5
    (0, 12, 0.080),     # 7,0 % + 0,5 % x 2
    (3, 25, 0.325),     # 30 % + 0,5 % x 5
    (None, 4, 0.120),   # non noté : 3 % x 4
])
def test_facteurs_de_stress_spread(params, cqs, duration, attendu):
    assert facteur_stress_spread(cqs, duration, params, covered=False) == pytest.approx(attendu, abs=1e-6)


def test_duration_plancher_un_an(params):
    assert facteur_stress_spread(2, 0.2, params, False) == pytest.approx(0.014)


def test_scr_type2(params):
    cp = pd.DataFrame([{"type_exposition": "type2_creances", "montant": 100.0},
                       {"type_exposition": "type2_creances_echues_3m", "montant": 10.0}])
    assert scr_type2(cp, params) == pytest.approx(0.15 * 100 + 0.90 * 10)


def test_agregation_marche_beneficie_de_la_diversification(resultats):
    table = resultats["marche"]
    agrege = table.loc[table["sous_module"] == "SCR marché (agrégé)", "scr"].iloc[0]
    somme = table.loc[table["sous_module"] != "SCR marché (agrégé)", "scr"].sum()
    plus_gros = table.loc[table["sous_module"] != "SCR marché (agrégé)", "scr"].max()
    assert plus_gros < agrege < somme


def test_be_rentes_conforme_au_calage(resultats):
    be = resultats["be"].set_index("segment")["be"]
    assert be["rentes"] == pytest.approx(150.0, rel=1e-3)
    # le calage porte sur les arrérages ; le BE inclut en plus les frais de gestion
    frais = charger_hypotheses()["frais"]["gestion_sante_slt"]
    assert be["sante_slt"] == pytest.approx(120.0 * (1 + frais), rel=1e-3)


def test_chain_ladder_retrouve_les_ultimes_simules():
    from scr.passif import chain_ladder
    from calculer_scr import charger_donnees
    from scr_data.config import RACINE, charger_hypotheses
    d = charger_donnees(RACINE / charger_hypotheses()["general"]["dossier_sortie"])
    tri = d["triangles_reglements"]
    vrais = d["ultimes_vrais"]
    for lob, sous in tri.groupby("lob"):
        m = sous.pivot(index="annee_survenance", columns="developpement", values="reglements_cumules")
        cl = chain_ladder(m)
        # l'écart au total reste modéré : la queue de développement n'est pas observée
        assert abs(cl["ultimes"].sum() / vrais.query("lob == @lob")["ultime_vrai"].sum() - 1) < 0.15


# ---------------------------------------------------------------------------
# Souscription vie et santé
# ---------------------------------------------------------------------------
def test_cat_vie_coherent_avec_les_capitaux_sous_risque(resultats, params):
    from calculer_scr import charger_donnees
    from scr_data.config import RACINE, charger_hypotheses
    d = charger_donnees(RACINE / charger_hypotheses()["general"]["dossier_sortie"])
    capitaux = d["mp_temporaire_deces"]["capitaux_sous_risque"].sum()
    attendu = params["vie"]["cat_mortalite_absolu"] * capitaux
    cat = resultats["vie"].set_index("sous_module").loc["catastrophe", "scr"]
    assert cat == pytest.approx(attendu, rel=0.20)   # actualisation et décrément près


def test_mortalite_ne_charge_pas_les_rentes(resultats):
    """Le choc de mortalité fait baisser le BE des rentes : il ne doit pas être retenu."""
    sante = resultats["sante"].set_index("sous_module")
    assert sante.loc["SLT · mortalite", "scr"] == 0.0
    assert sante.loc["SLT · longevite", "scr"] > 0.0


def test_rachat_retient_le_scenario_le_plus_defavorable(resultats):
    vie = resultats["vie"].set_index("sous_module")
    detail = vie.loc["rachat", "detail"]
    valeurs = [float(x) for x in detail.replace("hausse", "").replace("baisse", "")
               .replace("massif", "").split("/")]
    assert vie.loc["rachat", "scr"] == pytest.approx(max(valeurs), rel=1e-3)


def test_agregations_vie_et_sante(resultats):
    for cle, nom in (("vie", "SCR souscription vie (agrégé)"), ("sante", "SCR souscription santé (agrégé)")):
        table = resultats[cle].set_index("sous_module")
        agrege = table.loc[nom, "scr"]
        detail = table.drop(index=[i for i in table.index if "agrégé" in i])["scr"]
        assert detail.max() < agrege < detail.sum()


def test_scr_sous_modules_positifs(resultats):
    for cle in ("marche", "vie", "sante", "contrepartie"):
        assert (resultats[cle]["scr"] >= 0).all()


# ---------------------------------------------------------------------------
# Non-vie et agrégation finale
# ---------------------------------------------------------------------------
def test_primes_reserves_coherent_avec_trois_sigma(resultats, params):
    """SCR primes et réserves = 3 x sigma x volume, avec diversification entre lignes."""
    detail = resultats["non_vie_detail"].query("bloc == 'primes_reserves'")
    isole = detail["scr_isole"].sum()
    agrege = resultats["non_vie"].set_index("sous_module").loc["primes et réserves", "scr"]
    assert agrege < isole   # bénéfice de diversification entre lignes d'activité


def test_xl_cat_reduit_la_charge_catastrophe(resultats):
    detail = resultats["non_vie_detail"].set_index("composante")
    brut = detail.loc["CAT brut de réassurance", "scr_brut"]
    net = detail.loc["CAT net de réassurance", "scr_brut"]
    assert net < brut


def test_bscr_entre_le_plus_gros_module_et_la_somme(resultats):
    modules = resultats["modules"]
    bscr = resultats["bscr"].set_index("module").loc["BSCR", "scr"]
    assert max(modules.values()) < bscr < sum(modules.values())


def test_lac_dt_plafonnee(resultats, params):
    s25 = resultats["s25"].set_index("poste")["montant"]
    bscr = s25["Capital de solvabilité requis de base (BSCR)"]
    op = s25["Risque opérationnel"]
    lac_tp = s25["Capacité d'absorption des provisions techniques"]
    plafond = params["fiscalite"]["taux_impot"] * (bscr + op + lac_tp)
    assert -s25["Capacité d'absorption des impôts différés"] <= plafond + 1e-9


def test_scr_egal_a_la_somme_des_composantes(resultats):
    s25 = resultats["s25"].set_index("poste")["montant"]
    attendu = (s25["Capital de solvabilité requis de base (BSCR)"] + s25["Risque opérationnel"]
               + s25["Capacité d'absorption des provisions techniques"]
               + s25["Capacité d'absorption des impôts différés"])
    assert s25["Capital de solvabilité requis (SCR)"] == pytest.approx(attendu)


def test_mcr_dans_le_corridor(resultats, params):
    mcr = resultats["mcr_valeur"]
    scr = resultats["scr"]
    assert params["mcr"]["plancher_scr"] * scr <= mcr <= params["mcr"]["plafond_scr"] * scr
    assert mcr >= params["mcr"]["minimum_absolu"]


def test_ratio_de_couverture_plausible(resultats):
    ratio = resultats["moteur"].bof_base / resultats["scr"]
    assert 1.5 < ratio < 4.0
