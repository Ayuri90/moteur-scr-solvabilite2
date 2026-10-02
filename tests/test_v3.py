"""Tests du régime 2027 (extrapolation alternative, chocs de taux, marge de risque)."""
import numpy as np
import pytest

import calculer_scr
from calculer_v3 import charger_reforme
from scr.marge_risque import marge_risque, profil_run_off
from scr.reforme import courbe_choquee_2027, extrapoler_2027, params_2027
from scr_data.config import charger_hypotheses
from scr_data.courbe import Actualisation, construire_courbe


@pytest.fixture(scope="module")
def contexte():
    hyp = charger_hypotheses()
    courbe, _ = construire_courbe(hyp)
    return hyp, courbe, calculer_scr.charger_params(), charger_reforme(hyp)


def test_extrapolation_inchangee_avant_le_point_de_lissage(contexte):
    hyp, courbe, _, reforme = contexte
    c27 = extrapoler_2027(courbe, reforme, hyp["courbe"]["ufr"])
    fsp = reforme["extrapolation"]["premier_point_lissage"]
    avant = courbe["maturite"] <= fsp
    assert np.allclose(c27.loc[avant, "taux_spot"], courbe.loc[avant, "taux_spot"])


def test_extrapolation_converge_vers_l_ufr(contexte):
    hyp, courbe, _, reforme = contexte
    c27 = extrapoler_2027(courbe, reforme, hyp["courbe"]["ufr"]).set_index("maturite")
    forward_long = c27.loc[120, "forward_1an"]
    assert abs(forward_long - hyp["courbe"]["ufr"]) < 0.005


def test_chocs_2027_orientes_dans_le_bon_sens(contexte):
    hyp, courbe, _, reforme = contexte
    base = courbe.set_index("maturite")["taux_spot"]
    haut = courbe_choquee_2027(courbe, reforme, "hausse", hyp["courbe"]["ufr"]).set_index("maturite")["taux_spot"]
    bas = courbe_choquee_2027(courbe, reforme, "baisse", hyp["courbe"]["ufr"]).set_index("maturite")["taux_spot"]
    liquides = [1, 5, 10, 20]
    assert (haut.loc[liquides] > base.loc[liquides]).all()
    assert (bas.loc[liquides] < base.loc[liquides]).all()


def test_planchers_negatifs_du_choc_baissier(contexte):
    hyp, courbe, _, reforme = contexte
    bas = courbe_choquee_2027(courbe, reforme, "baisse", hyp["courbe"]["ufr"]).set_index("maturite")["taux_spot"]
    planchers = reforme["taux"]["planchers_baisse"]
    assert bas.loc[1] >= planchers[1] - 1e-12
    assert bas.loc[20] >= planchers[20] - 1e-12


def test_params_2027_modifient_bien_les_parametres(contexte):
    _, _, params, reforme = contexte
    p = params_2027(params, reforme, {"chocs_taux", "correlation_taux_spread",
                                      "reassurance_non_proportionnelle"})
    ordre = p["correlations_marche"]["ordre"]
    i, j = ordre.index("taux"), ordre.index("spread")
    assert p["correlations_marche"]["matrice"][i][j] == "B"
    assert p["correlations_marche"]["b_baisse"] == 0.25
    assert p["taux"]["methode"] == "2027"
    assert p["non_vie"]["sigma"]["auto_rc"]["primes"] == pytest.approx(
        params["non_vie"]["sigma"]["auto_rc"]["primes"] * reforme["non_vie"]["facteur_np"])
    # les paramètres d'origine ne sont pas modifiés
    assert params["taux"].get("methode") != "2027"


def test_profil_run_off_decroissant(contexte):
    _, courbe, _, _ = contexte
    actu = Actualisation(courbe)
    flux = np.full(30, 100.0)
    profil = profil_run_off(flux, actu)
    assert profil[0] == pytest.approx(1.0)
    assert np.all(np.diff(profil) < 0)


def test_marge_risque_reduite_par_le_regime_2027(contexte):
    _, courbe, _, reforme = contexte
    actu = Actualisation(courbe)
    flux = np.exp(-np.arange(40) / 12) * 300
    p = reforme["marge_risque"]
    actuelle, _ = marge_risque(400.0, flux, actu, p["cout_du_capital_actuel"])
    nouvelle, _ = marge_risque(400.0, flux, actu, p["cout_du_capital_2027"],
                               facteur_lambda=p["lambda"], plancher=p["plancher_lambda"])
    assert nouvelle < actuelle
    assert nouvelle / actuelle < 0.85
