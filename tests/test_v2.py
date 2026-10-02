"""Tests de la V2 : générateur de scénarios, modèle ALM, LAC TP."""
import numpy as np
import pytest

import calculer_v2
from scr.esg import HullWhite, generer_scenarios, scenario_central
from scr.esg import tests_martingale as controles_martingale
from scr_data.config import charger_hypotheses
from scr_data.courbe import construire_courbe


@pytest.fixture(scope="module")
def hyp():
    return charger_hypotheses()


@pytest.fixture(scope="module")
def courbe(hyp):
    return construire_courbe(hyp)[0]


@pytest.fixture(scope="module")
def scenarios(courbe, hyp):
    return generer_scenarios(courbe, hyp)


@pytest.fixture(scope="module")
def v2():
    return calculer_v2.calculer(ecrire=False)


def test_hull_white_reproduit_la_courbe_initiale(courbe, hyp):
    hw = HullWhite(courbe, hyp["esg"]["hull_white"]["a"], hyp["esg"]["hull_white"]["sigma"])
    for maturite in (1, 5, 10, 30):
        prix = hw.prix_zc(0.0, float(maturite), np.array([hw.r0]))[0]
        reference = float(courbe.set_index("maturite").loc[maturite, "facteur_actualisation"])
        assert prix == pytest.approx(reference, rel=2e-3)


def test_martingale_des_deflateurs(scenarios):
    tests = controles_martingale(scenarios)
    deflateurs = tests[tests["test"].str.contains("déflateur")]
    assert deflateurs["ecart_bp"].abs().max() < 25     # erreur de Monte-Carlo et pas annuel


def test_scenario_central_est_deterministe(courbe, hyp):
    central = scenario_central(courbe, hyp)
    assert central.nb_scenarios == 1
    reference = np.interp(np.arange(1, central.horizon + 1), courbe["maturite"],
                          courbe["facteur_actualisation"])
    assert np.allclose(central.deflateur[0], reference)


def test_fdb_positives_et_lac_tp_plafonnee(v2):
    assert v2["be"]["fdb"] > 0
    assert v2["be"]["garanti"] < v2["be"]["stochastique"]
    assert -v2["lac_tp"] <= v2["be"]["fdb"] + 1e-9
    assert v2["lac_tp"] <= 0


def test_bscr_brut_superieur_au_net(v2):
    brut = v2["bscr_brut"].set_index("module").loc["BSCR", "scr"]
    net = v2["bscr_net"].set_index("module").loc["BSCR", "scr"]
    assert brut >= net


def test_scr_v2_coherent(v2):
    s25 = v2["s25_v2"].set_index("poste")["montant"]
    attendu = (s25["Capital de solvabilité requis de base (BSCR)"] + s25["Risque opérationnel"]
               + s25["Capacité d'absorption des provisions techniques"]
               + s25["Capacité d'absorption des impôts différés"])
    assert s25["Capital de solvabilité requis (SCR)"] == pytest.approx(attendu)


def test_absorption_positive_sur_les_chocs_de_marche(v2):
    chocs = v2["chocs_euro"].set_index("choc")
    for nom in ("action", "immobilier", "spread"):
        assert chocs.loc[nom, "absorption"] > 0


def test_tvog_maximale_quand_les_garanties_sont_a_la_monnaie(v2):
    s = v2["sensibilite_tmg"].set_index("decalage_tmg")["tvog"]
    assert s.loc[0.02] > s.loc[0.0]
