"""Tests de la projection ORSA."""
import numpy as np
import pytest

import calculer_v5
from scr.orsa import besoin_global_solvabilite
from scr_data.config import charger_hypotheses


@pytest.fixture(scope="module")
def hyp():
    return charger_hypotheses()


@pytest.fixture(scope="module")
def v5():
    return calculer_v5.calculer(ecrire=False)


def test_bilan_equilibre_a_chaque_pas(v5):
    """Actif = passif + fonds propres, par construction du bouclage."""
    t = v5["trajectoires"]
    assert (t["actif"] - t["be_total"] - t["fonds_propres"]).abs().max() < 1e-6


def test_trajectoire_centrale_stable(v5):
    centrale = v5["trajectoires"].query("scenario == 'central'")
    assert len(centrale) == 5
    assert centrale["ratio"].between(1.5, 3.5).all()
    assert (centrale["resultat_net"] > 0).all()


def test_les_scenarios_adverses_degradent_le_ratio(v5):
    synthese = v5["synthese"].set_index("scenario")
    central = synthese.loc["Scénario central", "ratio_minimum"]
    for nom in synthese.index:
        if "Baisse des taux" in nom or nom == "Scénario central":
            continue
        assert synthese.loc[nom, "ratio_minimum"] <= central + 1e-9


def test_crise_combinee_est_le_pire_scenario(v5):
    synthese = v5["synthese"].set_index("scenario")
    pire = synthese["ratio_minimum"].idxmin()
    assert "Crise combinée" in pire


def test_stress_inverses_orientes(v5):
    inverses = v5["stress_inverses"].set_index("stress")
    assert inverses.loc["Choc action instantané", "amplitude_critique"] < 0
    assert inverses.loc["Hausse des taux", "amplitude_critique"] > 0
    assert inverses.loc["Sinistre exceptionnel net", "amplitude_critique"] > 0


def test_besoin_global_superieur_au_scr(hyp):
    bgs = besoin_global_solvabilite(800.0, hyp).set_index("composante")["montant"]
    assert bgs["Besoin global de solvabilité"] > 800.0
    complements = sum(v for k, v in hyp["orsa"]["besoin_global"].items() if k != "diversification")
    attendu = 800.0 + complements * (1 - hyp["orsa"]["besoin_global"]["diversification"])
    assert bgs["Besoin global de solvabilité"] == pytest.approx(attendu)
