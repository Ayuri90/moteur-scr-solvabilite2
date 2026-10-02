"""Tests du modèle interne partiel non-vie."""
import numpy as np
import pytest

import calculer_scr
from scr.modele_interne import (bootstrap_odp, copule, coupler, mesures_risque,
                                simuler_catastrophe, usp_reserve)
from scr.passif import chain_ladder
from scr_data.config import RACINE, charger_hypotheses


@pytest.fixture(scope="module")
def triangle():
    hyp = charger_hypotheses()
    d = calculer_scr.charger_donnees(RACINE / hyp["general"]["dossier_sortie"])
    sous = d["triangles_reglements"].query("lob == 'auto_rc'")
    return sous.pivot(index="annee_survenance", columns="developpement", values="reglements_cumules")


def test_bootstrap_centre_sur_chain_ladder(triangle):
    rng = np.random.default_rng(1)
    boot = bootstrap_odp(triangle, 800, rng)
    reference = chain_ladder(triangle)["reserves"].sum()
    assert boot["reserves"].mean() == pytest.approx(reference, rel=0.05)
    assert boot["reserves"].std(ddof=1) > 0


def test_traite_xl_ecrete_les_catastrophes():
    rng = np.random.default_rng(2)
    p = {"frequence_poisson": 1.0, "seuil_gpd": 8.0, "xi": 0.35, "beta": 12.0}
    brut = simuler_catastrophe(p, 20000, rng)
    net = simuler_catastrophe(p, 20000, np.random.default_rng(2), priorite=30.0, portee=250.0)
    assert np.quantile(net, 0.995) < np.quantile(brut, 0.995)
    assert net.mean() < brut.mean()


def test_copule_reproduit_la_correlation():
    rng = np.random.default_rng(3)
    M = np.array([[1.0, 0.5], [0.5, 1.0]])
    u = copule(M, 40000, rng, "gaussienne")
    from scipy.stats import spearmanr
    assert spearmanr(u[:, 0], u[:, 1]).statistic == pytest.approx(0.48, abs=0.05)


def test_student_a_des_queues_plus_epaisses():
    rng = np.random.default_rng(4)
    M = np.array([[1.0, 0.25], [0.25, 1.0]])
    gauss = copule(M, 60000, rng, "gaussienne")
    student = copule(M, 60000, np.random.default_rng(4), "student", 4)

    def dependance_de_queue(u, seuil=0.99):
        extreme = u[:, 0] > seuil
        return float((u[extreme, 1] > seuil).mean())

    assert dependance_de_queue(student) > dependance_de_queue(gauss)


def test_couplage_preserve_les_marginales():
    rng = np.random.default_rng(5)
    a, b = rng.lognormal(0, 0.3, 5000), rng.gamma(3, 2, 5000)
    u = copule(np.eye(2), 20000, rng, "gaussienne")
    couple = coupler([a, b], u)
    assert couple[:, 0].mean() == pytest.approx(a.mean(), rel=0.05)
    assert couple[:, 1].std() == pytest.approx(b.std(), rel=0.08)


def test_tvar_superieure_a_la_var():
    rng = np.random.default_rng(6)
    m = mesures_risque(rng.lognormal(3, 0.5, 50000))
    assert m["tvar"] > m["var"] > m["moyenne"]


def test_usp_encadre_par_les_deux_parametres():
    rng = np.random.default_rng(7)
    distribution = rng.lognormal(np.log(100), 0.06, 5000)
    u = usp_reserve(distribution, sigma_standard=0.09, nb_annees=10,
                    credibilite={10: 0.74, 15: 1.0})
    assert min(u["sigma_usp"], 0.09) <= u["sigma_retenu"] <= max(u["sigma_usp"], 0.09)
    assert u["credibilite"] == 0.74
