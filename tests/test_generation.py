"""Tests de non-régression du générateur de données."""
import numpy as np
import pandas as pd
import pytest

from generer_donnees import generer
from scr_data.config import charger_hypotheses
from scr_data.courbe import Actualisation, SmithWilson, construire_courbe, courbe_smith_wilson


@pytest.fixture(scope="module")
def hyp():
    return charger_hypotheses()


@pytest.fixture(scope="module")
def sorties():
    return generer(ecrire=False)


def test_smith_wilson_reprice_les_swaps(hyp):
    # test de la mécanique Smith-Wilson, indépendamment de la source configurée
    courbe, _ = courbe_smith_wilson(hyp["courbe"])
    actu = Actualisation(courbe)
    cra = hyp["courbe"]["cra_bp"] / 1e4
    for m, r in hyp["courbe"]["taux_swap"].items():
        t = np.arange(1, int(m) + 1)
        prix = (r - cra) * actu.df(t).sum() + actu.df(int(m))
        assert prix == pytest.approx(1.0, abs=1e-8)


def test_convergence_vers_ufr(hyp):
    _, alpha = courbe_smith_wilson(hyp["courbe"])
    sw = SmithWilson(list(hyp["courbe"]["taux_swap"]),
                     [r - hyp["courbe"]["cra_bp"] / 1e4 for r in hyp["courbe"]["taux_swap"].values()],
                     hyp["courbe"]["ufr"], alpha)
    ecart = abs(sw.forward_instantane(hyp["courbe"]["point_convergence"]) - np.log1p(hyp["courbe"]["ufr"]))
    assert ecart <= hyp["courbe"]["tolerance_convergence_bp"] / 1e4 + 1e-9


def test_aucune_alerte_de_calibrage(sorties):
    ctrl = sorties["controles"]
    assert (ctrl["statut"] != "ALERTE").all(), ctrl[ctrl["statut"] == "ALERTE"]


def test_bilan_equilibre_actif(sorties, hyp):
    bilan = sorties["bilan_prudentiel_indicatif"]
    total_actif = bilan.loc[bilan["cote"] == "Actif", "montant"].sum()
    assert total_actif == pytest.approx(sum(hyp["bilan_cible"]["actif"].values()), rel=1e-9)


def test_reproductibilite():
    a = generer(ecrire=False)["inventaire_actifs"]
    b = generer(ecrire=False)["inventaire_actifs"]
    pd.testing.assert_frame_equal(a, b)


def test_triangles_bien_formes(sorties, hyp):
    tri = sorties["triangles_reglements"]
    n = hyp["non_vie"]["nb_annees"]
    for _, sous in tri.groupby("lob"):
        assert len(sous) == n * (n + 1) // 2
        assert (sous["reglements_incrementaux"] >= 0).all()


def test_courbe_eiopa_officielle(hyp):
    """La courbe configurée est bien celle publiée par l'EIOPA au 31/12/2025."""
    courbe, info = construire_courbe(hyp)
    assert info["source"] == "EIOPA"
    spot = courbe.set_index("maturite")["taux_spot"]
    assert spot.loc[1] == pytest.approx(0.02076, abs=1e-6)
    assert spot.loc[20] == pytest.approx(0.03209, abs=1e-6)
    assert abs(spot.loc[150] - hyp["courbe"]["ufr"]) < 0.002       # convergence vers l'UFR


def test_facteurs_choc_officiels_conformes_au_repli(hyp):
    """Le fichier EIOPA et les tables de repli coïncident aux maturités de référence."""
    import numpy as np
    from calculer_scr import charger_params
    from scr.marche import facteurs_choc_taux

    params = charger_params()
    maturites = np.array(sorted(params["taux"]["facteurs_hausse"]), dtype=float)
    for sens in ("hausse", "baisse"):
        officiels = facteurs_choc_taux(params, sens, maturites)
        repli = np.array([params["taux"][f"facteurs_{sens}"][int(m)] for m in maturites])
        assert np.allclose(officiels, repli)
