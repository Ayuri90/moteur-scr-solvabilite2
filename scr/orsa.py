"""ORSA — projection pluriannuelle du bilan, du SCR et du ratio de couverture (V5).

Le point de départ est le bilan de la V2 (BE stochastique, SCR net d'absorption). La
projection applique ensuite, année par année :
- le plan d'affaires (collecte, primes, ratio combiné, frais) ;
- les rendements financiers du scénario retenu ;
- les chocs propres aux scénarios adverses (krach, taux, rachats, sinistre, inflation) ;
- un SCR projeté par inducteurs de volume, agrégé chaque année par la matrice de l'annexe IV.

C'est la simplification usuelle en ORSA : le SCR n'est pas recalculé intégralement à chaque
pas, il est mis à l'échelle des volumes qui le portent, puis réagrégé.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scr import agregation


@dataclass
class EtatBilan:
    placements_hors_uc: float
    actifs_uc: float
    autres_actifs: float
    be_epargne_euro: float
    be_uc: float
    be_autres_vie: float
    be_non_vie: float
    be_sante: float
    autres_passifs: float
    part_actions: float
    part_immobilier: float
    part_obligations: float
    part_monetaire: float

    @property
    def actif_total(self) -> float:
        return self.placements_hors_uc + self.actifs_uc + self.autres_actifs

    @property
    def passif_total(self) -> float:
        return (self.be_epargne_euro + self.be_uc + self.be_autres_vie + self.be_non_vie
                + self.be_sante + self.autres_passifs)

    @property
    def fonds_propres(self) -> float:
        return self.actif_total - self.passif_total


def etat_initial(be_segments: dict[str, float], hyp: dict[str, Any],
                 inventaire: pd.DataFrame) -> EtatBilan:
    """Bilan de départ : inventaire d'actifs réel et best estimates du calcul V2."""
    euro = inventaire[inventaire["portefeuille"] == "euro"]
    total = float(euro["valeur_marche"].sum())
    familles = {"actions": ["action"], "immobilier": ["immobilier"],
                "obligations": ["obligation", "pret_infrastructure"],
                "monetaire": ["monetaire", "depot"]}
    parts = {nom: float(euro.loc[euro["classe"].isin(classes), "valeur_marche"].sum()) / total
             for nom, classes in familles.items()}
    bc = hyp["bilan_cible"]
    return EtatBilan(
        placements_hors_uc=total,
        actifs_uc=float(inventaire.loc[inventaire["portefeuille"] == "uc", "valeur_marche"].sum()),
        autres_actifs=bc["actif"]["provisions_cedees"] + bc["actif"]["creances_tresorerie_autres"],
        be_epargne_euro=be_segments["epargne_euro"],
        be_uc=be_segments["epargne_uc"],
        be_autres_vie=be_segments["temporaire_deces"] + be_segments["rentes"],
        be_non_vie=be_segments["non_vie_sinistres"] + be_segments["non_vie_primes"],
        be_sante=be_segments["sante_slt"] + be_segments["sante_nslt"],
        autres_passifs=(bc["passif"]["marge_risque"] + bc["passif"]["impots_differes_passifs"]
                        + bc["passif"]["autres_passifs"]),
        part_actions=parts["actions"], part_immobilier=parts["immobilier"],
        part_obligations=parts["obligations"], part_monetaire=parts["monetaire"])


# ---------------------------------------------------------------------------
class ProjectionORSA:
    def __init__(self, etat: EtatBilan, modules_initiaux: dict[str, float], scr_op: float,
                 lac_tp: float, lac_dt: float, hyp: dict[str, Any], params: dict[str, Any],
                 taux_sans_risque: float):
        self.etat0 = etat
        self.modules0 = dict(modules_initiaux)
        self.scr_op0 = scr_op
        self.lac_tp0 = lac_tp
        self.lac_dt0 = lac_dt
        self.hyp, self.params = hyp, params
        self.taux = taux_sans_risque
        self.p = hyp["orsa"]
        self.bscr0, _ = agregation.bscr(self.modules0, params)

    def scr_initial(self) -> float:
        """Contrôle : reconstitution du SCR de départ avec les mêmes briques que la projection."""
        return self.bscr0 + self.scr_op0 + self.lac_tp0 + self.lac_dt0

    # -- inducteurs de volume ---------------------------------------------
    def _modules_projetes(self, etat: EtatBilan, primes_non_vie: float, primes_sante: float,
                          volumes0: dict[str, float]) -> dict[str, float]:
        sensibilite_actions = 0.3       # part du SCR marché portée par les actions et l'immobilier
        mix = ((etat.part_actions + etat.part_immobilier)
               / volumes0["part_risquee"]) if volumes0["part_risquee"] else 1.0
        inducteurs = {
            "marche": (etat.placements_hors_uc / volumes0["placements"]
                       * ((1 - sensibilite_actions) + sensibilite_actions * mix)),
            "defaut": 1.0,
            "vie": (etat.be_epargne_euro + etat.be_autres_vie + 0.3 * etat.be_uc) / volumes0["be_vie"],
            "sante": primes_sante / volumes0["primes_sante"],
            "non_vie": primes_non_vie / volumes0["primes_non_vie"],
        }
        return {m: self.modules0[m] * inducteurs[m] for m in self.modules0}

    def projeter(self, scenario: dict[str, Any] | None = None) -> pd.DataFrame:
        etat = EtatBilan(**vars(self.etat0))
        plan = self.p["plan_affaires"]
        rendements = self.p["rendements_centraux"]
        scenario = scenario or {}
        volumes0 = {"placements": etat.placements_hors_uc,
                    "part_risquee": etat.part_actions + etat.part_immobilier,
                    "be_vie": etat.be_epargne_euro + etat.be_autres_vie + 0.3 * etat.be_uc,
                    "primes_non_vie": sum(l["primes_n"] for l in self.hyp["non_vie"]["lignes"].values()),
                    "primes_sante": self.hyp["sante_nslt"]["primes_n"]}
        taux_court = self.taux
        lignes = []
        for annee in range(1, self.p["horizon"] + 1):
            declenche = scenario.get("annee", 0) == annee
            actif_debut = etat.placements_hors_uc
            fonds_propres_debut = etat.fonds_propres

            # --- environnement financier ---------------------------------
            if declenche and "choc_taux" in scenario:
                taux_court += scenario["choc_taux"]
            rendement_obligataire = taux_court
            rendement_action = taux_court + rendements["prime_risque_action"]
            rendement_immobilier = taux_court + rendements["prime_risque_immobilier"]
            if declenche and "choc_action" in scenario:
                rendement_action += scenario["choc_action"]
                rendement_immobilier += 0.5 * scenario["choc_action"]
            elif scenario.get("choc_action") and annee > scenario.get("annee", 0):
                rendement_action += scenario.get("recuperation_annuelle", 0.0)
            if declenche and "choc_taux" in scenario and scenario["choc_taux"] > 0:
                # perte de valeur instantanée du portefeuille obligataire
                rendement_obligataire -= 7.0 * scenario["choc_taux"]
            elif declenche and "choc_taux" in scenario:
                rendement_obligataire -= 7.0 * scenario["choc_taux"]

            rendements_classe = {"part_obligations": rendement_obligataire,
                                 "part_actions": rendement_action,
                                 "part_immobilier": rendement_immobilier,
                                 "part_monetaire": taux_court}
            rendement_global = (etat.part_obligations * rendement_obligataire
                                + etat.part_actions * rendement_action
                                + etat.part_immobilier * rendement_immobilier
                                + etat.part_monetaire * taux_court)

            # déformation de l'allocation sous l'effet des performances relatives
            for attribut, rendement in rendements_classe.items():
                setattr(etat, attribut, getattr(etat, attribut) * (1 + rendement)
                        / (1 + rendement_global))

            # --- activité vie ---------------------------------------------
            collecte_euro = plan["collecte_euro"] * (1 + plan["croissance_collecte_euro"]) ** (annee - 1)
            collecte_uc = plan["collecte_uc"] * (1 + plan["croissance_collecte_uc"]) ** (annee - 1)
            sorties = plan["taux_sortie_euro"]
            if scenario.get("rachats_supplementaires") and \
                    scenario.get("annee", 0) <= annee < scenario.get("annee", 0) + scenario.get("duree_rachats", 1):
                sorties += scenario["rachats_supplementaires"]
            taux_servi = max(0.0, rendement_global - rendements["marge_financiere_conservee"])
            be_euro_debut = etat.be_epargne_euro
            etat.be_epargne_euro = be_euro_debut * (1 + taux_servi - sorties) + collecte_euro
            etat.be_uc = etat.be_uc * (1 + rendement_action - plan["taux_sortie_uc"]) + collecte_uc
            etat.actifs_uc = etat.be_uc

            # --- activité non-vie et santé --------------------------------
            primes_non_vie = volumes0["primes_non_vie"] * (1 + plan["croissance_primes_non_vie"]) ** annee
            primes_sante = volumes0["primes_sante"] * (1 + plan["croissance_primes_sante"]) ** annee
            resultat_non_vie = primes_non_vie * (1 - plan["ratio_combine_non_vie"])
            resultat_sante = primes_sante * (1 - plan["ratio_combine_sante"])
            sinistre = scenario.get("sinistre_net", 0.0) if declenche else 0.0
            etat.be_non_vie *= (1 + plan["croissance_primes_non_vie"])
            etat.be_sante *= (1 + plan["croissance_primes_sante"])

            # --- compte de résultat ---------------------------------------
            frais = plan["frais_generaux"] * (1 + plan["croissance_frais"]) ** (annee - 1)
            if scenario.get("multiplicateur_frais") and annee >= scenario.get("annee", 0):
                frais *= scenario["multiplicateur_frais"]
            produits_financiers = rendement_global * actif_debut
            credit_assures = taux_servi * be_euro_debut
            marge_uc = self.hyp["frais"]["gestion_pm_uc"] * etat.be_uc
            resultat_brut = (produits_financiers - credit_assures + marge_uc
                             + resultat_non_vie + resultat_sante - frais - sinistre)
            impot = max(0.0, resultat_brut) * self.p["taux_impot"]
            resultat_net = resultat_brut - impot
            dividende = max(0.0, resultat_net) * self.p["taux_distribution"]

            # --- bouclage du bilan : l'actif suit le passif et les fonds propres
            fonds_propres = fonds_propres_debut + resultat_net - dividende
            passif_hors_uc = (etat.be_epargne_euro + etat.be_autres_vie + etat.be_non_vie
                              + etat.be_sante + etat.autres_passifs)
            etat.placements_hors_uc = passif_hors_uc + fonds_propres - etat.autres_actifs

            # --- SCR et ratio ------------------------------------------------
            modules = self._modules_projetes(etat, primes_non_vie, primes_sante, volumes0)
            bscr, _ = agregation.bscr(modules, self.params)
            echelle = bscr / self.bscr0
            scr_op = self.scr_op0 * primes_non_vie / volumes0["primes_non_vie"]
            lac_tp = self.lac_tp0 * echelle          # absorption supposée proportionnelle au BSCR
            plafond = self.params["fiscalite"]["taux_impot"] * (bscr + scr_op + lac_tp)
            lac_dt = -min(plafond, abs(self.lac_dt0) * echelle)
            scr = bscr + scr_op + lac_tp + lac_dt
            lignes.append({"annee": annee, "actif": etat.actif_total, "be_total": etat.passif_total,
                           "fonds_propres": fonds_propres, "bscr": bscr, "scr": scr,
                           "ratio": fonds_propres / scr, "resultat_net": resultat_net,
                           "dividende": dividende, "taux_servi": taux_servi,
                           "rendement_actif": rendement_global,
                           "collecte_nette_euro": etat.be_epargne_euro - be_euro_debut})
        return pd.DataFrame(lignes)


def besoin_global_solvabilite(scr: float, hyp: dict[str, Any]) -> pd.DataFrame:
    """Besoin global de solvabilité : SCR augmenté des risques mal captés par la formule standard."""
    p = hyp["orsa"]["besoin_global"]
    complements = {k: v for k, v in p.items() if k != "diversification"}
    somme = sum(complements.values())
    net = somme * (1 - p["diversification"])
    lignes = [{"composante": "SCR formule standard", "montant": scr}]
    lignes += [{"composante": k.replace("_", " "), "montant": v} for k, v in complements.items()]
    lignes += [{"composante": "Diversification des compléments", "montant": net - somme},
               {"composante": "Besoin global de solvabilité", "montant": scr + net}]
    return pd.DataFrame(lignes)
