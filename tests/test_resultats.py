"""Tests simples : les méthodes retrouvent (ou non) l'effet vrai connu par simulation."""
import pandas as pd
import pytest

from src import analyse, simulation


@pytest.mark.parametrize("effet_vrai", [0.0, 0.5])
def test_double_ml_retrouve_effet_vrai(effet_vrai):
    df = simulation.simuler_glaces_noyades(n=5000, effet=effet_vrai, seed=123)
    resultat = analyse.effet_double_ml(df)
    assert abs(resultat["effet"] - effet_vrai) < 0.05


def test_ols_naif_est_biaise():
    df = simulation.simuler_glaces_noyades(n=5000, effet=0.0, seed=123)
    resultat = analyse.effet_ols(df, ["glaces"])
    assert abs(resultat["effet"] - 0.0) > 0.1


def test_regression_fallacieuse_souvent_significative_en_niveaux():
    resultats_mc = analyse.monte_carlo_fallacieux(n_rep=500, n=100, derive=0.5, seed=0)
    taux = analyse.taux_significatifs(resultats_mc)
    assert taux["niveaux"] > 50


def test_cointegration_distingue_les_deux_paires():
    paire_fallacieuse = simulation.simuler_marches_aleatoires(n=100, derive=0.5, seed=42)
    paire_cointegree = simulation.simuler_paire_cointegree(n=100, derive=0.5, seed=7)
    assert analyse.test_cointegration(paire_cointegree["y"].values, paire_cointegree["x"].values) < 0.05
    assert analyse.test_cointegration(paire_fallacieuse["y"].values, paire_fallacieuse["x"].values) > 0.05


def test_erreur_de_mesure_fait_revenir_le_biais():
    df = simulation.simuler_glaces_noyades(n=5000, effet=0.0, seed=123)
    resultats = analyse.biais_erreur_mesure(df, sigmas=[0, 6])
    assert abs(resultats["dml"].iloc[0]) < 0.05
    assert resultats["dml"].iloc[1] > 0.1


def test_collider_cree_un_biais():
    df = simulation.simuler_glaces_noyades(n=5000, effet=0.0, seed=123)
    tableau = analyse.tableau_mauvais_controle(df, effet_vrai=0.0).set_index("Méthode")
    # Sans le collider, l'IC du Double ML contient 0 ; avec, il est entièrement négatif
    assert tableau.loc["Double ML (T)", "IC 95 % bas"] < 0 < tableau.loc["Double ML (T)", "IC 95 % haut"]
    assert tableau.loc["Double ML (T + presse)", "IC 95 % haut"] < 0


def test_importance_des_glaces_malgre_effet_nul():
    df = simulation.ajouter_temperature_mesuree(simulation.simuler_glaces_noyades(effet=0.0), sigma=3)
    importances = analyse.importances_permutation(df, ["glaces", "temp_mesuree"])
    assert importances["glaces"] > importances["temp_mesuree"]


def test_resume_monte_carlo_calcule_biais_et_couverture():
    mc = pd.DataFrame({"n": 100, "methode": "A", "effet": [0.1, 0.3],
                       "ic_bas": [-0.1, 0.2], "ic_haut": [0.2, 0.4]})
    resume = analyse.resume_monte_carlo(mc, effet_vrai=0.0)
    assert resume.loc[0, "biais"] == pytest.approx(0.2)
    assert resume.loc[0, "couverture"] == pytest.approx(50)
    assert resume.loc[0, "largeur_ic"] == pytest.approx(0.25)


def test_intervention_le_modele_predictif_se_trompe():
    resultat = analyse.prevoir_intervention(effet=0.0, facteur=0.5)
    # En réalité la taxe ne change rien, mais le Gradient Boosting prédit une forte baisse
    assert abs(resultat["Réalité après la taxe"] - resultat["Noyades observées"]) < 1e-9
    assert resultat["Prévision du Gradient Boosting"] < 0.8 * resultat["Noyades observées"]
    assert abs(resultat["Prévision du Double ML"] - resultat["Réalité après la taxe"]) < 0.5


@pytest.mark.parametrize("effet_vrai", [0.0, 0.5])
def test_experience_aleatoire_ols_naif_sans_biais(effet_vrai):
    df = simulation.simuler_experience_aleatoire(effet=effet_vrai)
    assert abs(analyse.effet_ols(df, ["glaces"])["effet"] - effet_vrai) < 0.05


@pytest.mark.parametrize("effet_vrai", [0.0, 0.5])
def test_variable_instrumentale(effet_vrai):
    bon = analyse.effet_iv(simulation.simuler_instrument(effet=effet_vrai))
    invalide = analyse.effet_iv(simulation.simuler_instrument(effet=effet_vrai, effet_greve_noyades=1.0))
    assert abs(bon["effet"] - effet_vrai) < 0.05
    assert bon["f_premiere_etape"] > 10
    # Instrument invalide : l'IC exclut la vraie valeur
    assert not (invalide["ic_bas"] <= effet_vrai <= invalide["ic_haut"])


def test_mediateur_effet_total_et_direct():
    df = simulation.simuler_mediateur()
    total = analyse.effet_ols(df, ["temperature"], traitement="temperature")
    direct = analyse.effet_ols(df, ["temperature", "frequentation"], traitement="temperature")
    assert abs(total["effet"] - 0.3) < 0.05
    assert abs(direct["effet"] - 0.1) < 0.05


def test_causalite_inverse_signe_inverse_sauf_iv():
    df = analyse.ajouter_temperature_carre(simulation.simuler_causalite_inverse(effet=-0.2))
    naif = analyse.effet_ols(df, ["maitres_nageurs"], traitement="maitres_nageurs")
    iv = analyse.effet_iv(df, traitement="maitres_nageurs", instrument="dotation",
                          controles=["temperature", "temperature2"])
    assert naif["effet"] > 0
    assert iv["ic_bas"] <= -0.2 <= iv["ic_haut"]
    assert iv["ic_haut"] < 0
