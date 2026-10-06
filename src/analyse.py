"""Estimations : régressions OLS, tests de stationnarité, Monte Carlo, machine learning et Double ML."""
import doubleml as dml
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, coint

from . import simulation


# ---------------------------------------------------------------------------
# Partie 1 : régression fallacieuse
# ---------------------------------------------------------------------------

def regression_ols(y: np.ndarray, x: np.ndarray) -> dict:
    """Régression simple y = a + b·x : renvoie pente, t-stat, p-value, R², Durbin-Watson et résidus."""
    modele = sm.OLS(y, sm.add_constant(x)).fit()
    return {
        "pente": modele.params[1],
        "t_stat": modele.tvalues[1],
        "p_value": modele.pvalues[1],
        "r2": modele.rsquared,
        "durbin_watson": durbin_watson(modele.resid),
        "residus": modele.resid,
    }


def test_adf(serie: np.ndarray, regression: str = "ct") -> float:
    """P-value du test ADF (H0 : racine unitaire, donc série non stationnaire)."""
    return adfuller(serie, regression=regression, result_object=False)[1]


def tableau_regression_fallacieuse(donnees: pd.DataFrame) -> pd.DataFrame:
    """Compare la régression y ~ x en niveaux et en différences premières."""
    niveaux = regression_ols(donnees["y"].values, donnees["x"].values)
    diff = regression_ols(np.diff(donnees["y"].values), np.diff(donnees["x"].values))
    lignes = []
    for nom, res in [("Niveaux", niveaux), ("Différences premières", diff)]:
        lignes.append({"Spécification": nom, "Pente": res["pente"], "t-stat": res["t_stat"],
                       "p-value": res["p_value"], "R²": res["r2"], "Durbin-Watson": res["durbin_watson"]})
    return pd.DataFrame(lignes)


def tableau_adf(donnees: pd.DataFrame) -> pd.DataFrame:
    """P-values ADF en niveaux (constante + tendance) et en différences (constante)."""
    lignes = []
    for nom in ["x", "y"]:
        serie = donnees[nom].values
        lignes.append({"Série": nom,
                       "p-value ADF niveaux": test_adf(serie, "ct"),
                       "p-value ADF différences": test_adf(np.diff(serie), "c")})
    return pd.DataFrame(lignes)


def monte_carlo_fallacieux(n_rep: int = 500, n: int = 100, derive: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Répète la régression sur des marches indépendantes : t-stats et p-values en niveaux et en différences."""
    rng = np.random.default_rng(seed)
    lignes = []
    for _ in range(n_rep):
        x = np.cumsum(derive + rng.normal(0, 1, n))
        y = np.cumsum(derive + rng.normal(0, 1, n))
        niv = sm.OLS(y, sm.add_constant(x)).fit()
        dif = sm.OLS(np.diff(y), sm.add_constant(np.diff(x))).fit()
        lignes.append({"t_niveaux": niv.tvalues[1], "p_niveaux": niv.pvalues[1],
                       "t_differences": dif.tvalues[1], "p_differences": dif.pvalues[1]})
    return pd.DataFrame(lignes)


def taux_significatifs(resultats_mc: pd.DataFrame, seuil: float = 0.05) -> dict:
    """Pourcentage de régressions « significatives » au seuil donné, en niveaux et en différences."""
    return {"niveaux": 100 * (resultats_mc["p_niveaux"] < seuil).mean(),
            "differences": 100 * (resultats_mc["p_differences"] < seuil).mean()}


def taux_selon_n(tailles: list, derive: float, n_rep: int = 300, seed: int = 0) -> pd.DataFrame:
    """Taux de faux positifs (en %) selon la taille d'échantillon."""
    lignes = []
    for i, n in enumerate(tailles):
        taux = taux_significatifs(monte_carlo_fallacieux(n_rep, n, derive, seed + i))
        lignes.append({"n": n, "niveaux": taux["niveaux"], "differences": taux["differences"]})
    return pd.DataFrame(lignes)


def test_cointegration(y: np.ndarray, x: np.ndarray) -> float:
    """P-value du test d'Engle-Granger (H0 : pas de cointégration entre y et x)."""
    return coint(y, x)[1]


def tableau_cointegration(paire_fallacieuse: pd.DataFrame, paire_cointegree: pd.DataFrame) -> pd.DataFrame:
    """Régression en niveaux et test d'Engle-Granger pour une paire fallacieuse et une paire cointégrée."""
    lignes = []
    for nom, paire in [("Paire indépendante (fallacieuse)", paire_fallacieuse), ("Paire cointégrée", paire_cointegree)]:
        res = regression_ols(paire["y"].values, paire["x"].values)
        lignes.append({"Paire": nom, "Pente en niveaux": res["pente"], "R²": res["r2"],
                       "p-value Engle-Granger": test_cointegration(paire["y"].values, paire["x"].values)})
    return pd.DataFrame(lignes)


def taux_rejet_cointegration(n_rep: int = 300, n: int = 100, seed: int = 0) -> dict:
    """Pourcentage de paires déclarées cointégrées (p < 0.05), pour des paires cointégrées et indépendantes."""
    rejets_coint, rejets_indep = 0, 0
    for i in range(n_rep):
        paire = simulation.simuler_paire_cointegree(n=n, seed=seed + i)
        rejets_coint += test_cointegration(paire["y"].values, paire["x"].values) < 0.05
        paire = simulation.simuler_marches_aleatoires(n=n, seed=seed + n_rep + i)
        rejets_indep += test_cointegration(paire["y"].values, paire["x"].values) < 0.05
    return {"cointegrees": 100 * rejets_coint / n_rep, "independantes": 100 * rejets_indep / n_rep}


# ---------------------------------------------------------------------------
# Partie 2 : le confondeur
# ---------------------------------------------------------------------------

def ajouter_temperature_carre(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne temperature² (forme fonctionnelle du vrai modèle)."""
    return df.assign(temperature2=df["temperature"] ** 2)


def effet_gradient_boosting(df: pd.DataFrame, variables: list, seed: int = 0, traitement: str = "glaces",
                            cible: str = "noyades") -> dict:
    """Gradient Boosting de `cible` sur `variables` : R² en validation croisée et effet implicite du traitement.

    Effet implicite = moyenne de prédiction(traitement + 1) - prédiction(traitement), les autres variables inchangées.
    """
    X, y = df[variables], df[cible]
    modele = GradientBoostingRegressor(random_state=seed)
    r2_cv = cross_val_score(modele, X, y, cv=KFold(5, shuffle=True, random_state=seed), scoring="r2").mean()
    modele.fit(X, y)
    X_plus_un = X.copy()
    X_plus_un[traitement] = X_plus_un[traitement] + 1
    effet = np.mean(modele.predict(X_plus_un) - modele.predict(X))
    return {"effet": effet, "ic_bas": np.nan, "ic_haut": np.nan, "r2_cv": r2_cv, "modele": modele}


def effet_ols(df: pd.DataFrame, variables: list, traitement: str = "glaces", cible: str = "noyades") -> dict:
    """OLS de `cible` sur `variables` (qui contient `traitement`) : coefficient du traitement et IC à 95 %."""
    modele = sm.OLS(df[cible], sm.add_constant(df[variables])).fit()
    ic = modele.conf_int().loc[traitement]
    return {"effet": modele.params[traitement], "ic_bas": ic.iloc[0], "ic_haut": ic.iloc[1]}


def effet_double_ml(df: pd.DataFrame, controles: list = None, n_estimators: int = 200,
                    min_samples_leaf: int = 5, seed: int = 0, traitement: str = "glaces",
                    cible: str = "noyades") -> dict:
    """Double ML (modèle partiellement linéaire) avec des forêts aléatoires et 5 folds : effet du traitement et IC à 95 %.

    `controles` : variables de contrôle données au modèle (par défaut la température).
    """
    if controles is None:
        controles = ["temperature"]
    donnees = dml.DoubleMLData(df, y_col=cible, d_cols=traitement, x_cols=controles)
    # Le calcul parallèle ne vaut le coût de lancement des processus que sur les grands échantillons
    n_jobs = -1 if len(df) >= 2000 else 1
    foret_y = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                    random_state=seed, n_jobs=n_jobs)
    foret_d = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                    random_state=seed, n_jobs=n_jobs)
    np.random.seed(seed)  # DoubleML tire le découpage en folds avec le générateur global de numpy
    modele = dml.DoubleMLPLR(donnees, foret_y, foret_d, n_folds=5)
    modele.fit()
    ic = modele.confint(level=0.95).loc[traitement]
    return {"effet": modele.coef[0], "ic_bas": ic.iloc[0], "ic_haut": ic.iloc[1]}


def tableau_comparatif(df: pd.DataFrame, effet_vrai: float, seed: int = 0) -> pd.DataFrame:
    """Compare les méthodes d'estimation de l'effet des glaces sur les noyades."""
    df = ajouter_temperature_carre(df)
    resultats = {
        "Gradient Boosting naïf": effet_gradient_boosting(df, ["glaces"], seed),
        "OLS naïf": effet_ols(df, ["glaces"]),
        "OLS + température + température²": effet_ols(df, ["glaces", "temperature", "temperature2"]),
        "Double ML (forêts aléatoires)": effet_double_ml(df, seed=seed),
        "Bonus : OLS + température (sans le carré)": effet_ols(df, ["glaces", "temperature"]),
        "Bonus : Gradient Boosting + température": effet_gradient_boosting(df, ["glaces", "temperature"], seed),
    }
    lignes = []
    for methode, res in resultats.items():
        lignes.append({"Méthode": methode, "Effet estimé": res["effet"], "IC 95 % bas": res["ic_bas"],
                       "IC 95 % haut": res["ic_haut"], "Écart à la vraie valeur": res["effet"] - effet_vrai})
    return pd.DataFrame(lignes)


def pentes_par_tranche(df: pd.DataFrame, n_tranches: int = 10) -> pd.DataFrame:
    """Pente de noyades sur glaces à l'intérieur de chaque tranche de température."""
    tranches = pd.qcut(df["temperature"], n_tranches, labels=False)
    lignes = []
    for k in range(n_tranches):
        sous_df = df[tranches == k]
        pente = np.polyfit(sous_df["glaces"], sous_df["noyades"], 1)[0]
        lignes.append({"tranche": k + 1, "temperature_min": sous_df["temperature"].min(),
                       "temperature_max": sous_df["temperature"].max(), "pente": pente})
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# Partie 3 : limites du contrôle du confondeur
# ---------------------------------------------------------------------------

def biais_erreur_mesure(df: pd.DataFrame, sigmas: list, n_estimators: int = 100) -> pd.DataFrame:
    """Effet estimé (Double ML et OLS + T + T²) quand la température est mesurée avec une erreur d'écart-type sigma."""
    lignes = []
    for sigma in sigmas:
        d = simulation.ajouter_temperature_mesuree(df, sigma)
        d = d.assign(temp_mesuree2=d["temp_mesuree"] ** 2)
        res_dml = effet_double_ml(d, controles=["temp_mesuree"], n_estimators=n_estimators)
        res_ols = effet_ols(d, ["glaces", "temp_mesuree", "temp_mesuree2"])
        lignes.append({"sigma": sigma, "dml": res_dml["effet"], "dml_ic_bas": res_dml["ic_bas"],
                       "dml_ic_haut": res_dml["ic_haut"], "ols": res_ols["effet"]})
    return pd.DataFrame(lignes)


def tableau_mauvais_controle(df: pd.DataFrame, effet_vrai: float) -> pd.DataFrame:
    """Compare l'OLS et le Double ML avec et sans le collider « presse » dans les contrôles."""
    df = ajouter_temperature_carre(simulation.ajouter_presse(df))
    resultats = {
        "OLS + T + T²": effet_ols(df, ["glaces", "temperature", "temperature2"]),
        "OLS + T + T² + presse": effet_ols(df, ["glaces", "temperature", "temperature2", "presse"]),
        "Double ML (T)": effet_double_ml(df, controles=["temperature"]),
        "Double ML (T + presse)": effet_double_ml(df, controles=["temperature", "presse"]),
    }
    lignes = []
    for methode, res in resultats.items():
        lignes.append({"Méthode": methode, "Effet estimé": res["effet"], "IC 95 % bas": res["ic_bas"],
                       "IC 95 % haut": res["ic_haut"], "Écart à la vraie valeur": res["effet"] - effet_vrai})
    return pd.DataFrame(lignes)


def importances_permutation(df: pd.DataFrame, variables: list, seed: int = 0) -> pd.Series:
    """Importance par permutation (baisse du R² sur un échantillon test) d'un Gradient Boosting qui prédit les noyades."""
    X_train, X_test, y_train, y_test = train_test_split(df[variables], df["noyades"], test_size=0.3, random_state=seed)
    modele = GradientBoostingRegressor(random_state=seed).fit(X_train, y_train)
    resultat = permutation_importance(modele, X_test, y_test, n_repeats=20, random_state=seed)
    return pd.Series(resultat.importances_mean, index=variables)


# ---------------------------------------------------------------------------
# Partie 3 : Monte Carlo des estimateurs
# ---------------------------------------------------------------------------

def monte_carlo_estimateurs(n_rep: int, n: int = 5000, effet: float = 0.0, seed: int = 10_000,
                            n_estimators: int = 100) -> pd.DataFrame:
    """Répète la simulation et l'estimation n_rep fois : effet estimé et IC à 95 % de chaque méthode."""
    lignes = []
    for r in range(n_rep):
        df = ajouter_temperature_carre(simulation.simuler_glaces_noyades(n=n, effet=effet, seed=seed + r))
        resultats = {
            "OLS naïf": effet_ols(df, ["glaces"]),
            "OLS + T + T²": effet_ols(df, ["glaces", "temperature", "temperature2"]),
            "OLS + T (sans le carré)": effet_ols(df, ["glaces", "temperature"]),
            "Double ML": effet_double_ml(df, n_estimators=n_estimators, seed=r),
        }
        for methode, res in resultats.items():
            lignes.append({"n": n, "methode": methode, "effet": res["effet"],
                           "ic_bas": res["ic_bas"], "ic_haut": res["ic_haut"]})
    return pd.DataFrame(lignes)


def resume_monte_carlo(mc: pd.DataFrame, effet_vrai: float) -> pd.DataFrame:
    """Biais moyen, écart-type, largeur moyenne de l'IC et couverture (% d'IC contenant la vraie valeur)."""
    mc = mc.assign(couvre=(mc["ic_bas"] <= effet_vrai) & (mc["ic_haut"] >= effet_vrai),
                   largeur_ic=mc["ic_haut"] - mc["ic_bas"])
    resume = mc.groupby(["n", "methode"], sort=False).agg(
        biais=("effet", "mean"), ecart_type=("effet", "std"),
        largeur_ic=("largeur_ic", "mean"), couverture=("couvre", "mean")).reset_index()
    resume["biais"] = resume["biais"] - effet_vrai
    resume["couverture"] = 100 * resume["couverture"]
    return resume


def monte_carlo_taille(tailles: list, n_rep: int, effet: float = 0.0, seed: int = 20_000) -> pd.DataFrame:
    """Monte Carlo des estimateurs pour plusieurs tailles d'échantillon."""
    return pd.concat([monte_carlo_estimateurs(n_rep, n=n, effet=effet, seed=seed + 1000 * i)
                      for i, n in enumerate(tailles)], ignore_index=True)


# ---------------------------------------------------------------------------
# Partie 4 : agir sur le monde (intervention, expérience aléatoire, instrument)
# ---------------------------------------------------------------------------

def prevoir_intervention(effet: float, facteur: float = 0.5, seed: int = 123) -> dict:
    """Taxe qui multiplie les ventes de glaces par `facteur` : noyades prédites par le GB et le Double ML, et réalité."""
    avant = simulation.simuler_glaces_noyades(effet=effet, seed=seed)
    apres = simulation.simuler_glaces_noyades(effet=effet, seed=seed, facteur_glaces=facteur)
    gb = GradientBoostingRegressor(random_state=0).fit(avant[["glaces"]], avant["noyades"])
    glaces_apres = avant[["glaces"]] * facteur
    effet_dml = effet_double_ml(avant)["effet"]
    return {
        "Effet vrai": effet,
        "Noyades observées": avant["noyades"].mean(),
        "Réalité après la taxe": apres["noyades"].mean(),
        "Prévision du Gradient Boosting": gb.predict(glaces_apres).mean(),
        "Prévision du Double ML": avant["noyades"].mean() + effet_dml * (glaces_apres["glaces"] - avant["glaces"]).mean(),
    }


def tableau_intervention(effets: list, facteur: float = 0.5) -> pd.DataFrame:
    """Prévisions et réalité après la taxe, pour plusieurs effets vrais."""
    return pd.DataFrame([prevoir_intervention(effet, facteur) for effet in effets])


def tableau_experience(effet_vrai: float) -> pd.DataFrame:
    """Compare l'estimation naïve sur données observées et sur une expérience aléatoire."""
    observees = simulation.simuler_glaces_noyades(effet=effet_vrai)
    experience = ajouter_temperature_carre(simulation.simuler_experience_aleatoire(effet=effet_vrai))
    resultats = {
        "OLS naïf (données observées)": effet_ols(observees, ["glaces"]),
        "OLS naïf (expérience)": effet_ols(experience, ["glaces"]),
        "OLS + T + T² (expérience)": effet_ols(experience, ["glaces", "temperature", "temperature2"]),
        "Gradient Boosting naïf (expérience)": effet_gradient_boosting(experience, ["glaces"]),
    }
    lignes = []
    for methode, res in resultats.items():
        lignes.append({"Méthode": methode, "Effet estimé": res["effet"], "IC 95 % bas": res["ic_bas"],
                       "IC 95 % haut": res["ic_haut"], "Écart à la vraie valeur": res["effet"] - effet_vrai})
    return pd.DataFrame(lignes)


def effet_iv(df: pd.DataFrame, traitement: str = "glaces", instrument: str = "greve", controles: list = None,
             cible: str = "noyades") -> dict:
    """Doubles moindres carrés (2SLS) : effet du traitement instrumenté, IC à 95 % et F de 1re étape.

    `controles` : variables exogènes ajoutées dans les deux étapes (pour la précision).
    """
    if controles is None:
        controles = []
    X = sm.add_constant(df[[traitement] + controles])
    Z = sm.add_constant(df[[instrument] + controles])
    modele = IV2SLS(df[cible], X, instrument=Z).fit()
    ic = modele.conf_int().loc[traitement]
    premiere_etape = sm.OLS(df[traitement], Z).fit()
    return {"effet": modele.params[traitement], "ic_bas": ic.iloc[0], "ic_haut": ic.iloc[1],
            "f_premiere_etape": premiere_etape.tvalues[instrument] ** 2}  # un seul instrument : F = t²


def tableau_instrument(effet_vrai: float) -> pd.DataFrame:
    """Température non observée : OLS naïf contre variable instrumentale (bonne, faible, invalide)."""
    donnees = {
        "OLS naïf": simulation.simuler_instrument(effet=effet_vrai),
        "IV : bon instrument": simulation.simuler_instrument(effet=effet_vrai),
        "IV : instrument faible": simulation.simuler_instrument(effet=effet_vrai, effet_greve_glaces=-2.0),
        "IV : instrument invalide": simulation.simuler_instrument(effet=effet_vrai, effet_greve_noyades=1.0),
    }
    lignes = []
    for methode, df in donnees.items():
        res = effet_ols(df, ["glaces"]) if methode == "OLS naïf" else effet_iv(df)
        lignes.append({"Méthode": methode, "Effet estimé": res["effet"], "IC 95 % bas": res["ic_bas"],
                       "IC 95 % haut": res["ic_haut"], "Écart à la vraie valeur": res["effet"] - effet_vrai,
                       "F 1re étape": res.get("f_premiere_etape", np.nan)})
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# Partie 5 : deux autres pièges (médiateur, causalité inverse)
# ---------------------------------------------------------------------------

def tableau_mediateur() -> pd.DataFrame:
    """Effet de la température sur les noyades, avec ou sans contrôle du médiateur (la fréquentation)."""
    df = simulation.simuler_mediateur()
    resultats = {
        "OLS sans contrôle": effet_ols(df, ["temperature"], traitement="temperature"),
        "OLS + fréquentation": effet_ols(df, ["temperature", "frequentation"], traitement="temperature"),
        "Double ML (contrôle = fréquentation)": effet_double_ml(df, controles=["frequentation"],
                                                                traitement="temperature"),
    }
    lignes = []
    for methode, res in resultats.items():
        lignes.append({"Méthode": methode, "Effet estimé": res["effet"], "IC 95 % bas": res["ic_bas"],
                       "IC 95 % haut": res["ic_haut"]})
    return pd.DataFrame(lignes)


def tableau_causalite_inverse(effet_vrai: float = -0.2) -> pd.DataFrame:
    """Effet des maîtres-nageurs sur les noyades quand les noyades causent aussi les maîtres-nageurs."""
    df = ajouter_temperature_carre(simulation.simuler_causalite_inverse(effet=effet_vrai))
    options = {"traitement": "maitres_nageurs"}
    resultats = {
        "OLS naïf": effet_ols(df, ["maitres_nageurs"], **options),
        "OLS + T + T²": effet_ols(df, ["maitres_nageurs", "temperature", "temperature2"], **options),
        "Double ML (contrôle = T)": effet_double_ml(df, **options),
        "Gradient Boosting naïf": effet_gradient_boosting(df, ["maitres_nageurs"], **options),
        "IV : dotation tirée au sort (+ T, T²)": effet_iv(df, instrument="dotation",
                                                          controles=["temperature", "temperature2"], **options),
    }
    lignes = []
    for methode, res in resultats.items():
        lignes.append({"Méthode": methode, "Effet estimé": res["effet"], "IC 95 % bas": res["ic_bas"],
                       "IC 95 % haut": res["ic_haut"], "Écart à la vraie valeur": res["effet"] - effet_vrai})
    return pd.DataFrame(lignes)
