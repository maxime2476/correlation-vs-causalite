"""Figures du projet, sauvegardées dans le dossier figures/."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

DOSSIER_FIGURES = Path(__file__).resolve().parent.parent / "figures"

BLEU = "#2a78d6"
ORANGE = "#eb6834"
VERT = "#1baf7a"
GRIS = "#8a8984"
TEXTE = "#52514e"
ROUGE = "#e34948"
# Échelle séquentielle à une seule teinte pour la température (on enlève les orangés trop pâles)
CMAP_TEMPERATURE = ListedColormap(plt.cm.Oranges(np.linspace(0.3, 1, 256)))

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRIS,
    "axes.labelcolor": TEXTE,
    "xtick.color": TEXTE,
    "ytick.color": TEXTE,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "axes.titleweight": "bold",
    "legend.frameon": False,
})


def sauvegarder(fig, nom: str) -> None:
    """Enregistre la figure en PNG dans figures/."""
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    fig.savefig(DOSSIER_FIGURES / nom, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Partie 1 : régression fallacieuse
# ---------------------------------------------------------------------------

def figure_series_fallacieuses(donnees, r2: float):
    """Les deux marches aléatoires superposées, avec un titre volontairement absurde."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(donnees["x"], color=BLEU, lw=2, label="Ventes de parapluies au Sahara")
    ax.plot(donnees["y"], color=ORANGE, lw=2, label="Nombre de photos de chats publiées")
    ax.set_title(f"Les parapluies sahariens font-ils poster des chats ? (R² = {r2:.2f})")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Niveau (unités arbitraires)")
    ax.legend(loc="upper left")
    ax.text(0.99, 0.02, "Deux séries simulées indépendantes : aucun lien réel.",
            transform=ax.transAxes, ha="right", fontsize=9, color=TEXTE, style="italic")
    sauvegarder(fig, "p1_01_series_fallacieuses.png")
    return fig


def figure_nuages_niveaux_differences(donnees):
    """Nuage y ~ x en niveaux (lien apparent) et en différences (plus de lien)."""
    x, y = donnees["x"].values, donnees["y"].values
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (a, b, titre) in zip(axes, [(x, y, "En niveaux"), (np.diff(x), np.diff(y), "En différences premières")]):
        pente, constante = np.polyfit(a, b, 1)
        grille = np.linspace(a.min(), a.max(), 50)
        ax.scatter(a, b, s=18, color=BLEU, alpha=0.7)
        ax.plot(grille, constante + pente * grille, color=ORANGE, lw=2)
        ax.set_title(f"{titre} : corrélation = {np.corrcoef(a, b)[0, 1]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    sauvegarder(fig, "p1_02_nuages_niveaux_differences.png")
    return fig


def figure_residus(residus_niveaux, residus_differences, dw_niveaux: float, dw_differences: float):
    """Résidus de la régression en niveaux (très persistants) et en différences (bruit blanc)."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    axes[0].plot(residus_niveaux, color=BLEU, lw=1.5)
    axes[0].set_title(f"Résidus en niveaux (Durbin-Watson = {dw_niveaux:.2f})")
    axes[1].plot(np.arange(1, len(residus_differences) + 1), residus_differences, color=VERT, lw=1.5)
    axes[1].set_title(f"Résidus en différences (Durbin-Watson = {dw_differences:.2f})")
    axes[1].set_xlabel("Temps")
    for ax in axes:
        ax.axhline(0, color=GRIS, lw=1)
    fig.tight_layout()
    sauvegarder(fig, "p1_03_residus.png")
    return fig


def figure_histogramme_tstats(resultats_mc, taux: dict):
    """Distribution des t-stats sur les répétitions Monte Carlo, en niveaux et en différences."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panneaux = [(resultats_mc["t_niveaux"], BLEU, f"En niveaux : {taux['niveaux']:.0f} % significatifs"),
                (resultats_mc["t_differences"], VERT, f"En différences : {taux['differences']:.1f} % significatifs")]
    for ax, (t, couleur, titre) in zip(axes, panneaux):
        ax.hist(t, bins=35, color=couleur, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvspan(-1.96, 1.96, color=GRIS, alpha=0.15, label="Zone non significative (|t| < 1.96)")
        ax.set_title(titre)
        ax.set_xlabel("t-stat de la pente")
    axes[0].set_ylabel("Nombre de simulations")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{len(resultats_mc)} paires de marches aléatoires indépendantes", fontsize=11, color=TEXTE)
    fig.tight_layout()
    sauvegarder(fig, "p1_04_histogramme_tstats.png")
    return fig


def figure_faux_positifs_selon_n(taux_avec_derive, taux_sans_derive):
    """Taux de faux positifs selon la taille d'échantillon : plus de données n'aide pas en niveaux."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(taux_avec_derive["n"], taux_avec_derive["niveaux"], "o-", color=BLEU, lw=2, ms=7,
            label="Niveaux, avec dérive")
    ax.plot(taux_sans_derive["n"], taux_sans_derive["niveaux"], "s-", color=ORANGE, lw=2, ms=7,
            label="Niveaux, sans dérive")
    ax.plot(taux_sans_derive["n"], taux_sans_derive["differences"], "^-", color=VERT, lw=2, ms=7,
            label="Différences premières")
    ax.axhline(5, color=GRIS, ls="--", lw=1)
    ax.text(taux_sans_derive["n"].iloc[-1], 10, "5 % attendus", ha="right", color=TEXTE, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(taux_sans_derive["n"])
    ax.set_xticklabels(taux_sans_derive["n"])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Taille de l'échantillon (n)")
    ax.set_ylabel("% de régressions « significatives » à 5 %")
    ax.set_title("Plus de données ne corrige pas une régression fallacieuse")
    ax.legend(loc="center right")
    sauvegarder(fig, "p1_05_faux_positifs_selon_n.png")
    return fig


# ---------------------------------------------------------------------------
# Partie 2 : le confondeur
# ---------------------------------------------------------------------------

def figure_dag():
    """Schéma causal : température -> glaces, température -> noyades, glaces -> noyades barré."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    noeuds = {"Température": (0.5, 0.82), "Ventes de glaces": (0.15, 0.2), "Noyades": (0.85, 0.2)}
    for nom, (x, y) in noeuds.items():
        ax.text(x, y, nom, ha="center", va="center", fontsize=12, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f3f2ee", "edgecolor": GRIS})
    fleche = {"arrowstyle": "-|>", "color": BLEU, "lw": 2, "mutation_scale": 18, "shrinkA": 28, "shrinkB": 28}
    ax.annotate("", xy=noeuds["Ventes de glaces"], xytext=noeuds["Température"], arrowprops=fleche)
    ax.annotate("", xy=noeuds["Noyades"], xytext=noeuds["Température"], arrowprops=fleche)
    fleche_barree = dict(fleche, color=GRIS, linestyle="--", shrinkA=62, shrinkB=40)
    ax.annotate("", xy=noeuds["Noyades"], xytext=noeuds["Ventes de glaces"], arrowprops=fleche_barree)
    ax.text(0.5, 0.2, "✕", ha="center", va="center", fontsize=26, color=ROUGE, fontweight="bold")
    ax.text(0.5, 0.07, "aucun effet causal (effet vrai = 0)", ha="center", fontsize=10, color=TEXTE)
    ax.text(0.27, 0.55, "cause", ha="center", color=BLEU, fontsize=10, rotation=50)
    ax.text(0.73, 0.55, "cause", ha="center", color=BLEU, fontsize=10, rotation=-50)
    ax.set_title("La température est un confondeur")
    sauvegarder(fig, "p2_01_dag.png")
    return fig


def figure_relations_temperature(df):
    """La température explique à la fois les ventes de glaces et les noyades."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, colonne, titre in [(axes[0], "glaces", "Ventes de glaces"), (axes[1], "noyades", "Noyades")]:
        ax.scatter(df["temperature"], df[colonne], s=4, alpha=0.25, color=BLEU)
        ax.set_title(f"{titre} selon la température")
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel(titre)
    fig.tight_layout()
    sauvegarder(fig, "p2_02_relations_temperature.png")
    return fig


def figure_nuage_confondu(df):
    """Noyades selon les glaces, coloré par la température : la pente vient du gradient de couleur."""
    fig, ax = plt.subplots(figsize=(8, 5))
    points = ax.scatter(df["glaces"], df["noyades"], s=5, c=df["temperature"], cmap=CMAP_TEMPERATURE, alpha=0.7)
    fig.colorbar(points, ax=ax, label="Température (°C)")
    ax.set_xlabel("Ventes de glaces")
    ax.set_ylabel("Noyades")
    ax.set_title(f"Corrélation glaces-noyades = {df['glaces'].corr(df['noyades']):.2f}, effet causal = 0")
    sauvegarder(fig, "p2_03_nuage_confondu.png")
    return fig


def figure_prediction_gb(df, modele_gb, r2_cv: float, effet_implicite: float):
    """Courbe de prédiction du Gradient Boosting naïf : il prédit bien, mais la pente n'est pas causale."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["glaces"], df["noyades"], s=4, alpha=0.2, color=GRIS, label="Observations")
    grille = pd.DataFrame({"glaces": np.linspace(df["glaces"].min(), df["glaces"].max(), 400)})
    ax.plot(grille["glaces"], modele_gb.predict(grille), color=BLEU, lw=2.5, label="Prédiction du Gradient Boosting")
    ax.set_xlabel("Ventes de glaces")
    ax.set_ylabel("Noyades")
    ax.set_title("Le modèle prédit bien… mais ne dit rien de la causalité")
    ax.text(0.02, 0.97, f"R² en validation croisée : {r2_cv:.2f}\n"
                        f"Effet implicite de +1 glace : {effet_implicite:+.3f}\nEffet causal vrai : 0",
            transform=ax.transAxes, va="top", fontsize=10, color=TEXTE,
            bbox={"facecolor": "white", "edgecolor": GRIS, "boxstyle": "round"})
    ax.legend(loc="lower right")
    sauvegarder(fig, "p2_04_prediction_gb.png")
    return fig


def figure_stratification(df, n_tranches: int = 10):
    """Droites glaces-noyades à température comparable (par tranche) contre la droite naïve globale."""
    tranches = pd.qcut(df["temperature"], n_tranches, labels=False)
    couleurs = CMAP_TEMPERATURE(np.linspace(0, 1, n_tranches))
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(n_tranches):
        sous_df = df[tranches == k]
        ax.scatter(sous_df["glaces"], sous_df["noyades"], s=4, alpha=0.3, color=couleurs[k])
        pente, constante = np.polyfit(sous_df["glaces"], sous_df["noyades"], 1)
        x = np.linspace(sous_df["glaces"].quantile(0.05), sous_df["glaces"].quantile(0.95), 20)
        ax.plot(x, constante + pente * x, color="#3a2a1a", lw=2,
                label="Droites à température comparable (déciles)" if k == 0 else None)
    pente, constante = np.polyfit(df["glaces"], df["noyades"], 1)
    x = np.linspace(df["glaces"].min(), df["glaces"].max(), 20)
    ax.plot(x, constante + pente * x, color=BLEU, lw=2, ls="--", label=f"Droite naïve (pente = {pente:.2f})")
    ax.set_xlabel("Ventes de glaces")
    ax.set_ylabel("Noyades")
    ax.set_title("À température comparable, le lien s'effondre")
    ax.legend(loc="upper left")
    sauvegarder(fig, "p2_05_stratification.png")
    return fig


def figure_estimations(tableau_0, tableau_05, nom_fichier: str = "p2_06_estimations.png",
                       mots_biaises: tuple = ("naïf", "sans le carré"),
                       legende_biais: str = "Confondeur ignoré ou mal modélisé"):
    """Estimations et IC à 95 % de chaque méthode, pour un effet vrai de 0 et de 0.5.

    Les méthodes dont le nom contient un des `mots_biaises` sont en orange, les autres en bleu.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * len(tableau_0) + 1.8), sharey=True)
    for ax, tableau, effet_vrai in [(axes[0], tableau_0, 0.0), (axes[1], tableau_05, 0.5)]:
        positions = np.arange(len(tableau))[::-1]
        for pos, (_, ligne) in zip(positions, tableau.iterrows()):
            biaise = any(mot in ligne["Méthode"] for mot in mots_biaises)
            couleur = ORANGE if biaise else BLEU
            if np.isnan(ligne["IC 95 % bas"]):
                ax.plot(ligne["Effet estimé"], pos, "D", color=couleur, ms=8, markeredgecolor="white")
            else:
                ax.errorbar(ligne["Effet estimé"], pos, color=couleur, fmt="o", ms=8, capsize=4, lw=2,
                            xerr=[[ligne["Effet estimé"] - ligne["IC 95 % bas"]],
                                  [ligne["IC 95 % haut"] - ligne["Effet estimé"]]],
                            markeredgecolor="white")
            ax.text(ligne["Effet estimé"], pos + 0.27, f"{ligne['Effet estimé']:.3f}", ha="center", fontsize=8,
                    color=TEXTE, zorder=5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        ax.axvline(effet_vrai, color=ROUGE, ls="--", lw=1.5)
        ax.set_title(f"Effet vrai = {effet_vrai}")
        ax.set_xlabel("Effet estimé des glaces sur les noyades")
        ax.set_yticks(positions)
        ax.set_yticklabels(tableau["Méthode"])
        ax.grid(axis="y", visible=False)
    # Légende construite à la main (couleur = la méthode tient compte correctement de la température ou non)
    axes[1].plot([], [], "o", color=ORANGE, label=legende_biais)
    axes[1].plot([], [], "o", color=BLEU, label="Confondeur contrôlé")
    if tableau_0["IC 95 % bas"].isna().any():
        axes[1].plot([], [], "D", color=GRIS, label="Pas d'intervalle de confiance")
    axes[1].plot([], [], "--", color=ROUGE, label="Vraie valeur")
    axes[1].legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    sauvegarder(fig, nom_fichier)
    return fig


def figure_cointegration(paire_fallacieuse, paire_cointegree, p_fallacieuse: float, p_cointegree: float):
    """Séries (en haut) et résidus de la régression en niveaux (en bas) : paire fallacieuse contre paire cointégrée."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    colonnes = [(paire_fallacieuse, f"Paire indépendante : p Engle-Granger = {p_fallacieuse:.2f}"),
                (paire_cointegree, f"Paire cointégrée : p Engle-Granger = {p_cointegree:.1e}")]
    for j, (paire, titre) in enumerate(colonnes):
        axes[0, j].plot(paire["x"], color=BLEU, lw=2, label="x")
        axes[0, j].plot(paire["y"], color=ORANGE, lw=2, label="y")
        axes[0, j].set_title(titre, fontsize=11)
        axes[0, j].legend(loc="upper left")
        pente, constante = np.polyfit(paire["x"], paire["y"], 1)
        residus = paire["y"] - (constante + pente * paire["x"])
        axes[1, j].plot(residus, color=VERT, lw=1.5)
        axes[1, j].axhline(0, color=GRIS, lw=1)
        axes[1, j].set_title("Résidus de y ~ x en niveaux", fontsize=10)
        axes[1, j].set_xlabel("Temps")
    fig.suptitle("Régression en niveaux : fallacieuse ou relation de long terme ?", fontweight="bold")
    fig.tight_layout()
    sauvegarder(fig, "p1_06_cointegration.png")
    return fig


def figure_erreur_mesure(resultats, effet_naif: float):
    """Effet estimé selon l'erreur de mesure sur la température : le biais de confusion revient."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(resultats["sigma"], resultats["dml"], color=BLEU, fmt="o-", ms=7, lw=2, capsize=4,
                yerr=[resultats["dml"] - resultats["dml_ic_bas"], resultats["dml_ic_haut"] - resultats["dml"]],
                label="Double ML (IC à 95 %)")
    ax.plot(resultats["sigma"], resultats["ols"], "s--", color=VERT, ms=6, lw=1.5, label="OLS + T mesurée + T mesurée²")
    ax.axhline(0, color=ROUGE, ls="--", lw=1.5, label="Vraie valeur")
    ax.axhline(effet_naif, color=ORANGE, ls=":", lw=2, label=f"OLS naïf ({effet_naif:.2f})")
    ax.set_xlabel("Écart-type de l'erreur de mesure sur la température (°C)")
    ax.set_ylabel("Effet estimé des glaces sur les noyades")
    ax.set_title("Un confondeur mal mesuré n'est que partiellement contrôlé")
    ax.legend(loc="center right")
    sauvegarder(fig, "p3_03_erreur_mesure.png")
    return fig


def figure_dag_collider():
    """Schéma causal avec un collider : glaces -> presse <- noyades (à ne pas contrôler)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 1)
    ax.axis("off")
    noeuds = {"Température": (0.5, 0.85), "Ventes de glaces": (0.15, 0.45), "Noyades": (0.85, 0.45),
              "Articles de presse": (0.5, 0.0)}
    for nom, (x, y) in noeuds.items():
        bord = ORANGE if nom == "Articles de presse" else GRIS
        ax.text(x, y, nom, ha="center", va="center", fontsize=12, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f3f2ee", "edgecolor": bord, "linewidth": 1.5})
    fleche = {"arrowstyle": "-|>", "color": BLEU, "lw": 2, "mutation_scale": 18, "shrinkA": 28, "shrinkB": 28}
    ax.annotate("", xy=noeuds["Ventes de glaces"], xytext=noeuds["Température"], arrowprops=fleche)
    ax.annotate("", xy=noeuds["Noyades"], xytext=noeuds["Température"], arrowprops=fleche)
    fleche_collider = dict(fleche, color=ORANGE)
    ax.annotate("", xy=noeuds["Articles de presse"], xytext=noeuds["Ventes de glaces"], arrowprops=fleche_collider)
    ax.annotate("", xy=noeuds["Articles de presse"], xytext=noeuds["Noyades"], arrowprops=fleche_collider)
    ax.text(0.5, 0.45, "effet vrai = 0", ha="center", va="center", fontsize=10, color=TEXTE)
    ax.text(0.5, -0.12, "Collider : causé par les glaces ET les noyades. À ne pas contrôler.",
            ha="center", fontsize=10, color=TEXTE, style="italic")
    ax.set_title("Un mauvais contrôle : le collider")
    sauvegarder(fig, "p3_05_dag_collider.png")
    return fig


def figure_importances(scenarios: dict):
    """Importance par permutation des variables d'un Gradient Boosting, pour plusieurs scénarios."""
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4 * len(scenarios), 3.2), sharex=True)
    for ax, (titre, importances) in zip(axes, scenarios.items()):
        couleurs = [ORANGE if nom == "glaces" else BLEU for nom in importances.index]
        ax.barh(importances.index[::-1], importances.values[::-1], color=couleurs[::-1], height=0.5)
        for i, valeur in enumerate(importances.values[::-1]):
            ax.text(valeur + 0.03, i, f"{valeur:.2f}", va="center", fontsize=9, color=TEXTE)
        ax.set_title(titre, fontsize=10)
        ax.set_xlabel("Baisse du R² quand on permute")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Importance pour prédire ≠ effet causal (glaces en orange)", fontweight="bold")
    fig.tight_layout()
    sauvegarder(fig, "p3_04_importances.png")
    return fig


def figure_monte_carlo_estimateurs(mc, resume, effet_vrai: float = 0.0):
    """Distribution des estimations sur les simulations, une ligne par méthode, avec biais et couverture."""
    methodes = list(resume["methode"])
    fig, axes = plt.subplots(len(methodes), 1, figsize=(9, 1.6 * len(methodes) + 1), sharex=True)
    bins = np.linspace(mc["effet"].min(), mc["effet"].max(), 90)  # mêmes intervalles pour toutes les méthodes
    for ax, methode in zip(axes, methodes):
        ligne = resume[resume["methode"] == methode].iloc[0]
        biaise = abs(ligne["biais"]) > 0.02
        ax.hist(mc.loc[mc["methode"] == methode, "effet"], bins=bins, color=ORANGE if biaise else BLEU)
        ax.axvline(effet_vrai, color=ROUGE, ls="--", lw=1.5)
        ax.set_title(f"{methode} : biais = {ligne['biais']:+.3f}, couverture de l'IC à 95 % = {ligne['couverture']:.0f} %",
                     fontsize=10, loc="left")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("Effet estimé des glaces sur les noyades (vraie valeur en pointillés rouges)")
    fig.suptitle(f"{mc['effet'].size // len(methodes)} simulations : biais et couverture des estimateurs",
                 fontweight="bold")
    fig.tight_layout()
    sauvegarder(fig, "p3_01_monte_carlo_estimateurs.png")
    return fig


def figure_taille_echantillon(resume_taille):
    """Double ML et OLS bien spécifié selon n : dispersion des estimations, largeur de l'IC et couverture."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for methode, couleur, marqueur in [("Double ML", BLEU, "o"), ("OLS + T + T²", VERT, "s")]:
        r = resume_taille[resume_taille["methode"] == methode]
        axes[0].errorbar(r["n"], r["biais"], yerr=1.96 * r["ecart_type"], color=couleur, fmt=marqueur + "-",
                         ms=7, lw=2, capsize=4, label=methode)
        axes[1].plot(r["n"], r["largeur_ic"], marqueur + "-", color=couleur, ms=7, lw=2, label=methode)
        for _, ligne in r.iterrows():
            axes[1].annotate(f"{ligne['couverture']:.0f} %", (ligne["n"], ligne["largeur_ic"]),
                             textcoords="offset points", xytext=(6, 6 if methode == "Double ML" else -12),
                             fontsize=8, color=couleur)
    axes[0].axhline(0, color=ROUGE, ls="--", lw=1.5)
    axes[0].set_title("Biais moyen ± 1.96 écart-type")
    axes[0].set_ylabel("Effet estimé - effet vrai")
    axes[1].set_title("Largeur moyenne de l'IC à 95 % (étiquettes : couverture)")
    axes[1].set_ylabel("Largeur de l'IC")
    for ax in axes:
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(resume_taille["n"].unique())
        ax.set_xticklabels(resume_taille["n"].unique())
        ax.set_xlabel("Taille de l'échantillon (n)")
        ax.legend(loc="upper right")
    fig.tight_layout()
    sauvegarder(fig, "p3_02_taille_echantillon.png")
    return fig
