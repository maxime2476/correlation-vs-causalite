"""Génération des données simulées du projet."""
import numpy as np
import pandas as pd


def simuler_marches_aleatoires(n: int = 100, derive: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Simule deux marches aléatoires indépendantes avec dérive : x_t = x_{t-1} + derive + bruit N(0, 1)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(derive + rng.normal(0, 1, n))
    y = np.cumsum(derive + rng.normal(0, 1, n))
    return pd.DataFrame({"x": x, "y": y})


def simuler_paire_cointegree(n: int = 100, derive: float = 0.5, seed: int = 7) -> pd.DataFrame:
    """Simule deux séries non stationnaires qui suivent la même tendance stochastique w (donc cointégrées)."""
    rng = np.random.default_rng(seed)
    w = np.cumsum(derive + rng.normal(0, 1, n))
    x = w + rng.normal(0, 1, n)
    y = 2 + 0.8 * w + rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "y": y})


def simuler_glaces_noyades(n: int = 5000, effet: float = 0.0, seed: int = 123) -> pd.DataFrame:
    """Simule température, ventes de glaces et noyades ; `effet` est l'effet causal vrai des glaces sur les noyades."""
    rng = np.random.default_rng(seed)
    temperature = rng.normal(20, 6, n)
    glaces = 2 * temperature + 0.05 * temperature**2 + rng.normal(0, 5, n)
    noyades = 0.3 * temperature + 0.01 * temperature**2 + effet * glaces + rng.normal(0, 2, n)
    return pd.DataFrame({"temperature": temperature, "glaces": glaces, "noyades": noyades})


def ajouter_temperature_mesuree(df: pd.DataFrame, sigma: float, seed: int = 99) -> pd.DataFrame:
    """Ajoute une température mesurée avec erreur : temperature + N(0, sigma).

    Le même tirage aléatoire est utilisé pour tous les sigma (seule l'amplitude change), pour des courbes lisses.
    """
    rng = np.random.default_rng(seed)
    return df.assign(temp_mesuree=df["temperature"] + sigma * rng.normal(0, 1, len(df)))


def ajouter_presse(df: pd.DataFrame, seed: int = 456) -> pd.DataFrame:
    """Ajoute une variable « collider » causée par les glaces ET les noyades (articles de presse sur l'été)."""
    rng = np.random.default_rng(seed)
    return df.assign(presse=0.2 * df["glaces"] + 2 * df["noyades"] + rng.normal(0, 3, len(df)))
