from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def heatmap(matrice_corr: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrice_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Coefficient de corrélation"},
    )
    plt.title(
        "Corrélations entre variables numériques : revenue lié uniquement à quantity et unit_price"
    )
    plt.tight_layout()
    plt.show()


def plot_distribution_revenue(
    data: pd.DataFrame, colonne: str = "revenue", figsize: tuple = (9, 5)
):
    """Histogramme du revenu avec moyenne/médiane, pour montrer l'asymétrie."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data=data, x=colonne, bins=50, color="skyblue", ax=ax)

    moyenne = float(data[colonne].mean())
    mediane = float(data[colonne].median())

    ax.axvline(
        moyenne,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Moyenne : {moyenne:.0f} $",
    )
    ax.axvline(
        mediane,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Médiane : {mediane:.0f} $",
    )
    ax.set_title(
        "Le revenu par commande est tiré vers le haut par une minorité de grosses commandes",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Chiffre d'affaires par commande ($)")
    ax.set_ylabel("Nombre de commandes")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_ca_par_categorie(data: pd.DataFrame, figsize: tuple = (9, 5)):
    """Chiffre d'affaires total par catégorie de produit, en barres horizontales triées."""
    fig, ax = plt.subplots(figsize=figsize)

    ca_cat = cast(
        pd.Series, data.groupby("product_category")["revenue"].sum()
    ).sort_values(ascending=True)

    valeurs = np.asarray(ca_cat.to_numpy(), dtype=float)
    ax.barh(ca_cat.index.astype(str), valeurs, color="steelblue")

    categorie_max = str(ca_cat.idxmax())
    categorie_min = str(ca_cat.idxmin())
    ax.set_title(
        f"{categorie_max} génère le plus de chiffre d'affaires, {categorie_min} le moins",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Chiffre d'affaires total ($)")
    ax.set_ylabel("Catégorie de produit")

    for i, v in enumerate(valeurs):
        ax.text(v, i, f" {v:,.0f} $", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_note_par_tranche(
    data: pd.DataFrame,
    colonne_tranche: str,
    titre: str,
    xlabel: str,
    couleur: str = "skyblue",
    figsize: tuple = (9, 5),
):
    """Boxplot générique de la note client selon une variable en tranches, échelle 1-5 fixe."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=data, x=colonne_tranche, y="customer_rating", ax=ax, color=couleur)
    ax.set_title(titre, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Note client (échelle 1 à 5)")
    ax.set_ylim(0.5, 5.5)  # échelle honnête : toute la plage possible de la note
    plt.tight_layout()
    return fig


def plot_ca_et_panier_par_region(data: pd.DataFrame, figsize: tuple = (13, 5)):
    """Chiffre d'affaires total et panier moyen par région, côte à côte."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ca_region = cast(pd.Series, data.groupby("region")["revenue"].sum()).sort_values(
        ascending=False
    )
    axes[0].bar(
        ca_region.index.astype(str),
        np.asarray(ca_region.to_numpy(), dtype=float),
        color="steelblue",
    )
    axes[0].set_title(
        "Chiffre d'affaires total par région", fontsize=11, fontweight="bold"
    )
    axes[0].set_xlabel("Région")
    axes[0].set_ylabel("Chiffre d'affaires total ($)")

    panier_region = cast(
        pd.Series, data.groupby("region")["revenue"].mean()
    ).sort_values(ascending=False)
    axes[1].bar(
        panier_region.index.astype(str),
        np.asarray(panier_region.to_numpy(), dtype=float),
        color="coral",
    )
    axes[1].set_title("Panier moyen par région", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Région")
    axes[1].set_ylabel("Panier moyen ($)")

    plt.tight_layout()
    return fig
