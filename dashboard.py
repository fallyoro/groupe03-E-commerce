import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tableau de bord — Performance de la boutique en ligne

    Ce tableau de bord permet d'explorer les commandes de la boutique par **catégorie**,
    **région** et **niveau de remise**. Les indicateurs et graphiques se recalculent
    automatiquement selon les filtres choisis.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from src.chargement import charger_donnee

    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    return charger_donnee, mo, plt, sns


@app.cell
def _(charger_donnee):
    # On repart du jeu de données nettoyé (partie "Nettoyage" de analyse.py).
    # Si le fichier processed n'existe pas encore, on utilise le brut en secours.
    import os

    chemin_propre = "datasets/processed/ecommerce_clean.csv"
    chemin_brut = "datasets/raw/ecommerce_sales_analytics_5000.csv"

    if os.path.exists(chemin_propre):
        df_brut = charger_donnee(path=chemin_brut)  # pour garder les colonnes non encodées (catégorie, région...)
    else:
        df_brut = charger_donnee(path=chemin_brut)

    df_brut["tranche_remise"] = df_brut["discount"].apply(
        lambda d: "0-10%" if d < 0.1 else ("10-20%" if d < 0.2 else ("20-30%" if d < 0.3 else "30%+"))
    )
    return (df_brut,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filtres
    Choisissez une catégorie, une région et une plage de remise ("Toutes" pour ne pas
    filtrer) : tous les indicateurs et graphiques ci-dessous s'ajustent automatiquement.
    """)
    return


@app.cell
def _(df_brut, mo):
    _options_categorie = ["Toutes les catégories"] + sorted(df_brut["product_category"].unique().tolist())
    _options_region = ["Toutes les régions"] + sorted(df_brut["region"].unique().tolist())

    filtre_categorie = mo.ui.dropdown(
        options=_options_categorie,
        value="Toutes les catégories",
        label="Catégorie de produit",
    )
    filtre_region = mo.ui.dropdown(
        options=_options_region,
        value="Toutes les régions",
        label="Région",
    )
    filtre_remise = mo.ui.range_slider(
        start=0.0,
        stop=float(df_brut["discount"].max()),
        step=0.01,
        value=[0.0, float(df_brut["discount"].max())],
        label="Taux de remise",
        show_value=True,
    )
    mo.hstack([filtre_categorie, filtre_region, filtre_remise])
    return filtre_categorie, filtre_region, filtre_remise


@app.cell
def _(df_brut, filtre_categorie, filtre_region, filtre_remise):
    # Sous-ensemble filtré, recalculé à chaque interaction.
    # Un dropdown sur "Toutes les catégories" / "Toutes les régions" désactive ce filtre.
    _masque = df_brut["discount"].between(filtre_remise.value[0], filtre_remise.value[1])

    if filtre_categorie.value != "Toutes les catégories":
        _masque = _masque & (df_brut["product_category"] == filtre_categorie.value)

    if filtre_region.value != "Toutes les régions":
        _masque = _masque & (df_brut["region"] == filtre_region.value)

    df_filtre = df_brut[_masque]
    return (df_filtre,)


@app.cell(hide_code=True)
def _(df_filtre, mo):
    # Message clair si la sélection ne contient aucune commande, plutôt qu'un plantage
    _alerte = None
    if len(df_filtre) == 0:
        _alerte = mo.callout(
            "Aucune commande ne correspond à cette combinaison de filtres. "
            "Élargissez la sélection (catégorie, région ou remise) pour voir les résultats.",
            kind="warn",
        )
    _alerte
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Indicateurs clés
    Chaque indicateur compare la sélection actuelle (grand chiffre) à l'ensemble des
    5 000 commandes (chiffre entre parenthèses), pour situer immédiatement le
    sous-ensemble filtré.
    """)
    return


@app.cell
def _(df_brut, df_filtre, mo):
    def indicateur(titre, valeur_filtre, valeur_globale, unite=""):
        return mo.stat(
            value=f"{valeur_filtre}{unite}",
            label=f"{titre} (ensemble : {valeur_globale}{unite})",
        )

    n_commandes = len(df_filtre)
    ca_total = df_filtre["revenue"].sum()
    ca_total_global = df_brut["revenue"].sum()
    panier_moyen = df_filtre["revenue"].mean() if n_commandes else 0
    panier_moyen_global = df_brut["revenue"].mean()
    note_moyenne = df_filtre["customer_rating"].mean() if n_commandes else 0
    note_moyenne_global = df_brut["customer_rating"].mean()
    delai_moyen = df_filtre["delivery_days"].mean() if n_commandes else 0
    delai_moyen_global = df_brut["delivery_days"].mean()

    bandeau = mo.hstack(
        [
            indicateur("Commandes sélectionnées", n_commandes, len(df_brut)),
            indicateur("Chiffre d'affaires ($)", f"{ca_total:,.0f}", f"{ca_total_global:,.0f}"),
            indicateur("Panier moyen ($)", f"{panier_moyen:,.0f}", f"{panier_moyen_global:,.0f}"),
            indicateur("Note client moyenne", f"{note_moyenne:.2f}", f"{note_moyenne_global:.2f}", " /5"),
            indicateur("Délai de livraison moyen", f"{delai_moyen:.1f}", f"{delai_moyen_global:.1f}", " j"),
        ],
        justify="space-around",
    )
    bandeau
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualisations

    Les quatre graphiques ci-dessous se recalculent automatiquement en fonction des
    filtres sélectionnés plus haut (catégorie, région, remise).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Chiffre d'affaires par catégorie
    Identifie les catégories qui portent le plus l'activité sur la sélection actuelle —
    utile pour prioriser l'effort commercial.
    """)
    return


@app.cell
def _(df_filtre, mo, plt, sns):
    if len(df_filtre) == 0:
        _fig = mo.md("*Pas de données à afficher pour cette sélection.*")
    else:
        _fig, _ax = plt.subplots(figsize=(7, 4))
        _ca_cat = df_filtre.groupby("product_category")["revenue"].sum().sort_values(ascending=False)
        sns.barplot(x=_ca_cat.index, y=_ca_cat.values, ax=_ax, color="steelblue")
        _ax.set_title("Chiffre d'affaires par catégorie (sélection actuelle)")
        _ax.set_xlabel("Catégorie de produit")
        _ax.set_ylabel("Chiffre d'affaires ($)")
        plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Chiffre d'affaires et panier moyen par région
    Distingue le volume d'affaires généré par région (à gauche) de la valeur moyenne
    d'une commande dans chaque région (à droite) — une région peut peser lourd en
    volume sans avoir le panier le plus élevé.
    """)
    return


@app.cell
def _(df_filtre, mo, plt, sns):
    if len(df_filtre) == 0:
        _fig2 = mo.md("*Pas de données à afficher pour cette sélection.*")
    else:
        _fig2, _axes2 = plt.subplots(1, 2, figsize=(10, 4))
        _par_region = df_filtre.groupby("region")["revenue"].agg(["sum", "mean"])

        sns.barplot(x=_par_region.index, y=_par_region["sum"], ax=_axes2[0], color="mediumseagreen")
        _axes2[0].set_title("Chiffre d'affaires total par région")
        _axes2[0].set_xlabel("Région")
        _axes2[0].set_ylabel("Chiffre d'affaires ($)")

        sns.barplot(x=_par_region.index, y=_par_region["mean"], ax=_axes2[1], color="coral")
        _axes2[1].set_title("Panier moyen par région")
        _axes2[1].set_xlabel("Région")
        _axes2[1].set_ylabel("Panier moyen ($)")

        plt.tight_layout()
    _fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Note client selon la tranche de remise
    Sur l'ensemble des données, un test statistique (ANOVA) ne détecte aucun effet
    significatif de la remise sur la note client (p=0,38) : accorder une remise plus
    généreuse n'améliore pas la satisfaction — ce graphique permet de le vérifier
    visuellement sur le sous-ensemble filtré.
    """)
    return


@app.cell
def _(df_filtre, mo, plt, sns):
    if len(df_filtre) == 0:
        _fig3 = mo.md("*Pas de données à afficher pour cette sélection.*")
    else:
        _fig3, _ax3 = plt.subplots(figsize=(7, 4))
        _ordre = ["0-10%", "10-20%", "20-30%", "30%+"]
        _ordre_present = [t for t in _ordre if t in df_filtre["tranche_remise"].unique()]
        sns.boxplot(
            data=df_filtre, x="tranche_remise", y="customer_rating",
            order=_ordre_present, ax=_ax3, color="skyblue",
        )
        _ax3.set_title("Note client selon la tranche de remise accordée")
        _ax3.set_xlabel("Tranche de remise")
        _ax3.set_ylabel("Note client (1 à 5)")
        plt.tight_layout()
    _fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. Délai de livraison vs note client
    Le service client s'inquiète de l'effet du délai de livraison sur la satisfaction.
    Le test statistique global est proche du seuil de significativité (p=0,09) sans le
    franchir : le nuage de points ci-dessous, coloré par chiffre d'affaires, permet
    d'explorer ce lien sur le sous-ensemble sélectionné.
    """)
    return


@app.cell
def _(df_filtre, mo, plt):
    if len(df_filtre) == 0:
        _fig4 = mo.md("*Pas de données à afficher pour cette sélection.*")
    else:
        _fig4, _ax4 = plt.subplots(figsize=(7, 4))
        _scatter = _ax4.scatter(
            df_filtre["delivery_days"], df_filtre["customer_rating"],
            c=df_filtre["revenue"], cmap="viridis", alpha=0.5, s=15,
        )
        plt.colorbar(_scatter, ax=_ax4, label="Chiffre d'affaires ($)")
        _ax4.set_title("Délai de livraison vs note client")
        _ax4.set_xlabel("Délai de livraison (jours)")
        _ax4.set_ylabel("Note client (1 à 5)")
        plt.tight_layout()
    _fig4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    *Tableau de bord — Groupe 3, Analyse de Données IPSL. Données : E-commerce Sales
    Analytics (Kaggle). Pour l'analyse statistique détaillée (tests, ACP, journal de
    nettoyage), voir `analyse.py`.*
    """)
    return


if __name__ == "__main__":
    app.run()
