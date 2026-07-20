import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from src.chargement import charger_donnee
    from src.nettoyage import (
        analyser_variables_categorielles,
        controler_revenue,
        detect_outliers_zscore,
        detecter_outliers_iqr,
        standardiser,
    )

    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    return (
        analyser_variables_categorielles,
        charger_donnee,
        controler_revenue,
        detect_outliers_zscore,
        detecter_outliers_iqr,
        mo,
        pd,
        plt,
        sns,
        standardiser,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Projet analyse vente E-commerce
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Nettoyage et préparation des données
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 Chargement
    """)
    return


@app.cell
def _(charger_donnee):
    path = "datasets/raw/ecommerce_sales_analytics_5000.csv"
    df = charger_donnee(path=path)
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Valeurs manquantes
    """)
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On remarque que pour ce dataset il n'y a aucune donnée manquante, ce qui nous facilite grandement le travail.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 Doublons
    """)
    return


@app.cell
def _(df):
    # Doublons de lignes complètes
    n_doublons_lignes = df.duplicated().sum()
    print(f"Doublons de lignes complètes : {n_doublons_lignes}")

    # Doublons sur order_id (chaque commande doit être unique)
    n_doublons_order_id = df.duplicated(subset=["order_id"]).sum()
    print(f"Doublons sur order_id : {n_doublons_order_id}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Aucun doublon détecté, ni sur les lignes complètes, ni sur `order_id`. Chaque commande est bien unique.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4 Contrôle de cohérence de `revenue`
    """)
    return


@app.cell
def _(controler_revenue, df):
    incoherents = controler_revenue(df)
    incoherents
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    24 lignes présentent un écart moyen de 0.48 %, dû à un arrondi de N décimales. L'impresision reste tres faible donc `revenue` peut être utilisé en confiance pour la suite de l'analyse.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.5 Cohérence des bornes
    """)
    return


@app.cell
def _(df):
    print(
        "discount hors [0, 1[ :",
        (~df["discount"].between(0, 1, inclusive="left")).sum(),
    )
    print("customer_rating hors [1, 5] :", (~df["customer_rating"].between(1, 5)).sum())
    print("quantity <= 0 :", (df["quantity"] <= 0).sum())
    print("delivery_days <= 0 :", (df["delivery_days"] <= 0).sum())
    print("unit_price <= 0 :", (df["unit_price"] <= 0).sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.6 Cohérence des dates
    """)
    return


@app.cell
def _(df, pd):
    print("Date min :", df["order_date"].min())
    print("Date max :", df["order_date"].max())

    dates_futures = df[df["order_date"] > pd.Timestamp.now()]
    print(f"Commandes avec date future : {len(dates_futures)}")

    borne_min = pd.Timestamp("2020-01-01")
    dates_trop_anciennes = df[df["order_date"] < borne_min]
    print(f"Commandes antérieures à {borne_min.date()} : {len(dates_trop_anciennes)}")

    doublons_id_date = df.duplicated(subset=["order_id", "order_date"]).sum()
    print(f"Doublons order_id + order_date : {doublons_id_date}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Les dates semblent cohérentes : bornées dans une plage plausible, aucune date future, aucun doublon `order_id` + `order_date`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.7 Détection des outliers
    """)
    return


@app.cell
def _(detecter_outliers_iqr, df):
    colonnes_numeriques = [
        "quantity",
        "unit_price",
        "discount",
        "delivery_days",
        "customer_rating",
        "revenue",
    ]

    for colonne in colonnes_numeriques:
        detecter_outliers_iqr(df, colonne)
        print("\n")
    return (colonnes_numeriques,)


@app.cell
def _(colonnes_numeriques, df, plt, sns):
    n = len(colonnes_numeriques)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n))

    for ax, _c in zip(axes, colonnes_numeriques):
        sns.boxplot(x=df[_c].dropna(), color="skyblue", fliersize=4, linewidth=1.2, ax=ax)
        ax.set_title(f"Diagramme à moustaches — {_c}", fontsize=11, fontweight="bold")
        ax.set_xlabel(_c)
        sns.despine(left=True, ax=ax)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(detect_outliers_zscore, df):
    resultats_outliers = detect_outliers_zscore(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Détection des outliers

    Deux méthodes ont été utilisées pour détecter les outliers : les **quartiles (IQR)** et le **score Z**.

    Dans les deux cas, **seule la variable `revenue`** présente des outliers.

    | Méthode | % d'outliers sur `revenue` |
    |---|---|
    | IQR | **1.34 %** |
    | Z-score | **0.52 %** |

    > Aucune autre variable ne présente d'outliers, quelle que soit la méthode utilisée.

    ### Interprétation

    Ces valeurs extrêmes semblent **légitimes** plutôt que des erreurs de saisie. `revenue` étant le produit de `quantity` (1 à 7) et `unit_price` (jusqu'à ~600), les commandes combinant une quantité élevée et un prix unitaire élevé génèrent mécaniquement des revenus bien supérieurs à la moyenne. Aucune valeur négative ni incohérente avec les bornes des variables sources n'a été observée. Ces outliers correspondent donc probablement à de grosses commandes réelles et **ne devraient pas être supprimés systématiquement**, au risque de fausser l'analyse en excluant des cas d'usage valides.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.8 Analyse des variables catégorielles
    """)
    return


@app.cell
def _(analyser_variables_categorielles, df):
    analyser_variables_categorielles(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Les trois variables ont une cardinalité faible (3-4 modalités) et aucune relation d'ordre naturelle (ni `payment_method`, ni `product_category`, ni `region` ne sont ordinales) → one-hot encoding est le choix logique pour les trois, pas de label encoding (qui introduirait un ordre artificiel).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.9 Encodage
    """)
    return


@app.cell
def _(df, pd):
    df_encoded = pd.get_dummies(
        df,
        columns=["payment_method", "product_category", "region"],
        drop_first=True,
        dtype=int,
    )
    df_encoded.head()
    return (df_encoded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.10 Standardisation
    """)
    return


@app.cell
def _(colonnes_numeriques, df_encoded, standardiser):
    df_scaled, scaler = standardiser(df_encoded, colonnes_numeriques)
    df_scaled[colonnes_numeriques].describe()
    return (df_scaled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Après standardisation, les variables numériques ont bien une moyenne ≈ 0 et un écart-type ≈ 1, sur la base du `describe()` ci-dessus. Le jeu de données est maintenant prêt pour l'analyse exploratoire (section 2).
    """)
    return


@app.cell
def _(df_scaled):
    df_scaled.to_csv("datasets/processed/ecommerce_clean.csv", index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
