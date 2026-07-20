import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from scipy import stats
    from src.chargement import charger_donnee
    from src.nettoyage import detecter_outliers_iqr
    from src.nettoyage import detect_outliers_zscore
    from src.nettoyage import analyser_variables_categorielles
    from src.nettoyage import standardiser
    from src.nettoyage import controler_revenue
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    #%matplotlib inline
    return (
        analyser_variables_categorielles,
        charger_donnee,
        controler_revenue,
        detect_outliers_zscore,
        detecter_outliers_iqr,
        mo,
        pd,
        plt,
        standardiser,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Projet analyse vente Ecommerce
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##1 Nettoyage et preparation des donnees
    """)
    return


@app.cell
def _(charger_donnee):
    #charger les donnees
    path ="datasets/raw/ecommerce_sales_analytics_5000.csv"
    df = charger_donnee(path=path)
    return (df,)


@app.cell
def _(df):
    #visualiser debut
    df.head()
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    on remarque que pour ce dataset il n'y as aucune donnee manquante ce qui nous facilite grandement le travail
    """)
    return


@app.cell(hide_code=True)
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell
def _(controler_revenue, df):
    controler_revenue(df)
    return


@app.cell
def _(df):
    #doublon 
    df[df.duplicated()]
    #aucune duplication n'est observe
    return


@app.cell
def _(df, pd):
    print("Date min :", df['order_date'].min())
    print("Date max :", df['order_date'].max())

    # 4. Dates dans le futur (par rapport à aujourd'hui)
    dates_futures = df[df['order_date'] > pd.Timestamp.now()]
    print(f"Commandes avec date future : {len(dates_futures)}")

    # 5. Dates antérieures à une borne plausible (ex: lancement de l'activité)
    borne_min = pd.Timestamp('2020-01-01')  # à ajuster selon ton contexte
    dates_trop_anciennes = df[df['order_date'] < borne_min]
    print(f"Commandes antérieures à {borne_min.date()} : {len(dates_trop_anciennes)}")

    # 6. Doublons exacts de dates+order_id (incohérence potentielle)
    doublons_id_date = df.duplicated(subset=['order_id', 'order_date']).sum()
    print(f"Doublons order_id + order_date : {doublons_id_date}")
    return


@app.cell
def _(df):
    print("=== DÉTECTION DES OUTLIERS — Quantity ===\n")

    # Série étudiée
    quantity = df["quantity"]

    # Calcul de l'IQR
    q1 = quantity.quantile(0.25)
    q3 = quantity.quantile(0.75)

    iqr = q3 - q1

    # Bornes inférieure et supérieure
    borne_inf = q1 - 1.5 * iqr
    borne_sup = q3 + 1.5 * iqr

    # Détection des outliers
    outliers = (quantity < borne_inf) | (quantity > borne_sup)

    # Résultats
    print(f"Q1 = {q1:.1f}")
    print(f"Q3 = {q3:.1f}")
    print(f"IQR = {iqr:.1f}")

    print(f"\nBornes IQR : [{borne_inf:.1f}, {borne_sup:.1f}]")

    print(
        f"Outliers IQR : {outliers.sum()} "
        f"({outliers.sum()/len(quantity)*100:.1f}%)"
    )
    return


@app.cell
def _(detecter_outliers_iqr, df):
    colonnes_numeriques = [
        "quantity",
        "unit_price",
        "discount",
        "delivery_days",
        "customer_rating",
        "revenue"
    ]

    for colonne in colonnes_numeriques:
        detecter_outliers_iqr(df, colonne)
        print("\n")
    return (colonnes_numeriques,)


@app.cell
def _(colonnes_numeriques, df, plt):




    for _c in colonnes_numeriques:
        plt.figure(figsize=(8, 3))

        plt.boxplot(
            df[_c].dropna(),
            vert=False
        )

        plt.title(f"Diagramme à moustaches — {_c}")
        plt.xlabel(_c)

        plt.show()
    return


@app.cell
def _(detect_outliers_zscore, df):
    resultats_outliers = detect_outliers_zscore(df)
    print(resultats_outliers)
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

    > 🔎 Aucune autre variable ne présente d'outliers, quelle que soit la méthode utilisée.

    ### Interprétation

    Ces valeurs extrêmes semblent **légitimes** plutôt que des erreurs de saisie. `revenue` étant le produit de `quantity` (1 à 7) et `unit_price` (jusqu'à ~600), les commandes combinant une quantité élevée et un prix unitaire élevé génèrent mécaniquement des revenus bien supérieurs à la moyenne. Aucune valeur négative ni incohérente avec les bornes des variables sources n'a été observée. Ces outliers correspondent donc probablement à de grosses commandes réelles et **ne devraient pas être supprimés systématiquement**, au risque de fausser l'analyse en excluant des cas d'usage valides.
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    les dates semble coherentes
    """)
    return


@app.cell
def _(analyser_variables_categorielles, df):
    analyser_variables_categorielles(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Les trois variables ont une cardinalité faible (3-4 modalités) et aucune relation d'ordre naturelle (ni payment_method, ni product_category, ni region ne sont ordinales) → one-hot encoding est le choix logique pour les trois, pas de label encoding (qui introduirait un ordre artificiel).
    """)
    return


@app.cell
def _(df, pd):
    df_encoded = pd.get_dummies(
        df,
        columns=["payment_method", "product_category", "region"],
        drop_first=True,  # évite la colinéarité parfaite (piège du dummy variable)
        dtype=int
    )
    return (df_encoded,)


@app.cell
def _(colonnes_numeriques, df_encoded, standardiser):
    df_scaled, scaler = standardiser(df_encoded, colonnes_numeriques)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
