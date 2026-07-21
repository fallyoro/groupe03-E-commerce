import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from src.chargement import charger_donnee
    from src.visualisation import heatmap
    from src.nettoyage import (
        analyser_variables_categorielles,
        controler_revenue,
        detect_outliers_zscore,
        detecter_outliers_iqr,
        standardiser,
    )
    from src.visualisation import (
        plot_distribution_revenue,
        plot_ca_par_categorie,
        plot_note_par_tranche,
        plot_ca_et_panier_par_region,
    )

    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    return (
        analyser_variables_categorielles,
        charger_donnee,
        controler_revenue,
        detect_outliers_zscore,
        detecter_outliers_iqr,
        heatmap,
        mo,
        np,
        pd,
        plot_ca_et_panier_par_region,
        plot_ca_par_categorie,
        plot_distribution_revenue,
        plot_note_par_tranche,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Exploration et description des données
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 Statistiques descriptives univariées
    """)
    return


@app.cell
def _(df, pd):
    from scipy.stats import skew, kurtosis

    colonnes_numeriques_desc = ["quantity", "unit_price", "discount", "delivery_days", "customer_rating", "revenue"]

    stats_univariees = pd.DataFrame({
        "moyenne": df[colonnes_numeriques_desc].mean(),
        "mediane": df[colonnes_numeriques_desc].median(),
        "ecart_type": df[colonnes_numeriques_desc].std(),
        "min": df[colonnes_numeriques_desc].min(),
        "max": df[colonnes_numeriques_desc].max(),
        "asymetrie": df[colonnes_numeriques_desc].apply(skew),
        "aplatissement": df[colonnes_numeriques_desc].apply(kurtosis),
    }).round(2)

    stats_univariees
    return (colonnes_numeriques_desc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lecture du tableau :**

    Toutes les variables sauf `revenue` ont une asymétrie quasi nulle (entre -0.07 et 0.04) : moyenne et médiane concordent, la moyenne est donc un résumé fiable de la tendance centrale. Leur aplatissement négatif (~-1.2) indique des distributions plutôt uniformes que gaussiennes, cohérent avec des bornes contrôlées dans la génération des données.

    `revenue` fait exception : son asymétrie de 1.0 traduit une distribution étalée vers la droite, confirmée par l'écart entre moyenne (1021.96) et médiane (796.65). Quelques commandes à forte quantité/prix unitaire tirent la moyenne vers le haut — cohérent avec les outliers légitimes déjà identifiés (1.34% par IQR). **Pour cette variable, la médiane est l'indicateur de tendance centrale à privilégier.**
    """)
    return


@app.cell
def _(colonnes_numeriques_desc, df, plt, sns):
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 8))
    _axes = _axes.flatten()

    for _ax, _c in zip(_axes, colonnes_numeriques_desc):
        sns.histplot(df[_c].dropna(), kde=True, ax=_ax, color="skyblue")
        _ax.axvline(df[_c].mean(), color="red", linestyle="--", label="Moyenne")
        _ax.axvline(df[_c].median(), color="green", linestyle="--", label="Médiane")
        _ax.set_title(_c)
        _ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On observe visuellement l'écart entre moyenne et médiane sur chaque variable — plus cet écart est grand, plus la distribution est asymétrique, confirmant le choix de l'indicateur à privilégier.
    "\"\")
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Analyse bivariée
    """)
    return


@app.cell
def _(colonnes_numeriques_desc, df, heatmap):
    matrice_corr = df[colonnes_numeriques_desc].corr().round(2)
    heatmap(matrice_corr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lecture de la matrice de corrélation :**

    `revenue` est fortement lié à `unit_price` (r = 0.68) et `quantity` (r = 0.62), confirmant que le chiffre d'affaires par commande est porté à parts quasi égales par le prix unitaire et le volume acheté — cohérent avec sa formule de calcul. La corrélation avec `discount` est plus faible (r = -0.14) mais de signe négatif, comme attendu : la remise réduit mécaniquement le revenu, sans que l'effet soit dominant dans ce dataset.

    Toutes les autres paires de variables sont quasiment non corrélées (|r| ≤ 0.02), notamment `delivery_days` et `customer_rating` (r = -0.02) : le délai de livraison ne semble pas expliquer la satisfaction client dans ces données, contrairement à l'inquiétude du service client — un point à confirmer par un test statistique. De même, `customer_rating` n'est corrélé à aucune autre variable numérique, suggérant que sa variation dépend de facteurs non mesurés ici (qualité produit, expérience individuelle...).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Test statistique : effet de la remise et du délai sur la satisfaction (ANOVA)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Pourquoi une ANOVA ?** On veut comparer la moyenne de `customer_rating` (variable numérique continue) entre plusieurs groupes indépendants (tranches de remise, ou tranches de délai — variables catégorielles à plus de 2 modalités). L'ANOVA à un facteur est le test adapté à cette configuration.

    **Hypothèses de l'ANOVA à vérifier :**
    1. Indépendance des observations (vérifiée : chaque ligne = une commande distincte).
    2. Normalité des résidus dans chaque groupe (approximativement vérifiée ici vu le grand effectif — théorème central limite).
    3. Homogénéité des variances entre groupes (test de Levene, à vérifier avant de conclure).

    **Hypothèses statistiques testées :**
    - H0 : la moyenne de `customer_rating` est identique entre toutes les tranches (pas d'effet).
    - H1 : au moins une tranche a une moyenne différente (effet présent).
    """)
    return


@app.cell
def _(df, pd):
    from scipy.stats import f_oneway, levene
    df["tranche_remise"] = pd.cut(df["discount"], bins=[0, 0.1, 0.2, 0.3, 1], labels=["0-10%", "10-20%", "20-30%", "30%+"], include_lowest=True)
    df["tranche_delai"] = pd.cut(df["delivery_days"], bins=[0, 3, 6, 9, 15], labels=["1-3j", "4-6j", "7-9j", "10j+"], include_lowest=True)
    df[["tranche_remise", "tranche_delai"]].head()
    return f_oneway, levene


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Test 1 — Remise → Note**
    """)
    return


@app.cell
def _(df, f_oneway, levene):
    groupes_remise = [g["customer_rating"].values for _, g in df.groupby("tranche_remise", observed=True)]
    stat_lev_r, p_lev_r = levene(*groupes_remise)
    stat_f_r, p_f_r = f_oneway(*groupes_remise)

    print("=== Remise → Note client ===")
    print(f"Levene   : stat={stat_lev_r:.4f}, p={p_lev_r:.4f}")
    print(f"ANOVA    : F={stat_f_r:.4f}, p={p_f_r:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Test 2 → note client**
    """)
    return


@app.cell
def _(df, f_oneway, levene):
    groupes_delai = [g["customer_rating"].values for _, g in df.groupby("tranche_delai", observed=True)]
    stat_lev_d, p_lev_d = levene(*groupes_delai)
    stat_f_d, p_f_d = f_oneway(*groupes_delai)

    print("=== Délai de livraison → Note client ===")
    print(f"Levene   : stat={stat_lev_d:.4f}, p={p_lev_d:.4f}")
    print(f"ANOVA    : F={stat_f_d:.4f}, p={p_f_d:.4f}")
    return


@app.cell
def _(df, plt, sns):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(data=df, x="tranche_remise", y="customer_rating", ax=_axes[0], color="skyblue")
    _axes[0].set_title("Note client selon la tranche de remise")
    _axes[0].set_xlabel("Tranche de remise")
    _axes[0].set_ylabel("Note client (1 à 5)")

    sns.boxplot(data=df, x="tranche_delai", y="customer_rating", ax=_axes[1], color="salmon")
    _axes[1].set_title("Note client selon la tranche de délai de livraison")
    _axes[1].set_xlabel("Délai de livraison")
    _axes[1].set_ylabel("Note client (1 à 5)")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Remise → Note client**
    Le test de Levene (p=0.9649) valide l'homogénéité des variances. L'ANOVA ne détecte aucune différence significative de note client selon la tranche de remise (F=1.03, p=0.38). **La remise n'a pas d'effet mesurable sur la satisfaction client** : accorder une remise plus généreuse n'améliore pas la note laissée par le client.

    **Délai de livraison → Note client**
    Le test de Levene (p=0.6191) valide l'homogénéité des variances. L'ANOVA ne détecte pas de différence statistiquement significative au seuil de 5% (F=2.16, p=0.091), bien que le résultat soit plus proche du seuil de significativité que pour la remise. On ne peut pas conclure formellement à un effet du délai de livraison sur la satisfaction avec ces données, même si une tendance faible n'est pas exclue.

    **Implication pour la direction commerciale :** la politique de remises généreuses ne semble pas justifiée par un gain de satisfaction client — ce budget pourrait être réorienté. L'inquiétude du service client sur les délais de livraison n'est pas confirmée statistiquement ici, mais mérite d'être surveillée plutôt qu'écartée, vu la proximité du seuil de significativité.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 Visualisations à message
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Graphique 1**
    Distribution du chiffre d'affaires par commande**

    L'histogramme ci-dessous illustre l'asymétrie déjà identifiée dans les statistiques univariées : la moyenne (1021.96 $) est nettement supérieure à la médiane (796.65 $), signe qu'une minorité de commandes à forte valeur tire l'ensemble vers le haut. Ces commandes correspondent aux cas où quantité et prix unitaire sont simultanément élevés — des valeurs légitimes, pas des erreurs de saisie.
    """)
    return


@app.cell
def _(df, plot_distribution_revenue, plt):
    plot_distribution_revenue(df)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Graphique 2
    pythonmo.md(r"\"\"
    **Chiffre d'affaires par catégorie de produit**

    Ce graphique compare la contribution de chaque catégorie au chiffre d'affaires total, permettant d'identifier les catégories à prioriser dans la stratégie commerciale.
    "\"\")
    """)
    return


@app.cell
def _(df, plot_ca_par_categorie, plt):
    plot_ca_par_categorie(df)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Graphique 3**
    **Effet de la remise sur la satisfaction client**

    Le test ANOVA réalisé plus haut (F=1.03, p=0.38) n'a détecté aucune différence significative de note client entre les tranches de remise. Le graphique ci-dessous confirme visuellement ce résultat : les distributions de note se superposent largement d'une tranche à l'autre, sans tendance claire à la hausse ou à la baisse.
    """)
    return


@app.cell
def _(df, plot_note_par_tranche, plt):
    plot_note_par_tranche(
        df, "tranche_remise",
        titre="La remise n'a pas d'effet significatif sur la note client (ANOVA, p=0.38)",
        xlabel="Tranche de remise accordée",
        couleur="skyblue",
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Graphique 4**
    **Effet du délai de livraison sur la satisfaction client**

    Contrairement à l'inquiétude exprimée par le service client, le test ANOVA (F=2.16, p=0.09) ne permet pas de conclure à un effet statistiquement significatif du délai de livraison sur la note client au seuil de 5%, même si le résultat est plus proche de la significativité que pour la remise. Le graphique montre des distributions globalement comparables entre tranches de délai.
    """)
    return


@app.cell
def _(df, plot_note_par_tranche, plt):
    plot_note_par_tranche(
        df, "tranche_delai",
        titre="Le délai de livraison n'a pas d'effet significatif sur la note (ANOVA, p=0.09)",
        xlabel="Délai de livraison",
        couleur="salmon",
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Graphique 5**
    **Chiffre d'affaires et panier moyen par région**

    Ces deux graphiques permettent de distinguer deux dynamiques différentes : le volume d'affaires généré par région (à gauche) et la valeur moyenne d'une commande dans chaque région (à droite). Une région peut générer un fort chiffre d'affaires total simplement par volume de commandes, sans que son panier moyen soit le plus élevé — ce croisement aide à nuancer l'interprétation.
    """)
    return


@app.cell
def _(df, plot_ca_et_panier_par_region, plt):
    plot_ca_et_panier_par_region(df)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion de la partie 2 — Exploration et description des données

    **Sur la forme des variables.** La majorité des variables numériques (`quantity`, `unit_price`, `discount`, `delivery_days`, `customer_rating`) présentent des distributions quasi symétriques (asymétrie proche de 0), pour lesquelles la moyenne est un résumé fiable. Seule `revenue` s'écarte de ce constat : son asymétrie positive (1.0) et l'écart entre moyenne (1021.96 $) et médiane (796.65 $) montrent qu'une minorité de commandes à forte valeur tire l'ensemble vers le haut — un phénomène jugé légitime (grosses commandes réelles) et non une anomalie, comme confirmé par l'analyse des outliers en partie 1.

    **Sur les relations entre variables.** La matrice de corrélation confirme que `revenue` est structurellement dépendant de `unit_price` (r=0.68) et `quantity` (r=0.62), et plus faiblement de `discount` (r=-0.14). En dehors de ce lien mécanique, aucune variable n'est corrélée aux autres de façon notable (|r| ≤ 0.02) — en particulier `customer_rating` n'est corrélé à rien, y compris `delivery_days` (r=-0.02). Cette faible interdépendance générale suggère qu'une ACP menée sur ce noyau numérique aurait peu de chances de révéler une structure de variance concentrée sur un petit nombre d'axes.

    **Sur les leviers testés statistiquement.** Les deux hypothèses métier centrales du sujet ont été testées par ANOVA :
    - **La remise n'achète pas la satisfaction** : aucune différence significative de note client entre tranches de remise n'a été détectée (F=1.03, p=0.38). Accorder des remises plus généreuses n'améliore donc pas mécaniquement la perception client dans ces données.
    - **Le délai de livraison n'a pas d'effet démontré, mais reste à surveiller** : le test n'atteint pas le seuil de significativité de 5% (F=2.16, p=0.09), bien que ce résultat soit sensiblement plus proche du seuil que celui de la remise. On ne peut ni conclure à un effet réel, ni l'exclure totalement avec ces données.

    **Implications pour la direction commerciale.** La politique actuelle de remises généreuses ne semble pas justifiée par un gain de satisfaction mesurable — ce budget promotionnel pourrait être réévalué ou réorienté, par exemple vers la logistique. À l'inverse, l'inquiétude du service client sur les délais de livraison, bien que non confirmée statistiquement au seuil conventionnel, n'est pas à écarter et mériterait une analyse complémentaire (échantillon plus large, ou segmentation par catégorie/région) avant toute décision.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Réduction de dimension (ACP)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1. L'ACP est-elle pertinente ici ?
    Avant de lancer quoi que ce soit, on se pose la question que le sujet nous impose de trancher : l'ACP a-t-elle un intérêt sur ce jeu de données ?

    Le noyau numérique dont on dispose (`quantity`, `unit_price`, `discount`, `delivery_days`, `customer_rating`, `revenue`) ne compte que **6 variables** — c'est peu. Et surtout, la matrice de corrélation de la partie 2.2 a déjà répondu en partie à la question : en dehors du lien mécanique entre `revenue` et ses composantes (`unit_price` r=0.68, `quantity` r=0.62), toutes les autres paires de variables sont quasiment non corrélées (|r| ≤ 0.02).

    Or l'ACP ne devient intéressante que lorsque des variables nombreuses et corrélées permettent de les résumer en quelques axes sans perdre trop d'information. Ici, on a peu de variables, et elles sont peu corrélées entre elles. On s'attend donc à ce que l'ACP n'apporte **pas grand-chose** — mais on ne le suppose pas, on va le vérifier avec les chiffres, comme demandé.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2. Standardisation;
    Les variables ont déjà été standardisées en partie 1.10 (`df_scaled`), condition indispensable avant toute ACP : sans cela, `revenue` (variance forte, valeurs jusqu'à plusieurs milliers) écraserait à lui seul l'inertie du nuage de points face à `discount` (valeurs entre 0 et 0.3).
    """)
    return


@app.cell
def _(colonnes_numeriques_desc, df_scaled):
    from sklearn.decomposition import PCA

    X_acp = df_scaled[colonnes_numeriques_desc]
    pca = PCA()
    composantes = pca.fit_transform(X_acp)
    return X_acp, composantes, pca


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cette cellule prépare simplement les objets `pca` et `composantes` qui seront utilisés dans les cellules suivantes. L'ACP est calculée sur les 6 variables numériques déjà standardisées.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3. Éboulis des valeurs propres : combien d'axes garder ?
    """)
    return


@app.cell
def _(np, pca, plt):
    variance_expliquee = pca.explained_variance_ratio_
    variance_cumulee = np.cumsum(variance_expliquee)

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.bar(range(1, len(variance_expliquee) + 1), variance_expliquee * 100,
            color="skyblue", label="Variance expliquée (%)")
    _ax.plot(range(1, len(variance_cumulee) + 1), variance_cumulee * 100,
             color="red", marker="o", label="Variance cumulée (%)")
    _ax.axhline(80, color="grey", linestyle=":", linewidth=1, label="Seuil 80%")
    _ax.set_xlabel("Composante principale")
    _ax.set_ylabel("% de variance expliquée")
    _ax.set_title("Éboulis des valeurs propres — ACP sur le noyau numérique")
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return variance_cumulee, variance_expliquee


@app.cell
def _(variance_cumulee, variance_expliquee):
    for j, (v, c) in enumerate(zip(variance_expliquee, variance_cumulee), start=1):
        print(f"CP{j} : {v*100:.1f}% (cumulé {c*100:.1f}%)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    CP1 : 32,2% — CP2 : 17,2% (cumulé 49,4%) — CP3 : 16,9% (cumulé 66,3%) — CP4 : 16,3% (cumulé 82,6%) — CP5 : 16,2% (cumulé 98,9%) — CP6 : 1,1% (cumulé 100%). Quatre composantes quasi identiques en poids (CP2 à CP5, toutes entre 16,2% et 17,2%) confirment l'absence de hiérarchie claire entre les axes : il faut attendre CP6 pour voir une vraie chute.



    CP1 explique 32,2% de la variance, CP2 17,2% : à eux deux, ils ne couvrent que **49,4%** de l'information totale. Il faut 4 axes pour dépasser le seuil de 80% (CP1 à CP4 cumulent 82,6%), et la chute nette n'intervient qu'au 6ᵉ axe (1,1% seulement, contre 16,2% pour CP5). L'éboulis est donc plat sur les 5 premiers axes plutôt que décroissant : la variance se répartit de façon presque égale entre eux, signe qu'aucun petit sous-ensemble de dimensions ne concentre l'essentiel de l'information.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4. cercle des corrélations
    """)
    return


@app.cell
def _(X_acp, np, pca, plt):
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    _fig, _ax = plt.subplots(figsize=(7, 7))
    _circle = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--")
    _ax.add_artist(_circle)

    for i, var in enumerate(X_acp.columns):
        _ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                  head_width=0.03, color="steelblue")
        _ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, var, fontsize=10)

    _ax.axhline(0, color="grey", lw=0.8)
    _ax.axvline(0, color="grey", lw=0.8)
    _ax.set_xlim(-1.2, 1.2)
    _ax.set_ylim(-1.2, 1.2)
    _ax.set_xlabel(f"CP1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    _ax.set_ylabel(f"CP2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    _ax.set_title("Cercle des corrélations (CP1 x CP2)")
    _ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **CP1 ("taille de la commande") :** `revenue`, `unit_price` et `quantity` pointent tous vers la droite, confirmant leur lien mécanique déjà vu en 2.2 (r=0,68 et r=0,62 avec revenue), CP1 résume bien la valeur de la commande.

    **CP2 :** dominé par `delivery_days` (vers le haut) et `customer_rating` (vers le bas), à l'opposé l'un de l'autre sur cet axe. Attention à ne pas y voir une anti-corrélation entre les deux : la matrice de corrélation (partie 2.2) avait déjà montré qu'ils sont quasiment indépendants (r=-0,02). Cette position opposée reflète simplement le fait que ce sont les deux seules variables qui ne s'alignent pas sur CP1 : l'ACP leur construit un axe à part pour capter leur variance propre, sans que cela traduise un lien réel entre elles.

    **`discount` :** flèche minuscule, quasi au centre du cercle, cette variable est très mal représentée sur le plan CP1-CP2. Sa variance se répartit sur les axes suivants (CP3 à CP5), qui pèsent presque autant que CP2 (16-17% chacun).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.5. Projection des individus
    """)
    return


@app.cell
def _(composantes, df, plt):
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _scatter = _ax.scatter(composantes[:, 0], composantes[:, 1],
                            c=df["revenue"], cmap="viridis", alpha=0.5, s=15)
    plt.colorbar(_scatter, label="Revenue ($)")
    _ax.set_xlabel("CP1")
    _ax.set_ylabel("CP2")
    _ax.set_title("Projection des commandes sur les deux premiers axes")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Le nuage confirme visuellement le rôle de CP1 : le dégradé de couleur (revenue) progresse clairement de gauche (violet, faible revenue) à droite (jaune, fort revenue), sans aucun regroupement en sous-populations distinctes. Les commandes se répartissent en un nuage continu, homogène, aucun segment naturel de clientèle ne se détache sur ces deux axes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.6. Conclusion de la partie 3

    L'ACP confirme l'hypothèse posée en 3.1 : avec 49,4% de variance cumulée sur les deux premiers axes seulement (et 4 axes nécessaires pour dépasser 80%), il n'existe pas de dimension latente qui résume efficacement les 6 variables numériques. CP1 (32,2%) capture la "taille de la commande" (`revenue`, `unit_price`, `quantity`, mécaniquement liés), tandis que CP2 (17,2%) isole `delivery_days` et `customer_rating` sans que cela traduise une corrélation entre eux (r=-0,02, déjà établi en partie 2.2). `discount`, quant à elle, n'est bien représentée sur aucun des deux premiers axes.

    **Ce n'est pas un échec de la méthode, c'est un résultat en soi :** le comportement d'achat de cette boutique ne se réduit pas à deux ou trois facteurs cachés. Pour la direction, cela signifie que remise, délai de livraison, catégorie et région doivent continuer à être pilotés comme des leviers **indépendants** plutôt que via un indicateur composite, ce que confirment aussi les tests ANOVA de la partie 2, qui montrent des effets propres à chaque variable plutôt qu'un facteur commun.
    """)
    return


if __name__ == "__main__":
    app.run()
