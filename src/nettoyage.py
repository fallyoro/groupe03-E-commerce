import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler


def detecter_outliers_iqr(df, colonne):
    print(f"=== DÉTECTION DES OUTLIERS — {colonne} ===\n")

    serie = df[colonne]

    # Quartiles
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)

    # Intervalle interquartile
    iqr = q3 - q1

    # Bornes
    borne_inf = q1 - 1.5 * iqr
    borne_sup = q3 + 1.5 * iqr

    # Détection
    outliers = (serie < borne_inf) | (serie > borne_sup)

    print(f"Q1  = {q1:.2f}")  # 2 chiffre apres la virgule
    print(f"Q3  = {q3:.2f}")
    print(f"IQR = {iqr:.2f}")

    print(f"\nBornes IQR : [{borne_inf:.2f}, {borne_sup:.2f}]")

    print(f"Outliers : {outliers.sum()} " f"({outliers.sum()/len(serie)*100:.2f}%)")

    return outliers


def detect_outliers_zscore(
    data: pd.DataFrame | pd.Series,
    seuil: float = 3.0,
    exclure: list[str] = ["id", "order_id", "order_date", "customer_id"],
) -> dict:
    """Détecte les outliers par score Z."""
    if isinstance(data, pd.Series):
        data = pd.DataFrame({data.name: data})
    cols = [
        c for c in data.select_dtypes(include=[np.number]).columns if c not in exclure
    ]
    resultats = {}
    for c in cols:
        data_clean = data[c].dropna()
        if len(data_clean) < 3:
            continue
        z_scores = np.abs(np.asarray(stats.zscore(data_clean)))
        outliers_mask = z_scores > seuil
        n_outliers = int(outliers_mask.sum())

        if n_outliers > 0:
            resultats[c] = {
                "n_outliers": n_outliers,
                "pct_outliers": (n_outliers / len(data_clean)) * 100,
                "valeurs_outliers": (
                    data_clean.to_numpy()[outliers_mask].tolist()
                    if n_outliers > 0
                    else []
                ),
            }
        print(
            f"{c:20s} | Outliers: {n_outliers:3d} ({n_outliers/len(data_clean)*100:5.2f}%) | Min: {data_clean.min():8.2f} | Max: {data_clean.max():8.2f}"
        )
    return resultats


def analyser_variables_categorielles(data: pd.DataFrame) -> pd.DataFrame:
    """Recense les variables catégorielles et leur cardinalité."""
    cols_cat = data.select_dtypes(include=["object", "category"]).columns

    infos = []
    for c in cols_cat:
        n_uniques = data[c].nunique()
        infos.append(
            {
                "colonne": c,
                "n_valeurs_uniques": n_uniques,
                "pct_uniques": n_uniques / len(data) * 100,
            }
        )

    resultat = pd.DataFrame(infos).sort_values("n_valeurs_uniques")
    return resultat


def standardiser(
    data: pd.DataFrame,
    colonnes: list[str] | None = None,
    exclure: list[str] = ["order_id", "customer_id"],
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardise les colonnes numériques (moyenne=0, écart-type=1)."""
    data = data.copy()

    if colonnes is None:
        colonnes = [
            c
            for c in data.select_dtypes(include=[np.number]).columns
            if c not in exclure
        ]

    scaler = StandardScaler()
    data[colonnes] = scaler.fit_transform(data[colonnes])

    return data, scaler


def controler_revenue(data: pd.DataFrame, tolerance: float = 0.01) -> pd.DataFrame:
    """Vérifie la cohérence entre revenue et quantity*unit_price*(1-discount)."""
    data = data.copy()
    data["revenue_calcule"] = (
        data["quantity"] * data["unit_price"] * (1 - data["discount"])
    )
    data["ecart"] = (data["revenue"] - data["revenue_calcule"]).abs()
    data["ecart_pct"] = data["ecart"] / data["revenue_calcule"] * 100

    incoherents = data[data["ecart_pct"] > tolerance]
    print(
        f"{len(incoherents)} lignes ({len(incoherents)/len(data)*100:.2f}%) avec écart > {tolerance}%"
    )
    print(
        f"Écart moyen : {data['ecart'].mean():.4f} | Écart max : {data['ecart'].max():.4f}"
    )
    return incoherents
