import math

import numpy as np
import pandas as pd
from scipy import stats


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
