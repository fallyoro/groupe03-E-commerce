import pandas as pd


def charger_donnee(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["order_date"] = pd.to_datetime(
        df["order_date"], format="%m/%d/%Y", errors="coerce"
    )

    return df
