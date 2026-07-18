import pandas as pd


def charger_donnee(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce",  # une erreur est remplace par NAT(not at time)
    )

    return df
