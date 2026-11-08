
import json
import joblib
import pandas as pd
from pathlib import Path


folder = Path(__file__).resolve().parent

model = joblib.load(folder / "social_trust_model.joblib")

with open(folder / "supported_countries.json") as file:
    supported_countries = json.load(file)


def predict_trust(person):


    if not isinstance(person, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")


    if len(person) != 1:
        raise ValueError("Enter exactly one person.")


    required_columns = list(model.feature_names_in_)

    missing_columns = [
        col for col in required_columns
        if col not in person.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")


    extra_columns = [
        col for col in person.columns
        if col not in required_columns
    ]

    if extra_columns:
        raise ValueError(f"Unexpected columns: {extra_columns}")


    try:
        person = person[required_columns].apply(
            pd.to_numeric, errors="raise"
        ).astype(float)
    except (TypeError, ValueError):
        raise ValueError("Features must contain numeric survey codes.")


    if person.isin([-1, -2, -3, -4, -5]).any().any():
        raise ValueError("Invalid survey code detected.")


    for column in required_columns:
        values = person[column].dropna()

        if not (values % 1 == 0).all():
            raise ValueError(f"{column} must be a whole number.")


    valid_ranges = {
        "X003": (16, 82),
        "A173": (1, 10),
        "A002": (1, 4)
    }

    for column, (minimum, maximum) in valid_ranges.items():
        if not person[column].dropna().between(minimum, maximum).all():
            raise ValueError(
                f"{column} must be between {minimum} and {maximum}."
            )


    valid_codes = {
        "X001": [1, 2],
        "X025R": [1, 2, 3],
        "X028": list(range(1, 9)),
        "X007": list(range(1, 7)),
        "A009": list(range(1, 6)),
        "A074": [0, 1],
        "A080_01": [0, 1],
        "F034": [1, 2, 3]
    }

    for column, allowed in valid_codes.items():
        if not person[column].dropna().isin(allowed).all():
            raise ValueError(
                f"Invalid value for {column}. Allowed: {allowed}"
            )


    if not person["cntry"].dropna().isin(supported_countries).all():
        raise ValueError("Country code was not in the training data.")


    prediction = int(model.predict(person)[0])
    probability = float(model.predict_proba(person)[0, 1])

    return prediction, probability
