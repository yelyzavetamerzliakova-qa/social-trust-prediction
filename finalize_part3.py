"""Final verification and packaging of the user's local social trust production tool.

Place this file in ~/social_trust and run:
    python finalize_part3.py

No survey CSV, notebook, or training variables are needed.
"""
from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "social_trust_predictor.py",
    "social_trust_model.joblib",
    "supported_countries.json",
]


def main():
    print("Part III: independent production checks")
    print("Project folder:", ROOT)
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Files missing from project folder: " + ", ".join(missing))

    # The downloaded file has a descriptive filename; normalize it for users.
    readme = ROOT / "README.md"
    alternate = ROOT / "README_Social_Trust_With_Feature_Codes.md"
    if not readme.exists() and alternate.exists():
        shutil.copy2(alternate, readme)
        print("Created README.md from your existing README file.")
    if not readme.exists():
        raise FileNotFoundError("README.md not found in the project folder.")

    # A README drafted before the expanded validation still contains an outdated note.
    content = readme.read_text(encoding="utf-8")
    old = ("**Important:** The list above documents acceptable input codes, but documentation is not validation. "
           "The current simple predictor has checks for a few fields only; additional input checks "
           "(especially categorical codes and country) should be added before calling the tool production-ready. "
           "The dataset's age variable is top-coded at 82, so the earlier generic check `0–120` "
           "is not a suitable final production age validation rule.")
    new = ("**Input validation:** The predictor checks required fields, numeric survey codes, "
           "valid categorical values and ranges, and the supported country list. "
           "The survey age value 82 represents ages 82 or older. Missing values that pass "
           "validation are handled by the training-time imputation in the fitted pipeline.")
    if old in content:
        content = content.replace(old, new)
    if "`supported_countries.json`" not in content.split("## Installation")[0]:
        content = content.replace("- `requirements.txt` — required Python package versions", 
                                  "- `supported_countries.json` — country codes accepted by the tool\n"
                                  "- `requirements.txt` — required Python package versions")
    readme.write_text(content, encoding="utf-8")

    versions = {
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "pandas": importlib.metadata.version("pandas"),
        "joblib": importlib.metadata.version("joblib"),
    }
    (ROOT / "requirements.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in versions.items()),
        encoding="utf-8",
    )

    # Because this file runs as a standalone script in ROOT, this imports the copy
    # adjacent to the saved .joblib, not a duplicate from an old notebook folder.
    from social_trust_predictor import model, predict_trust

    supported = json.loads((ROOT / "supported_countries.json").read_text())
    country = 752 if 752 in supported else int(supported[0])
    person = pd.DataFrame([{
        "X003": 45, "X001": 1, "X025R": 3, "X028": 1,
        "X007": 2, "A009": 1, "A074": 1, "A080_01": 1,
        "F034": 1, "A173": 9, "A002": 1, "cntry": country,
    }])

    results = []

    def check(description, function):
        try:
            function()
        except Exception as exc:
            results.append(("FAIL", description, f"{type(exc).__name__}: {exc}"))
            print("FAIL", description, "-", exc)
        else:
            results.append(("PASS", description, ""))
            print("PASS", description)

    def expect_value_error(data):
        try:
            predict_trust(data)
        except ValueError:
            return
        raise AssertionError("Invalid data were accepted without a ValueError")

    def identical_to_saved_model():
        pred, probability = predict_trust(person)
        assert pred == int(model.predict(person)[0]), "Class mismatch"
        assert abs(probability - float(model.predict_proba(person)[0, 1])) < 1e-10, "Score mismatch"
        assert isinstance(pred, int) and 0 <= probability <= 1

    def missing_age():
        data = person.copy()
        data.loc[data.index[0], "X003"] = float("nan")
        prediction, probability = predict_trust(data)
        assert prediction in (0, 1) and 0 <= probability <= 1

    def repeatability():
        assert predict_trust(person) == predict_trust(person)

    def bad_value(column, value):
        data = person.copy()
        data.loc[data.index[0], column] = value
        expect_value_error(data)

    def missing_column():
        expect_value_error(person.drop(columns=["X003"]))

    def with_target():
        data = person.copy()
        data["A165"] = 1
        expect_value_error(data)

    check("Prediction and score match the saved model", identical_to_saved_model)
    check("Same input returns the same prediction", repeatability)
    check("Missing age is imputed", missing_age)
    check("Out-of-range age is rejected", lambda: bad_value("X003", 200))
    check("Out-of-range education is rejected", lambda: bad_value("X025R", 99))
    check("Unknown country is rejected", lambda: bad_value("cntry", 999))
    check("Negative missing survey code is rejected", lambda: bad_value("X025R", -2))
    check("Missing required column is rejected", missing_column)
    check("Target A165 is rejected as an input", with_target)

    passed = sum(status == "PASS" for status, _, _ in results)
    report = [
        "SOCIAL TRUST PREDICTION TOOL - PRODUCTION TEST RESULTS",
        f"Run at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Python: {sys.version.split()[0]}",
        *(f"{name}: {version}" for name, version in versions.items()),
        f"Project folder: {ROOT}",
        "",
        *(f"{status}: {name}" + (f" | {detail}" if detail else "") for status, name, detail in results),
        "",
        f"Result: {passed}/{len(results)} checks passed.",
        "Predictions are estimates, not validated causal effects or calibrated probabilities.",
    ]
    (ROOT / "production_test_results.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    if passed != len(results):
        raise RuntimeError("Some tests failed. See production_test_results.txt; fix those before submitting.")

    # A report PDF is supplied separately: copy it into this folder before final packaging.
    pdf = ROOT / "Part_III_Production_Report.pdf"
    if not pdf.exists():
        print("\nAll tests passed, but the Part III PDF is not in the folder yet.")
        print("Download Part_III_Production_Report.pdf and place it in:", ROOT)
        print("Then run this script again to create the submission ZIP.")
        return

    attachments = REQUIRED + [
        "README.md", "requirements.txt", "production_test_results.txt",
        "Part_III_Production_Report.pdf", "finalize_part3.py",
    ]
    archive = ROOT / "Social_Trust_Part_III_Submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for name in attachments:
            out.write(ROOT / name, arcname=f"social_trust/{name}")

    print("\nSUCCESS: All", len(results), "production tests passed.")
    print("Submission created:", archive)
    print("This ZIP does not include survey microdata, passwords, or the training notebook.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nFINALIZATION STOPPED:", exc)
        sys.exit(1)
