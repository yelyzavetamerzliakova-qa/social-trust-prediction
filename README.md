# Social Trust Prediction Tool

## What does it do?

This Python tool predicts whether a respondent is likely to answer that **most people can be trusted**, based on their demographic, social and personal characteristics. It uses a trained MLP neural network fitted to the **Joint EVS/WVS 2017–2022, version 5.0** survey data.

**Predicted result:** `1` = Most people can be trusted; `0` = Cannot be too careful.

The tool also returns the model's estimated probability of class `1`. These are estimates, not certain facts about a person.

## Required files

Keep the following files in the **same folder**:

- `social_trust_predictor.py` — reusable prediction function
- `social_trust_model.joblib` — fitted model and preprocessing
- `supported_countries.json` — country codes accepted by the tool
- `requirements.txt` — required Python package versions
- `README.md` — instructions (this document)

## Installation

Install Python and run in a terminal in that folder:

```bash
pip install -r requirements.txt
```

## Allowed inputs and survey codes

Supply **one row of data with all 12 feature columns**, using the original survey codes. The numbers are **codes**, not necessarily continuous scales.

| Feature | Meaning | Valid codes or values |
|---|---|---|
| `X003` | Age | Age in years; the provided dataset contains **16–82**, where **82 means 82 or older** (top-coded) |
| `X001` | Sex as recorded by the survey | **1** = Male; **2** = Female |
| `X025R` | Education (recoded) | **1** = Lower; **2** = Middle; **3** = Upper |
| `X028` | Employment status | **1** = Full-time (30+ hours/week); **2** = Part-time (<30 hours/week); **3** = Self-employed; **4** = Retired/pensioned; **5** = Homemaker/not otherwise employed; **6** = Student; **7** = Unemployed; **8** = Other |
| `X007` | Marital status | **1** = Married; **2** = Living together as married; **3** = Divorced; **4** = Separated; **5** = Widowed; **6** = Single/never married |
| `A009` | Self-reported health | **1** = Very good; **2** = Good; **3** = Fair; **4** = Poor; **5** = Very poor |
| `A074` | Belongs to sports/recreation organization | **0** = Not mentioned/not a member; **1** = Mentioned/member. Active and inactive members are both coded `1` in the harmonized variable. |
| `A080_01` | Belongs to humanitarian/charitable organization | **0** = Not mentioned/not a member; **1** = Mentioned/member. Active and inactive members are both coded `1`. |
| `F034` | Self-described religious status | **1** = Religious person; **2** = Not a religious person; **3** = Convinced atheist |
| `A173` | Perceived freedom of choice and control over life | Integers **1–10**: **1** = None at all; **10** = A great deal |
| `A002` | Importance of friends in respondent's life | **1** = Very important; **2** = Rather important; **3** = Not very important; **4** = Not at all important |
| `cntry` | Country/territory | Country/territory **numeric identifier**, not a 1–10 scale. Examples: **203** = Czechia; **276** = Germany; **752** = Sweden; **840** = United States; **826** = Great Britain; **909** = Northern Ireland. |

### All `cntry` values appearing in the training dataset

The model was developed with respondents from **92 countries/territories**. Use an appropriate code from this observed set:

```
8, 20, 31, 32, 36, 40, 50, 51, 68, 70, 76, 100, 104, 112, 124, 152,
156, 158, 170, 191, 196, 203, 208, 218, 231, 233, 246, 250, 268, 276,
300, 320, 344, 348, 352, 356, 360, 364, 368, 380, 392, 398, 400, 404,
410, 417, 422, 428, 434, 440, 446, 458, 462, 484, 496, 499, 504, 528,
554, 558, 566, 578, 586, 604, 608, 616, 620, 630, 642, 643, 688, 702,
703, 704, 705, 716, 724, 752, 756, 762, 764, 788, 792, 804, 807, 818,
826, 840, 858, 860, 862, 909
```

### Missing and invalid values

The original survey uses special negative codes: `-1` = Don't know; `-2` = No answer; `-3` = Not applicable; `-4` = Not asked; `-5` = Other missing. **Do not supply these codes to this prediction tool**; it is designed to reject them. For an unknown age, `None`/`NaN` may be used, and the fitted imputer fills it from the training data. Other missing values should be handled consistently with the saved preprocessing pipeline; do not invent category codes.

**Input validation:** The predictor checks required fields, numeric survey codes, valid categorical values and ranges, and the supported country list. The survey age value 82 represents ages 82 or older. Missing values that pass validation are handled by the training-time imputation in the fitted pipeline.

## Example use

```python
import pandas as pd
from social_trust_predictor import predict_trust

new_person = pd.DataFrame([{
    'X003': 45,       # Age (82 means 82 or older)
    'X001': 1,        # Male
    'X025R': 3,       # Upper education
    'X028': 1,        # Full-time work
    'X007': 2,        # Living together as married
    'A009': 1,        # Very good health
    'A074': 1,        # Sports/recreation organization member
    'A080_01': 1,     # Humanitarian/charity organization member
    'F034': 1,        # Religious person
    'A173': 9,        # High perceived freedom of choice
    'A002': 1,        # Friends are very important
    'cntry': 752      # Sweden
}])

prediction, probability = predict_trust(new_person)

print('Predicted trust:', prediction)
print('Estimated probability of trust:', round(probability * 100, 2), '%')
```

## Limitations

Predictions are **not facts** about an individual and do not demonstrate what **causes** social trust. The model reflects patterns in the survey; it may perform differently across countries, demographic groups and future populations. Country codes outside the training data should be refused or explicitly flagged, not silently treated as known categories. Model probabilities should not be interpreted as perfectly calibrated without a calibration check.

## Source for the survey codes

EVS/WVS (2024). *Joint EVS/WVS 2017–2022 Dataset – Variable Report (Documentation/Tables)*, version June 2024 (v5), corresponding to dataset version 5.0.0. GESIS Data Archive, Cologne. DOI: `10.4232/1.14320`.

Codebook: https://access.gesis.org/dbk/69548?download_purpose=-99
