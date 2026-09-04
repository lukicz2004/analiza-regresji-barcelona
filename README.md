**English** · [Polski](README.pl.md)

# FC Barcelona Financial Regression Analysis

An econometric analysis of FC Barcelona's debt over the 2010/11–2021/22 period,
originally prepared as part of a bachelor's thesis. The goal is to identify which
financial factors best explain the growth of the club's long-term debt.

## Research problem

Dependent variable: **long-term debt** (EUR million). Candidate explanatory
variables: player transfer amortization, wages, net transfer balance, and a
dummy variable marking the COVID-19 period. Predictors are standardized
(z-score), so the coefficients are directly comparable.

## Models

Four OLS models (`statsmodels`) were estimated on a sample of n = 12 seasons:

| Model | Explanatory variables | Adjusted R² |
|-------|-----------------------|:---:|
| M1 | Amortization + COVID | 0.732 |
| M2 | Amortization + Wages + COVID | 0.699 |
| M3 | Amortization | 0.696 |
| **M4 (main)** | **Transfer balance + Wages + COVID** | **0.813** |

The main model **M4** explains about 86% of the variance in debt (R² = 0.864),
and all three predictors are statistically significant. Diagnostics include the
Durbin-Watson test (autocorrelation), the Jarque-Bera test (residual normality),
and VIF (multicollinearity).

## Structure

```
.
├── regresja_barcelona.py   # full analysis: models, tables, charts, export
├── data/
│   └── fcb_dane_regresja.csv
├── outputs/                # generated results (Excel + PNG charts)
├── requirements.txt
├── README.md               # English
└── README.pl.md            # Polish
```

Data come from the club's financial statements and public sources
(including La Liga reports). Column names in the dataset are in Polish.

## Running

```bash
pip install -r requirements.txt
python regresja_barcelona.py
```

The script prints the tables to the console and writes to `outputs/`:

- `wyniki_regresji.xlsx` — model comparison, coefficients, descriptive statistics, VIF (each table on its own sheet),
- trend charts: `wykres_dlug_w_czasie.png` (debt over time), `wykres_amortyzacja_w_czasie.png` (amortization over time), `wykres_przychody_place.png` (revenue vs. wages), `wykres_scatter_amortyzacja.png` (amortization vs. debt),
- diagnostic charts: `wykres_reszty_m4.png` (residuals), `wykres_qq_m4.png` (Q-Q), `wykres_korelacji.png` (correlation matrix).

## Technologies

Python, pandas, NumPy, statsmodels, scikit-learn, matplotlib, seaborn, SciPy.
