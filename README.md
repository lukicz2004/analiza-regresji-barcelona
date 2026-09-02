# Analiza regresji finansów FC Barcelona

Analiza ekonometryczna zadłużenia FC Barcelona w latach 2010/11–2021/22,
przygotowana w ramach pracy licencjackiej. Celem jest zbadanie, które czynniki
finansowe najlepiej tłumaczą wzrost długu długoterminowego klubu.

## Problem badawczy

Zmienna zależna: **dług długoterminowy** (mln EUR). Rozważane zmienne
objaśniające: amortyzacja transferów graczy, płace, saldo transferowe oraz
zmienna zero-jedynkowa oznaczająca okres pandemii COVID-19. Predyktory są
standaryzowane (z-score), więc współczynniki są bezpośrednio porównywalne.

## Modele

Oszacowano cztery modele OLS (`statsmodels`) na próbie n = 12 sezonów:

| Model | Zmienne objaśniające | R² skorygowane |
|-------|----------------------|:---:|
| M1 | Amortyzacja + COVID | 0,732 |
| M2 | Amortyzacja + Płace + COVID | 0,699 |
| M3 | Amortyzacja | 0,696 |
| **M4 (główny)** | **Saldo transferowe + Płace + COVID** | **0,813** |

Model główny **M4** wyjaśnia ok. 86% zmienności długu (R² = 0,864), a wszystkie
trzy predyktory są istotne statystycznie. Diagnostyka obejmuje test
Durbina-Watsona (autokorelacja), test Jarque-Bera (normalność reszt) oraz VIF
(współliniowość).

## Struktura

```
.
├── regresja_barcelona.py   # cała analiza: modele, tabele, wykresy, eksport
├── data/
│   └── fcb_dane_regresja.csv
├── outputs/                # generowane wyniki (Excel + wykresy PNG)
├── requirements.txt
└── README.md
```

Dane pochodzą ze sprawozdań finansowych klubu i źródeł publicznych
(m.in. raporty La Liga).

## Uruchomienie

```bash
pip install -r requirements.txt
python regresja_barcelona.py
```

Skrypt wypisze tabele w konsoli oraz zapisze do `outputs/`:

- `wyniki_regresji.xlsx` — porównanie modeli, współczynniki, statystyki opisowe, VIF (każda tabela w osobnym arkuszu),
- wykresy trendów: `wykres_dlug_w_czasie.png`, `wykres_amortyzacja_w_czasie.png`, `wykres_przychody_place.png`, `wykres_scatter_amortyzacja.png`,
- wykresy diagnostyczne: `wykres_reszty_m4.png`, `wykres_qq_m4.png`, `wykres_korelacji.png`.

## Technologie

Python, pandas, NumPy, statsmodels, scikit-learn, matplotlib, seaborn, SciPy.
