# FC Barcelona financial regression analysis (2010/11-2021/22)
# Dependent variable: long-term debt. Predictors standardized (z-score).
# Run cell by cell (#%%) or all at once.

#%% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from scipy.stats import jarque_bera, probplot
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

os.makedirs("outputs", exist_ok=True)

# FC Barcelona colours
GRANAT, KARMAZYN, ZLOTY = "#004D98", "#A50044", "#EDBB00"
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlecolor": GRANAT,
})

#%% Load data and derived variables
df = pd.read_csv("data/fcb_dane_regresja.csv", sep=";")
df["Bilans_TransferowyMln"] = (
    df["Bilans_TransferowyMln"].astype(str).str.strip()
    .str.replace(",", ".").replace("nan", np.nan).astype(float)
)
df["Amortyzacja_mln"] = df["Amortyzacja_graczy_tys_eur"] / 1000
df["Place_mln"] = df["Place_tys_eur"] / 1000
df["Przychody_mln"] = df["Przychody_tys_eur"] / 1000

# Regression sample: 12 seasons with available long-term debt
df12 = df.iloc[:12].copy()
y = df12["Dlug_dlugoterminowy_mln_eur"]

print(df12[["Sezon", "Dlug_dlugoterminowy_mln_eur", "Amortyzacja_mln",
            "Place_mln", "Bilans_TransferowyMln", "COVID_dummy"]].to_string(index=False))

#%% Descriptive statistics (n=12)
zmienne = {
    "Dlug_dlugoterminowy_mln_eur": "Dlug dlugoterminowy (mln EUR)",
    "Amortyzacja_mln": "Amortyzacja graczy (mln EUR)",
    "Place_mln": "Place (mln EUR)",
    "Bilans_TransferowyMln": "Bilans transferowy (mln EUR)",
    "COVID_dummy": "COVID (dummy)",
}
wiersze = []
for kol, opis in zmienne.items():
    s = df12[kol]
    wiersze.append({
        "Zmienna": opis, "Srednia": round(s.mean(), 2), "Odch. std.": round(s.std(), 2),
        "Min": round(s.min(), 2), "Mediana": round(s.median(), 2), "Max": round(s.max(), 2),
        "Skosnosc": round(s.skew(), 3), "Kurtoza": round(s.kurt(), 3),
    })
statystyki = pd.DataFrame(wiersze).set_index("Zmienna")
print(statystyki.to_string())

#%% Model M1: amortization + COVID
X1 = pd.DataFrame(StandardScaler().fit_transform(df12[["Amortyzacja_graczy_tys_eur"]]),
                  columns=["Amortyzacja_graczy_tys_eur"], index=df12.index)
X1["COVID_dummy"] = df12["COVID_dummy"].values
model_m1 = sm.OLS(y, sm.add_constant(X1)).fit()
print(model_m1.summary())

#%% Model M2: amortization + wages + COVID
X2 = pd.DataFrame(StandardScaler().fit_transform(df12[["Amortyzacja_graczy_tys_eur", "Place_tys_eur"]]),
                  columns=["Amortyzacja_graczy_tys_eur", "Place_tys_eur"], index=df12.index)
X2["COVID_dummy"] = df12["COVID_dummy"].values
model_m2 = sm.OLS(y, sm.add_constant(X2)).fit()
print(model_m2.summary())

#%% Model M3: amortization only (no COVID)
X3 = pd.DataFrame(StandardScaler().fit_transform(df12[["Amortyzacja_graczy_tys_eur"]]),
                  columns=["Amortyzacja_graczy_tys_eur"], index=df12.index)
model_m3 = sm.OLS(y, sm.add_constant(X3)).fit()
print(model_m3.summary())

#%% Model M4 (main): transfer balance + wages + COVID
X4 = pd.DataFrame(StandardScaler().fit_transform(df12[["Bilans_TransferowyMln", "Place_tys_eur"]]),
                  columns=["Bilans_TransferowyMln", "Place_tys_eur"], index=df12.index)
X4["COVID_dummy"] = df12["COVID_dummy"].values
model_m4 = sm.OLS(y, sm.add_constant(X4)).fit()
print(model_m4.summary())

#%% Model comparison table
modele = [
    ("M1: Amortyzacja + COVID", model_m1),
    ("M2: Amortyzacja + Place + COVID", model_m2),
    ("M3: Amortyzacja", model_m3),
    ("M4: Bilans + Place + COVID", model_m4),
]
wiersze = []
for nazwa, m in modele:
    wiersze.append({
        "Model": nazwa, "R2": round(m.rsquared, 3), "Adj. R2": round(m.rsquared_adj, 3),
        "AIC": round(m.aic, 2), "BIC": round(m.bic, 2), "F-stat": round(m.fvalue, 3),
        "F p-val": round(m.f_pvalue, 4), "DW": round(durbin_watson(m.resid), 3),
        "JB p-val": round(jarque_bera(m.resid).pvalue, 4),
    })
porownanie = pd.DataFrame(wiersze).set_index("Model")
print(porownanie.to_string())

#%% Coefficients of each model (beta, p-value, significance)
def gwiazdki(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""

def tabela_modelu(model):
    wiersze = []
    for var in model.params.index:
        p = model.pvalues[var]
        wiersze.append({"Zmienna": var, "beta": round(model.params[var], 3),
                        "p-value": round(p, 4), "istotnosc": gwiazdki(p)})
    return pd.DataFrame(wiersze).set_index("Zmienna")

t1 = tabela_modelu(model_m1)
t2 = tabela_modelu(model_m2)
t3 = tabela_modelu(model_m3)
t4 = tabela_modelu(model_m4)
print("\nModel M4 (glowny):")
print(t4.to_string())

#%% VIF for the main model (multicollinearity)
X_vif = StandardScaler().fit_transform(df12[["Bilans_TransferowyMln", "Place_tys_eur"]])
vif = pd.DataFrame({
    "Zmienna": ["Bilans_TransferowyMln", "Place_tys_eur"],
    "VIF": [round(variance_inflation_factor(X_vif, i), 3) for i in range(X_vif.shape[1])],
}).set_index("Zmienna")
print(vif.to_string())

#%% Export tables to Excel (each on its own sheet)
with pd.ExcelWriter("outputs/wyniki_regresji.xlsx", engine="openpyxl") as writer:
    statystyki.to_excel(writer, sheet_name="Statystyki opisowe")
    porownanie.to_excel(writer, sheet_name="Porownanie modeli")
    vif.to_excel(writer, sheet_name="VIF M4")
    t1.to_excel(writer, sheet_name="M1")
    t2.to_excel(writer, sheet_name="M2")
    t3.to_excel(writer, sheet_name="M3")
    t4.to_excel(writer, sheet_name="M4")
print("Zapisano: outputs/wyniki_regresji.xlsx")

#%% Chart 1 - long-term debt over time
x = range(len(df12))
plt.figure(figsize=(10, 5))
plt.plot(x, df12["Dlug_dlugoterminowy_mln_eur"], marker="o", color=GRANAT, linewidth=2, label="Dlug dlugoterminowy")
plt.axvspan(6.5, 11.5, alpha=0.10, color=KARMAZYN, label="Kryzys zadluzenia 2017-2022")
plt.axvspan(8.5, 11.5, alpha=0.12, color=ZLOTY, label="Okres COVID")
plt.xticks(list(x), df12["Sezon"], rotation=45, ha="right")
plt.ylabel("mln EUR")
plt.title("Dlug dlugoterminowy FC Barcelona 2010/11-2021/22")
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)} mln"))
plt.legend()
plt.tight_layout()
plt.savefig("outputs/wykres_dlug_w_czasie.png", dpi=150)
plt.show()

#%% Chart 2 - player amortization over time
plt.figure(figsize=(10, 5))
plt.plot(x, df12["Amortyzacja_mln"], marker="o", color=KARMAZYN, linewidth=2)
plt.fill_between(list(x), df12["Amortyzacja_mln"], alpha=0.15, color=KARMAZYN)
plt.xticks(list(x), df12["Sezon"], rotation=45, ha="right")
plt.ylabel("mln EUR")
plt.title("Amortyzacja graczy FC Barcelona 2010/11-2021/22")
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)} mln"))
plt.tight_layout()
plt.savefig("outputs/wykres_amortyzacja_w_czasie.png", dpi=150)
plt.show()

#%% Chart 3 - revenue vs wages (2010-2025)
x_all = range(len(df))
plt.figure(figsize=(12, 5))
plt.plot(x_all, df["Przychody_mln"], marker="o", color=GRANAT, linewidth=2, label="Przychody")
plt.plot(x_all, df["Place_mln"], marker="s", color=KARMAZYN, linewidth=2, label="Place")
plt.axvline(x=11.5, color="gray", linestyle="--", linewidth=1, label="Dane szacowane ->")
plt.axvspan(8.5, 11.5, alpha=0.10, color=ZLOTY, label="Okres COVID")
plt.xticks(list(x_all), df["Sezon"], rotation=45, ha="right")
plt.ylabel("mln EUR")
plt.title("Przychody a place FC Barcelona 2010/11-2024/25")
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)} mln"))
plt.legend()
plt.tight_layout()
plt.savefig("outputs/wykres_przychody_place.png", dpi=150)
plt.show()

#%% Chart 4 - scatter: amortization vs debt + regression line
plt.figure(figsize=(8, 6))
plt.scatter(df12["Amortyzacja_mln"], df12["Dlug_dlugoterminowy_mln_eur"], color=GRANAT, s=70, zorder=3)
for _, row in df12.iterrows():
    plt.annotate(row["Sezon"], (row["Amortyzacja_mln"], row["Dlug_dlugoterminowy_mln_eur"]),
                 textcoords="offset points", xytext=(6, 4), fontsize=8, color="gray")
coef = np.polyfit(df12["Amortyzacja_mln"], df12["Dlug_dlugoterminowy_mln_eur"], 1)
xl = np.linspace(df12["Amortyzacja_mln"].min(), df12["Amortyzacja_mln"].max(), 100)
plt.plot(xl, np.polyval(coef, xl), color=KARMAZYN, linewidth=1.8,
         label=f"Regresja: y = {coef[0]:.2f}x + {coef[1]:.1f}")
plt.xlabel("Amortyzacja graczy (mln EUR)")
plt.ylabel("Dlug dlugoterminowy (mln EUR)")
plt.title("Amortyzacja graczy a dlug dlugoterminowy")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/wykres_scatter_amortyzacja.png", dpi=150)
plt.show()

#%% Chart 5 - residuals vs fitted values (M4)
plt.figure(figsize=(8, 6))
plt.axhline(0, color=GRANAT, linestyle="--", linewidth=2)
plt.scatter(model_m4.fittedvalues, model_m4.resid, color=KARMAZYN, s=70, edgecolors="white", zorder=3)
for xf, yr, sezon in zip(model_m4.fittedvalues, model_m4.resid, df12["Sezon"]):
    plt.annotate(sezon, (xf, yr), textcoords="offset points", xytext=(5, 5), fontsize=8, color=GRANAT)
plt.xlabel("Wartosci dopasowane (mln EUR)")
plt.ylabel("Reszty (mln EUR)")
plt.title("Reszty vs wartosci dopasowane (model M4)")
plt.tight_layout()
plt.savefig("outputs/wykres_reszty_m4.png", dpi=150)
plt.show()

#%% Chart 6 - Q-Q of residuals (M4)
plt.figure(figsize=(7, 5))
(osq, osr), (slope, intercept, _) = probplot(model_m4.resid, dist="norm")
plt.scatter(osq, osr, color=GRANAT, s=65, edgecolors=ZLOTY, linewidths=0.8, zorder=3)
plt.plot(osq, slope * np.array(osq) + intercept, color=KARMAZYN, linewidth=1.8)
plt.xlabel("Kwantyle teoretyczne N(0,1)")
plt.ylabel("Kwantyle empiryczne reszt")
plt.title("Wykres Q-Q reszt (model M4)")
plt.tight_layout()
plt.savefig("outputs/wykres_qq_m4.png", dpi=150)
plt.show()

#%% Chart 7 - correlation matrix
kol_corr = ["Amortyzacja_graczy_tys_eur", "Bilans_TransferowyMln",
            "Przychody_tys_eur", "Place_tys_eur", "COVID_dummy"]
corr = df12[kol_corr].corr()
corr.index = corr.columns = ["Amortyzacja", "Bilans transf.", "Przychody", "Place", "COVID"]
barca_cmap = LinearSegmentedColormap.from_list("fcb", [KARMAZYN, "white", GRANAT])
plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap=barca_cmap, vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size": 10})
plt.title("Macierz korelacji zmiennych")
plt.tight_layout()
plt.savefig("outputs/wykres_korelacji.png", dpi=150)
plt.show()

print("\nGotowe. Wyniki zapisane w folderze outputs/.")
