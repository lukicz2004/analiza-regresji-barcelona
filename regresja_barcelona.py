"""
Analiza regresji finansow FC Barcelona (2010/11-2021/22).

Modeluje dlug dlugoterminowy klubu (zmienna zalezna) w zaleznosci od
amortyzacji transferow, plac, salda transferowego i efektu pandemii COVID.
Oszacowane sa cztery modele OLS na zmiennych standaryzowanych; model glowny
to M4 (saldo transferowe + place + COVID).

Skrypt:
  - wczytuje dane z data/fcb_dane_regresja.csv,
  - buduje modele M1-M4 i tabele porownawcza,
  - liczy VIF (wspolliniowosc) dla modelu glownego,
  - zapisuje tabele do outputs/wyniki_regresji.xlsx,
  - generuje wykresy diagnostyczne do outputs/.

Uruchomienie:  python regresja_barcelona.py
"""

from pathlib import Path

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

# --- Konfiguracja -----------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "fcb_dane_regresja.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

GRANAT, KARMAZYN, ZLOTY = "#004D98", "#A50044", "#EDBB00"
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlecolor": GRANAT,
})
FCB_CMAP = LinearSegmentedColormap.from_list("fcb", [KARMAZYN, "white", GRANAT])


# --- Dane -------------------------------------------------------------------
def wczytaj_dane():
    df = pd.read_csv(DATA, sep=";", encoding="utf-8")
    df["Bilans_TransferowyMln"] = (
        df["Bilans_TransferowyMln"].astype(str).str.strip()
        .str.replace(",", ".").replace("nan", np.nan).astype(float)
    )
    df["Obciazenie_Placowe"] = df["Place_tys_eur"] / df["Przychody_tys_eur"]
    df["Koszty_Operacyjne"] = df["Place_tys_eur"] + df["Amortyzacja_graczy_tys_eur"]
    df["Amortyzacja_mln"] = df["Amortyzacja_graczy_tys_eur"] / 1000
    df["Place_mln"] = df["Place_tys_eur"] / 1000
    df["Przychody_mln"] = df["Przychody_tys_eur"] / 1000
    return df


# --- Modele -----------------------------------------------------------------
def standaryzuj(df12, kolumny):
    """Standaryzacja predyktorow (z-score) -> wspolczynniki sa porownywalne."""
    return pd.DataFrame(
        StandardScaler().fit_transform(df12[kolumny]),
        columns=kolumny, index=df12.index,
    )


# --- Tabele -----------------------------------------------------------------
def tabela_porownawcza(modele):
    rows = []
    for nazwa, m in modele.items():
        rows.append({
            "Model": nazwa,
            "R2": round(m.rsquared, 3),
            "Adj. R2": round(m.rsquared_adj, 3),
            "AIC": round(m.aic, 2),
            "BIC": round(m.bic, 2),
            "F-stat": round(m.fvalue, 3),
            "F p-val": round(m.f_pvalue, 4),
            "DW": round(durbin_watson(m.resid), 3),
            "JB p-val": round(jarque_bera(m.resid).pvalue, 4),
        })
    return pd.DataFrame(rows).set_index("Model")


def gwiazdki(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def tabela_wspolczynnikow(model):
    rows = [
        {"Zmienna": v, "beta": round(model.params[v], 3),
         "p-value": round(model.pvalues[v], 4), "istotnosc": gwiazdki(model.pvalues[v])}
        for v in model.params.index
    ]
    return pd.DataFrame(rows).set_index("Zmienna")


def statystyki_opisowe(df12):
    etykiety = {
        "Dlug_dlugoterminowy_mln_eur": "Dlug dlugoterminowy (mln EUR)",
        "Amortyzacja_mln": "Amortyzacja graczy (mln EUR)",
        "Place_mln": "Place (mln EUR)",
        "Bilans_TransferowyMln": "Bilans transferowy (mln EUR)",
        "COVID_dummy": "COVID (dummy)",
    }
    rows = []
    for col, label in etykiety.items():
        s = df12[col]
        rows.append({
            "Zmienna": label, "N": int(s.count()),
            "Srednia": round(s.mean(), 2), "Odch. std.": round(s.std(), 2),
            "Min": round(s.min(), 2), "Mediana": round(s.median(), 2),
            "Max": round(s.max(), 2), "Skosnosc": round(s.skew(), 3),
            "Kurtoza": round(s.kurt(), 3),
        })
    return pd.DataFrame(rows).set_index("Zmienna")


def vif_modelu_glownego(df12):
    kolumny = ["Bilans_TransferowyMln", "Place_tys_eur"]
    X = StandardScaler().fit_transform(df12[kolumny])
    return pd.DataFrame({
        "Zmienna": kolumny,
        "VIF": [round(variance_inflation_factor(X, i), 3) for i in range(X.shape[1])],
    }).set_index("Zmienna")


# --- Wykresy ----------------------------------------------------------------
def wykres_reszty(model, df12):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0, color=GRANAT, linestyle="--", linewidth=2)
    ax.scatter(model.fittedvalues, model.resid, color=KARMAZYN, s=70, edgecolors="white", zorder=3)
    for x, y, sezon in zip(model.fittedvalues, model.resid, df12["Sezon"]):
        ax.annotate(sezon, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, color=GRANAT)
    ax.set_xlabel("Wartosci dopasowane (mln EUR)")
    ax.set_ylabel("Reszty (mln EUR)")
    ax.set_title("Reszty vs wartosci dopasowane (model M4)")
    fig.tight_layout()
    fig.savefig(OUT / "wykres_reszty_m4.png", dpi=150)
    plt.close(fig)


def wykres_qq(model):
    fig, ax = plt.subplots(figsize=(7, 5))
    (osq, osr), (slope, intercept, _) = probplot(model.resid, dist="norm")
    ax.scatter(osq, osr, color=GRANAT, s=65, edgecolors=ZLOTY, linewidths=0.8, zorder=3)
    ax.plot(osq, slope * np.array(osq) + intercept, color=KARMAZYN, linewidth=1.8)
    ax.set_xlabel("Kwantyle teoretyczne N(0,1)")
    ax.set_ylabel("Kwantyle empiryczne reszt")
    ax.set_title("Wykres Q-Q reszt (model M4)")
    fig.tight_layout()
    fig.savefig(OUT / "wykres_qq_m4.png", dpi=150)
    plt.close(fig)


def wykres_korelacji(df12):
    kolumny = ["Amortyzacja_graczy_tys_eur", "Bilans_TransferowyMln",
               "Przychody_tys_eur", "Place_tys_eur", "COVID_dummy"]
    corr = df12[kolumny].corr()
    corr.index = corr.columns = ["Amortyzacja", "Bilans transf.", "Przychody", "Place", "COVID"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=FCB_CMAP, vmin=-1, vmax=1,
                linewidths=0.5, annot_kws={"size": 10}, ax=ax)
    ax.set_title("Macierz korelacji zmiennych")
    fig.tight_layout()
    fig.savefig(OUT / "wykres_korelacji.png", dpi=150)
    plt.close(fig)


def _os_mln(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)} mln"))


def wykres_dlug_w_czasie(df12):
    x = range(len(df12))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, df12["Dlug_dlugoterminowy_mln_eur"], marker="o", color=GRANAT,
            linewidth=2, label="Dług długoterminowy")
    ax.axvspan(6.5, 11.5, alpha=0.10, color=KARMAZYN, label="Kryzys zadłużenia 2017–2022")
    ax.axvspan(8.5, 11.5, alpha=0.12, color=ZLOTY, label="Okres COVID")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df12["Sezon"], rotation=45, ha="right")
    ax.set_ylabel("mln EUR")
    ax.set_title("Dług długoterminowy FC Barcelona 2010/11–2021/22")
    _os_mln(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "wykres_dlug_w_czasie.png", dpi=150)
    plt.close(fig)


def wykres_scatter_amortyzacja(df12):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df12["Amortyzacja_mln"], df12["Dlug_dlugoterminowy_mln_eur"],
               color=GRANAT, s=70, zorder=3)
    for _, row in df12.iterrows():
        ax.annotate(row["Sezon"], (row["Amortyzacja_mln"], row["Dlug_dlugoterminowy_mln_eur"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color="gray")
    coef = np.polyfit(df12["Amortyzacja_mln"], df12["Dlug_dlugoterminowy_mln_eur"], 1)
    xl = np.linspace(df12["Amortyzacja_mln"].min(), df12["Amortyzacja_mln"].max(), 100)
    ax.plot(xl, np.polyval(coef, xl), color=KARMAZYN, linewidth=1.8,
            label=f"Regresja: y = {coef[0]:.2f}x + {coef[1]:.1f}")
    ax.set_xlabel("Amortyzacja graczy (mln EUR)")
    ax.set_ylabel("Dług długoterminowy (mln EUR)")
    ax.set_title("Amortyzacja graczy a dług długoterminowy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "wykres_scatter_amortyzacja.png", dpi=150)
    plt.close(fig)


def wykres_amortyzacja_w_czasie(df12):
    x = range(len(df12))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, df12["Amortyzacja_mln"], marker="o", color=KARMAZYN, linewidth=2)
    ax.fill_between(list(x), df12["Amortyzacja_mln"], alpha=0.15, color=KARMAZYN)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df12["Sezon"], rotation=45, ha="right")
    ax.set_ylabel("mln EUR")
    ax.set_title("Amortyzacja graczy FC Barcelona 2010/11–2021/22")
    _os_mln(ax)
    fig.tight_layout()
    fig.savefig(OUT / "wykres_amortyzacja_w_czasie.png", dpi=150)
    plt.close(fig)


def wykres_przychody_place(df):
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, df["Przychody_mln"], marker="o", color=GRANAT, linewidth=2, label="Przychody")
    ax.plot(x, df["Place_mln"], marker="s", color=KARMAZYN, linewidth=2, label="Płace")
    ax.axvline(x=11.5, color="gray", linestyle="--", linewidth=1, label="Dane szacowane →")
    ax.axvspan(8.5, 11.5, alpha=0.10, color=ZLOTY, label="Okres COVID")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Sezon"], rotation=45, ha="right")
    ax.set_ylabel("mln EUR")
    ax.set_title("Przychody a płace FC Barcelona 2010/11–2024/25")
    _os_mln(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "wykres_przychody_place.png", dpi=150)
    plt.close(fig)


# --- Main -------------------------------------------------------------------
def main():
    df = wczytaj_dane()
    df12 = df.iloc[:12].copy()  # probka regresyjna: sezony z dostepnym dlugiem
    y = df12["Dlug_dlugoterminowy_mln_eur"]

    # Modele OLS budowane jawnie - dla kazdego widac jego specyfikacje.
    # Model M1: amortyzacja transferow + COVID
    X1 = standaryzuj(df12, ["Amortyzacja_graczy_tys_eur"])
    X1["COVID_dummy"] = df12["COVID_dummy"].values
    model_m1 = sm.OLS(y, sm.add_constant(X1)).fit()

    # Model M2: amortyzacja + place + COVID
    X2 = standaryzuj(df12, ["Amortyzacja_graczy_tys_eur", "Place_tys_eur"])
    X2["COVID_dummy"] = df12["COVID_dummy"].values
    model_m2 = sm.OLS(y, sm.add_constant(X2)).fit()

    # Model M3: sama amortyzacja (bez COVID)
    X3 = standaryzuj(df12, ["Amortyzacja_graczy_tys_eur"])
    model_m3 = sm.OLS(y, sm.add_constant(X3)).fit()

    # Model M4 (glowny): saldo transferowe + place + COVID
    X4 = standaryzuj(df12, ["Bilans_TransferowyMln", "Place_tys_eur"])
    X4["COVID_dummy"] = df12["COVID_dummy"].values
    model_m4 = sm.OLS(y, sm.add_constant(X4)).fit()

    modele = {
        "M1: Amortyzacja + COVID": model_m1,
        "M2: Amortyzacja + Place + COVID": model_m2,
        "M3: Amortyzacja": model_m3,
        "M4: Bilans + Place + COVID": model_m4,
    }
    model_glowny = model_m4

    porownanie = tabela_porownawcza(modele)
    wspolczynniki = {nazwa: tabela_wspolczynnikow(m) for nazwa, m in modele.items()}
    statystyki = statystyki_opisowe(df12)
    vif = vif_modelu_glownego(df12)

    print("\n=== Statystyki opisowe (n=12) ===")
    print(statystyki.to_string())
    print("\n=== Porownanie modeli ===")
    print(porownanie.to_string())
    print("\n=== Wspolczynniki modelu glownego (M4) ===")
    print(wspolczynniki["M4: Bilans + Place + COVID"].to_string())
    print("\n=== VIF (M4) ===")
    print(vif.to_string())

    # Eksport do Excela: kazda tabela na osobnym arkuszu
    with pd.ExcelWriter(OUT / "wyniki_regresji.xlsx", engine="openpyxl") as writer:
        statystyki.to_excel(writer, sheet_name="Statystyki opisowe")
        porownanie.to_excel(writer, sheet_name="Porownanie modeli")
        vif.to_excel(writer, sheet_name="VIF M4")
        for nazwa, tab in wspolczynniki.items():
            tab.to_excel(writer, sheet_name=nazwa.split(":")[0])

    # Wykresy fabularne (trendy w czasie i zaleznosci)
    wykres_dlug_w_czasie(df12)
    wykres_scatter_amortyzacja(df12)
    wykres_amortyzacja_w_czasie(df12)
    wykres_przychody_place(df)
    # Wykresy diagnostyczne modelu glownego
    wykres_reszty(model_glowny, df12)
    wykres_qq(model_glowny)
    wykres_korelacji(df12)

    print(f"\nZapisano wyniki do: {OUT}")


if __name__ == "__main__":
    main()
