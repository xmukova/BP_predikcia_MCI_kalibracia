# interpretacia logistickej regresie - zobrazenie koeficientov premennych a ich vplyvu na predikciu
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
matplotlib.use("Agg")

from config_based_on_EDA import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS
pipeline = joblib.load("models/lr_pipeline.pkl")

# ziskanie nazvov premennych z pipeline
all = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
feature_selector = pipeline.named_steps["feature_selection"]
if hasattr(feature_selector, "selected_features_"): # CustomFeatureSelector
    selected_indices = feature_selector.selected_features_
    selected_names = [all[i] for i in selected_indices]
elif hasattr(feature_selector, "get_support"):
    mask = feature_selector.get_support()
    selected_names = [all[i] for i, v in enumerate(mask) if v]
else:
    selected_names = all

# ziskanie koeficientov z modelu
lr_model = pipeline.named_steps["model"]
koeficienty = lr_model.coef_[0]

df_koef = pd.DataFrame({
    "Premenná":      selected_names,
    "Koeficient":    koeficienty,
    "Odds Ratio":    np.exp(koeficienty),      # e^koeficient
    "Abs. hodnota":  np.abs(koeficienty),
}).sort_values("Abs. hodnota", ascending=False)
df_koef["Smer"] = df_koef["Koeficient"].apply(lambda x: "zvyšuje riziko" if x > 0 else "znižuje riziko")

# vypis premennych spolu s koeficientami, OR a smerom vplyvu
TOP_N = 30
df_top = df_koef.head(TOP_N)
print(f"\n{'='*75}")
print(f"  TOP {TOP_N} NAJDÔLEŽITEJŠÍCH PREMENNÝCH")
print(f"{'='*75}")
for _, row in df_top.iterrows():
    print(f"  {row['Premenná']:<35} "
          f"koef={row['Koeficient']:+.4f}  "
          f"OR={row['Odds Ratio']:.3f}  "
          f"{row['Smer']}")

# vizualizacia 
fig, ax = plt.subplots(figsize=(10, 8))
colors_or = ["#d73027" if v > 1 else "#4575b4" for v in df_top["Odds Ratio"]]
colors_or = ["#d73027" if v > 1 else "#4575b4" if v < 1 else "gray" for v in df_top["Odds Ratio"]]
ax.barh(
    df_top["Premenná"][::-1],
    df_top["Odds Ratio"][::-1],
    color=colors_or[::-1],
    edgecolor="white",
    linewidth=0.5
)
ax.axvline(x=1, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Odds Ratio (OR = e^koeficient)", fontsize=11)
ax.set_title(f"Top {TOP_N} premenných — Odds Ratios v logistickej regresii", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

legend_elements = [
    Patch(facecolor="#d73027", label="Zvyšuje riziko konverzie"),
    Patch(facecolor="#4575b4", label="Znižuje riziko konverzie"),
    Patch(facecolor="gray", label="Neovplyvňuje riziko konverzie")]

fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
# plt.savefig("lr_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()