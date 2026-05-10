# implementacia kalibracnych metod a ich porovnanie
import numpy as np
import pandas as pd
import json
import mlflow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy import stats
from betacal import BetaCalibration
import statsmodels.api as sm
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")
N_BINS = 10

# nacitanie modelov a ich nekalibrovanych predikcii, na ktore sa aplikuju kalibracne metodz a porovnaju sa vysledky
def load_predictions(experiment_name, artifact_name):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'" 
    )

    TARGET_RUNS = {"LogReg_FINAL_MODEL", "XGBoost_FINAL_MODEL"}
    probs, y_true = {}, None
    for run in runs:
        model = run.data.params.get("model")
        if model is None:
            continue
        run_name = run.data.tags.get("mlflow.runName")
        if run_name not in TARGET_RUNS:
            continue
        path = client.download_artifacts(run.info.run_id, artifact_name)
        with open(path) as f:
            data = json.load(f)
        probs[model] = np.array(data["y_prob"])
        y_true = np.array(data["y_true"])
    return y_true, probs

# kalibracna regresia s evaluaciou interceptu a slopeu a ich p-hodnotami a intervalmi spolahlivosti
def calibration_regression(y_true, y_prob):
    eps = 1e-10
    logit_pred = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))
    X = sm.add_constant(logit_pred)
    model = sm.Logit(y_true, X).fit(disp=0)
    intercept = model.params[0]
    slope     = model.params[1]
    # H0: intercept == 0
    z_intercept = intercept / model.bse[0]
    p_intercept = 2 * (1 - stats.norm.cdf(abs(z_intercept)))
    # H0: slope == 1
    z_slope = (slope - 1) / model.bse[1]
    p_slope = 2 * (1 - stats.norm.cdf(abs(z_slope)))
    # 95% confidence interval
    ci = model.conf_int(alpha=0.05)
    ci_intercept = (ci[0][0], ci[0][1])
    ci_slope     = (ci[1][0], ci[1][1])
    return {
        "intercept":    intercept,
        "slope":        slope,
        "p_intercept":  p_intercept,
        "p_slope":      p_slope,
        "ci_intercept": ci_intercept,
        "ci_slope":     ci_slope,
    }

# vypocet bin based metrik
def compute_ece_mce(y_true, y_prob, strategy):
    if strategy == "uniform":       #rovnako siroke intervaly
        bins = np.linspace(0.0, 1.0, N_BINS + 1)
    else:   # quantile - kazdy interval ma rovnaky pocet vzoriek, ale inu sirku
        bins = np.percentile(y_prob, np.linspace(0, 100, N_BINS + 1))
        bins = np.unique(bins)
 
    bin_indices = np.digitize(y_prob, bins[1:-1])  
    ece = 0.0
    mce = 0.0
    n = len(y_true)
 
    for b in range(len(bins) - 1):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()  # priemerny empiric vyskyt
        bin_conf = y_prob[mask].mean()   #priemerna pravdepodobnost       
        bin_n    = mask.sum()
        gap = abs(bin_acc - bin_conf)
        ece += (bin_n / n) * gap
        mce = max(mce, gap)
 
    return ece, mce

# vypocet vsetkych metrik kalibracie
def compute_all_metrics(y_true, y_prob):
    calibration   = calibration_regression(y_true, y_prob)
    intercept    = calibration["intercept"]
    slope        = calibration["slope"]
    ece, mce = compute_ece_mce(y_true, y_prob, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)
    return {"Intercept": intercept, "Slope": slope, "ECE": ece, "MCE": mce, "Brier": brier}

# pomocna funkcia pre logit transformaciu pravdepodobnosti
def _logit(p, eps=1e-10):
    return np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))

# platt scaling
class PlattScaling:
    def __init__(self):
        self.model = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 1e9],
            cv=5,
            solver="lbfgs",
            scoring="neg_log_loss",
            max_iter=1000,
        )

    def fit(self, y_true_calib, y_prob_calib):
        X = _logit(y_prob_calib).reshape(-1, 1)
        self.model.fit(X, y_true_calib)
        return self

    def predict(self, y_prob):
        X = _logit(y_prob).reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]

# isotonicka regresia
class IsotonicCalibration:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_true_calib, y_prob_calib):
        self.model.fit(y_prob_calib, y_true_calib)
        return self

    def predict(self, y_prob):
        return self.model.predict(y_prob)

# teplotne skalovanie
class TemperatureScaling:
    def __init__(self):
        self.T = 1.0

    def fit(self, y_true_calib, y_prob_calib):
        logit = _logit(y_prob_calib)
        eps = 1e-10

        def min_T(T):
            p = expit(logit / T)
            p = np.clip(p, eps, 1 - eps)
            return -np.mean(y_true_calib * np.log(p) + (1 - y_true_calib) * np.log(1 - p))

        result  = minimize_scalar(min_T, bounds=(0.01, 10.0), method="bounded")
        self.T  = result.x
        return self

    def predict(self, y_prob):
        return expit(_logit(y_prob) / self.T)

# beta kalibracia
class BetaCalib:
    def __init__(self):
        self.model = BetaCalibration(parameters="abm")

    def fit(self, y_true_calib, y_prob_calib):
        self.model.fit(y_prob_calib.reshape(-1, 1), y_true_calib)
        return self

    def predict(self, y_prob):
        return self.model.predict(y_prob.reshape(-1, 1))

# Spline-based kalibracia
class SplineCalibration:
    def __init__(self, n_knots=5, degree=3):
        self.pipeline = Pipeline([
            ("spline", SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)),
            ("logit",  LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000))
        ])

    def fit(self, y_true_calib, y_prob_calib):
        X = y_prob_calib.reshape(-1, 1)
        self.pipeline.fit(X, y_true_calib)
        return self

    def predict(self, y_prob):
        X = y_prob.reshape(-1, 1)
        return self.pipeline.predict_proba(X)[:, 1]


CALIBRATION_METHODS = {
    "Platt scaling":       PlattScaling,
    "Isotonic regression": IsotonicCalibration,
    "Temperature scaling": TemperatureScaling,
    "Spline kalibrácia":   SplineCalibration,
    "Beta kalibrácia":     BetaCalib,
}

# vykonanie kalibracie a vyhodnotenie metrik pred a po kalibracii
def do_calibration_and_evaluate(model_name, y_true_calib, y_prob_calib, y_true_test,  y_prob_test):
    print(f"\n{'='*65}")
    print(f"  Kalibrácia modelu: {model_name}")
    results = {}

    # Hodnoty pred kalibraciou – baseline
    results["pred kalibráciou"] = {
        "calib": compute_all_metrics(y_true_calib, y_prob_calib),
        "test":  compute_all_metrics(y_true_test,  y_prob_test),
        "y_prob_test": y_prob_test,
    }
    # aplikacia kalibracie
    for method_name, MethodClass in CALIBRATION_METHODS.items():
        try:
            calibrator = MethodClass()
            calibrator.fit(y_true_calib, y_prob_calib)
            y_cal_calib = calibrator.predict(y_prob_calib)
            y_cal_test  = calibrator.predict(y_prob_test)
            results[method_name] = {
                "calib":      compute_all_metrics(y_true_calib, y_cal_calib),
                "test":       compute_all_metrics(y_true_test,  y_cal_test),
                "y_prob_test": y_cal_test,
                "calibrator": calibrator,
            }
        except Exception as e:
            print(f"Chyba pri {method_name}: {e}")

    return results

# kontrola, ci kalibracia nie je pretrenovana na validacnych datach - vypis ECE na validacnej a testovacej mnozine a ich rozdiel
def overfit_check(results):
    print(f"\n{'='*65}")
    print(f"  OVERENIE OVERFITTINGU KALIBRÁCIE")
    print(f"  {'Metóda':<25} {'ECE calib':>10} {'ECE test':>10} {'Rozdiel':>10}")
    print(f"  {'-'*57}")
    for method, vals in results.items():
        ece_calib = vals["calib"]["ECE"]
        ece_test  = vals["test"]["ECE"]
        diff      = ece_test - ece_calib
        print(f"  {method:<25} {ece_calib:>10.4f} {ece_test:>10.4f} {diff:>10.4f}")

# vypis metrik kalibracie
def summary_metriky(results):
    vysledky = []
    for method, values in results.items():
        t = values["test"]
        vysledky.append({
            "METODA":       method,
            "Intercept":    round(t["Intercept"], 4),
            #"p (int=0)":    round(t["p_intercept"], 4),
            "Slope":        round(t["Slope"], 4),
            #"p (slope=1)":  round(t["p_slope"], 4),
            "ECE":          round(t["ECE"], 4),
            "MCE":          round(t["MCE"], 4),
            "Brier":        round(t["Brier"], 4),
        })
    df = pd.DataFrame(vysledky).set_index("METODA")
    print(f"\n{'='*65}")
    print("Kalibracna kvalita modelov – TEST set")
    print(f"{'='*65}")
    print(df.to_string())
    return df

# vizualizacia kalibracnych kriviek a tabulka s metrikami pre vsetky kalibracne metody
def plot_calibration(all_results):
    model_names = list(all_results.keys())
    n_models    = len(model_names)

    fig = plt.figure(figsize=(7 * n_models, 10))
    gs  = gridspec.GridSpec(2, n_models, height_ratios=[2, 1], hspace=0.15)

    fig.suptitle("Kalibračná výkonnosť pred a po kalibrácii", fontsize=14, fontweight="bold", y=1.01)

    for col, model_name in enumerate(model_names):
        results   = all_results[model_name]
        n_methods = len(results)
        colors    = plt.cm.tab10(np.linspace(0, 1, n_methods))

        # kalibracne krivky
        ax = fig.add_subplot(gs[0, col])
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Ideálna kalibrácia", zorder=5)

        for (method, vals), color in zip(results.items(), colors):
            y_prob = vals["y_prob_test"]
            y_true = vals.get("y_true_test")
            if y_true is None:
                continue

            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=N_BINS, strategy="uniform")
            ls  = "--" if method == "pred kalibráciou" else "-"
            lw  = 2.5  if method == "pred kalibráciou" else 1.8
            ax.plot(prob_pred, prob_true, linestyle=ls, color=color, lw=lw, marker="o", markersize=4, label=f"{method}", zorder=3)

        ax.set_title(model_name.upper(), fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Predikovaná pravdepodobnosť", fontsize=9)
        ax.set_ylabel("Empirický výskyt udalosti", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # tabulka s metrikami
        ax_tbl = fig.add_subplot(gs[1, col])
        ax_tbl.axis("off")
        col_labels = ["", "Intercept", "Slope", "ECE", "MCE", "Brier"]
        metric_keys = ["Intercept", "Slope", "ECE", "MCE", "Brier"]

        rows = []
        for method, vals in results.items():
            t = vals["test"]
            rows.append([
                method,
                f"{t['Intercept']:+.4f}",
                f"{t['Slope']:.4f}",
                f"{t['ECE']:.4f}",
                f"{t['MCE']:.4f}",
                f"{t['Brier']:.4f}",
            ])

        # top hodnota pre metriku
        best_row_per_col = {}
        method_list = list(results.keys())
        for col_idx, key in enumerate(metric_keys, start=1):
            values = [results[m]["test"][key] for m in method_list]
            if key == "Intercept":
                best = min(range(len(values)), key=lambda i: abs(values[i]))
            elif key == "Slope":
                best = min(range(len(values)), key=lambda i: abs(values[i] - 1))
            else:
                best = min(range(len(values)), key=lambda i: values[i])
            best_row_per_col[col_idx] = best + 1   # +1 kvôli hlavičke

        # matica farieb
        row_colors = []
        for row_idx, method in enumerate(method_list):
            row_color = []
            for col_idx in range(len(col_labels)):
                if col_idx == 0:
                    # stĺpec s názvom metódy
                    c = "#cbc9c9" if method == "pred kalibráciou" else "#ffffff"
                elif best_row_per_col.get(col_idx) == row_idx + 1:
                    c = "#abd3a7"   # zelená = najlepšia hodnota v stĺpci
                elif method == "pred kalibráciou":
                    c = "#cbc9c9"   # sivá pre baseline
                else:
                    c = "#ffffff"
                row_color.append(c)
            row_colors.append(row_color)

        tbl = ax_tbl.table(
            cellText    = rows,
            colLabels   = col_labels,
            cellLoc     = "center",
            loc         = "center",
            cellColours = row_colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)

        for row_idx in range(len(rows) + 1):
            tbl[row_idx, 0].set_width(0.3)

        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#3c444d")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

    plt.savefig("porovnanie_calib_metod_5H_uniform.png", dpi=150, bbox_inches="tight")
    plt.show()
    return fig

# spustenie kalibracie na porovnanie metod
def run_calibration(experiment_name, models_to_calibrate=("xgboost", "lr")):
    print("Načítavam predikcie z MLflow...")
    y_true_calib, probs_calib = load_predictions(experiment_name, artifact_name="predictions_validation.json")
    y_true_test, probs_test = load_predictions(experiment_name, artifact_name="predictions.json")

    all_results = {}
    for model in models_to_calibrate:
        if model not in probs_calib:
            print(f"Model '{model}' nebol nájdený v MLflow.")
            continue

        results = do_calibration_and_evaluate(model, y_true_calib, probs_calib[model], y_true_test,  probs_test[model])

        for v in results.values():
            v["y_true_test"] = y_true_test

        overfit_check(results)
        df = summary_metriky(results)
        all_results[model] = {"results": results, "summary": df}

    plot_calibration(all_results={name: data["results"] for name, data in all_results.items()})
    return all_results


if __name__ == "__main__":
    EXPERIMENT = "BP:NACC - finalne hodnotenie + kalibracia"
    all_results = run_calibration(EXPERIMENT, ("xgboost", "lr"))