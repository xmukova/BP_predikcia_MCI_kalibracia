# vyhodnotenie kalibracnej kvality modelov pred kalibraciou
import numpy as np
import pandas as pd
import mlflow
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss
from scipy import stats
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")
N_BINS = 10
EXPERIMENT_NAME = "NACC - finalne modely s kalibráciou"

# nacitanie nekalibrovanych predikcii z mlflow pre kazdy model
def load_predictions(experiment_name):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'")
    TARGET_RUNS = {"LogReg_FINAL_calibrated_MODEL", "XGBoost_FINAL_calibrated_MODEL", "SVM_FINAL_calibrated_MODEL"} 
    model_probability = {}
    y_true = None
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "")
        if run_name not in TARGET_RUNS:
            continue
        model_name = run.data.params.get("model")
        if model_name is None: continue

        artifact_path = client.download_artifacts(run.info.run_id, "predictions.json") # = nekalibrovanie predikcie
        with open(artifact_path, "r") as f:
            data = json.load(f)
        
        model_probability[model_name] = np.array(data["y_prob"])
        y_true = np.array(data["y_true"])
    
    return y_true, model_probability

# vypocet metrik kalibracnej regresie a testovanie H0 pre intercept a slope
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

# vypocet bin-based metrik - ECE a MCE, a detailne statistiky pre kazdy bin (interval) 
def compute_ece_mce(y_true, y_prob, strategy="quantile"):
    if strategy == "uniform":       #rovnako siroke intervaly
        bins = np.linspace(0.0, 1.0, N_BINS + 1)
    else:   # quantile - kazdy interval ma rovnaky pocet vzoriek, ale inu sirku
        bins = np.percentile(y_prob, np.linspace(0, 100, N_BINS + 1))
        bins = np.unique(bins)
 
    bin_indices = np.digitize(y_prob, bins[1:-1])  
    ece = 0.0
    mce = 0.0
    n = len(y_true)
    bin_stats = []
 
    for b in range(len(bins) - 1):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()          
        bin_n    = mask.sum()
        gap = abs(bin_acc - bin_conf)
        ece += (bin_n / n) * gap
        mce = max(mce, gap)
        bin_stats.append({
            "bin_lower": bins[b],
            "bin_upper": bins[b + 1],
            "pocet_pacientov": bin_n,
            "mean_prediction": bin_conf,
            "realita_ochorenia": bin_acc,
            "rozdiel": gap
        })
 
    return ece, mce, pd.DataFrame(bin_stats)
 
# ziskanie vypocitanych metrik a kalibracnej krivky 
def evaluate_calibration(model_name, y_true, y_prob, strategy, ax=None):
    calibration = calibration_regression(y_true, y_prob)  
    intercept    = calibration["intercept"]
    slope        = calibration["slope"]
    p_intercept  = calibration["p_intercept"]
    p_slope      = calibration["p_slope"]
    ci_intercept = calibration["ci_intercept"]
    ci_slope     = calibration["ci_slope"]
    ece, mce, bin_df = compute_ece_mce(y_true, y_prob, strategy=strategy)
    brier = brier_score_loss(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=N_BINS, strategy=strategy)

    if ax is not None:
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Ideálna kalibrácia")
        ax.plot(prob_pred, prob_true, "o-", lw=2, color="steelblue", markersize=5, label=model_name)

        if strategy == "uniform":       #rovnako siroke intervaly
            bins = np.linspace(0.0, 1.0, N_BINS + 1)
            bin_counts = []
            bin_indices = np.digitize(y_prob, bins[1:-1])
            for b in range(len(bins) - 1):
                mask = bin_indices == b
                if mask.sum() > 0:
                    bin_counts.append(mask.sum())

            for x, y, count in zip(prob_pred, prob_true, bin_counts):
                ax.annotate(str(count), xy=(x, y), ha="center", va="center", fontsize=6,
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="steelblue", linewidth=1))

        ax2 = ax.twinx()
        ax2.hist(y_prob, bins=N_BINS, range=(0, 1), alpha=0.15, color="steelblue")
        ax2.set_ylabel("Počet vzoriek", fontsize=8, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=7)

        ax.set_xlabel("Predikovaná pravdepodobnosť", fontsize=9)
        ax.set_ylabel("Empirický výskyt udalosti", fontsize=9)
        ax.set_title(
            f"{model_name}\n"
            f"Intercept={intercept:.3f}     Slope={slope:.3f}\n"
            f"ECE={ece:.4f}     MCE={mce:.4f}\n"
            f"Brier score={brier:.4f}",
            fontsize=9
        )
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    return {
        "model":        model_name,
        "intercept":    intercept,
        "slope":        slope,
        "p_intercept":  p_intercept,
        "p_slope":      p_slope,
        "ci_intercept": ci_intercept,
        "ci_slope":     ci_slope,
        "ECE":          ece,
        "MCE":          mce,
        "Brier":        brier,
        "bin_df":       bin_df
    }
 
 
def run_calibration_analysis(models_probabilities, y_true):
    n_models = len(models_probabilities)
    fig = plt.figure(figsize=(6 * n_models, 5))
    gs  = gridspec.GridSpec(1, n_models, figure=fig)
    fig.suptitle("Kalibračná analýza výkonnosti modelov ", fontsize=13, fontweight="bold", y=1.02)
 
    results = []
    for i, (name, y_prob) in enumerate(models_probabilities.items()):
        ax = fig.add_subplot(gs[0, i])
        res = evaluate_calibration(name, y_true, y_prob, strategy="uniform",ax=ax)
        results.append(res)
 
    plt.tight_layout()
    plt.savefig("kalibracna_kvalita_PRED_H5.png", dpi=150, bbox_inches="tight")
    plt.show()
 
    summary = pd.DataFrame([{
        "model":       r["model"],
        "Intercept":   r["intercept"],
        "p (int=0)":   r["p_intercept"],
        "CI intercept": f"({r['ci_intercept'][0]:.3f}, {r['ci_intercept'][1]:.3f})",
        "Slope":       r["slope"],
        "p (slope=1)": r["p_slope"],
        "CI slope":    f"({r['ci_slope'][0]:.3f}, {r['ci_slope'][1]:.3f})",
        "ECE":         r["ECE"],
        "MCE":         r["MCE"],
        "Brier":       r["Brier"],
        }for r in results
    ]).set_index("model")

    print("\n===== Kalibracna kvalita modelov =====")
    print(summary.round(4).to_string())
    return summary, results
 
 
if __name__ == "__main__":
    y_true, models_probabilities = load_predictions(EXPERIMENT_NAME)
    summary, detail = run_calibration_analysis(models_probabilities, y_true)

    for i in detail:
        print(f"\n----- Detailné bin štatistiky: {i['model']} -----")
        print(i["bin_df"].round(4).to_string(index=False))