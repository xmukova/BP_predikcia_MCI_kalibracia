# finalny model s kalibraciou a evaluacia a zaznamenanie modelu aj metrik
import os
import json
import hashlib
import joblib
from matplotlib import gridspec
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import ( brier_score_loss, RocCurveDisplay, PrecisionRecallDisplay,)

from pipeline import create_pipeline_complete
from train_model import load_data, split_data_60_20_20, evaluate_model
from vizualizacia_calibration import compute_ece_mce, calibration_regression
from config_experiments import FINAL_MODELS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")
N_BINS = 10

# pomocna funkcia pre logit transformaciu pravdepodobnosti pre Platt scaling
def _logit(p, eps=1e-10):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

# kalibracna evaluacia - kalibracna regresia, ECE, MCE, Brier score
def evaluate_calibration(y_true, y_prob, label=""):
    cal   = calibration_regression(y_true, y_prob)
    ece, mce, bin_df = compute_ece_mce(y_true, y_prob, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)

    print(f"\n  {'─'*48}")
    print(f"  Kalibračná evaluácia  {label}")
    print(f"  {'─'*48}")
    print(f"  Intercept: {cal['intercept']:+.4f}  " f"(p={cal['p_intercept']:.4f}, "
          f"CI: {cal['ci_intercept'][0]:.3f}–{cal['ci_intercept'][1]:.3f})")
    print(f"  Slope:     {cal['slope']:+.4f}  " f"(p={cal['p_slope']:.4f}, "
          f"CI: {cal['ci_slope'][0]:.3f}–{cal['ci_slope'][1]:.3f})")
    print(f"  ECE:       {ece:.4f}")
    print(f"  MCE:       {mce:.4f}")
    print(f"  Brier:     {brier:.4f}")

    return {**cal, "ECE": ece, "MCE": mce, "Brier": brier, "bin_df": bin_df}

# implementacia Platt scaling kalibracie
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
        print(f"    Platt – A={self.model.coef_[0][0]:.4f}  "f"B={self.model.intercept_[0]:.4f}  "f"C={self.model.C_[0]:.4f}")
        return self

    def predict_proba(self, y_prob):
        X = _logit(y_prob).reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, y_prob, threshold=0.5):
        return (self.predict_proba(y_prob) >= threshold).astype(int)
      
# vizualizacia diskriminacie ROC a PR krivka pred a po kalibracii
def make_discrimination_plots(y_true, y_proba_before, y_proba_after, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Diskriminačná výkonnosť – {model_name}", fontsize=13, fontweight="bold")
    for proba, label, ls in [(y_proba_before, "pred kalibráciou", "--"), (y_proba_after, "po kalibrácii", "-")]:
        RocCurveDisplay.from_predictions(y_true, proba, ax=axes[0], name=label, linestyle=ls)
        PrecisionRecallDisplay.from_predictions(y_true, proba, ax=axes[1], name=label, linestyle=ls)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("ROC krivka")
    axes[1].set_title("Precision-Recall krivka")
    plt.tight_layout()
    return fig

# vizualizacia kalibracie - krivka a metriky - porovnanie pred a po kalibracii
def make_calibration_plot(y_true, y_proba_before, y_proba_after, cal_before, cal_after, model_name):
    fig = plt.figure(figsize=(8, 9))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[0.5, 4], hspace=0.05, figure=fig)
    fig.suptitle(f"Kalibrácia modelu – {model_name}", fontsize=13, fontweight="bold", y=0.99)

    # tabuľka metriky
    ax_tbl = fig.add_subplot(gs[0, 0])
    ax_tbl.axis("off")
    col_labels = ["", "Intercept", "Slope", "ECE", "MCE", "Brier"]
    metrics = [("intercept", "{:+.4f}"), ("slope", "{:.4f}"), ("ECE", "{:.4f}"), ("MCE", "{:.4f}"), ("Brier", "{:.4f}")]

    row_pred = ["pred kalibráciou"]
    row_po = ["po kalibrácii"]      
    colors_po = []

    for key, fmt in metrics:
        row_pred.append(fmt.format(cal_before[key]))
        row_po.append(fmt.format(cal_after[key]))
        # farba bunky PO – zelena ak zlepsenie, cervena ak zhorsenie
        if key in ("ECE", "MCE", "Brier"):
            improved = cal_after[key] < cal_before[key]
        elif key == "intercept":
            improved = abs(cal_after[key]) < abs(cal_before[key])
        else:  # slope
            improved = abs(cal_after[key] - 1) < abs(cal_before[key] - 1)
        colors_po.append("#c8f0c8" if improved else "#f5c6c6")

    colors_pred = ["#ffffff"] * 6
    colors_po   = ["#ffffff"] + colors_po   
    cell_colors = [colors_pred, colors_po]
    rows        = [row_pred, row_po]

    tbl = ax_tbl.table(
        cellText    = rows,
        colLabels   = col_labels,
        cellLoc     = "center",
        loc         = "upper center",
        cellColours = cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#BAC2C9")
        tbl[0, j].set_text_props(color="black", fontweight="bold")

    tbl[1, 0].set_text_props(fontweight="bold", ha="left")
    tbl[2, 0].set_text_props(fontweight="bold", ha="left")
    tbl[1, 0].get_text().set_wrap(True)
    tbl[2, 0].get_text().set_wrap(True)
    # kalibracna krivka
    ax = fig.add_subplot(gs[1, 0])
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Ideálna kalibrácia")

    for y_prob, label, ls, color in [
        (y_proba_before, "pred kalibráciou", "--", "steelblue"),
        (y_proba_after,  "po kalibrácii",    "-",  "darkorange")]:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=N_BINS, strategy="uniform")
        bins = np.linspace(0.0, 1.0, N_BINS + 1)
        bin_counts = []
        bin_indices = np.digitize(y_prob, bins[1:-1])
        for b in range(len(bins) - 1):
            mask = bin_indices == b
            if mask.sum() > 0:
                bin_counts.append(mask.sum())

        ax.plot(prob_pred, prob_true, ls, color=color, lw=2, label=label)
        for x, y, cnt in zip(prob_pred, prob_true, bin_counts):
            ax.annotate(str(cnt), xy=(x, y), ha="center", va="center", fontsize=6,
                bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor=color, lw=1.0))  

    ax.set_xlabel("Predikovaná pravdepodobnosť")
    ax.set_ylabel("Empirický výskyt udalosti")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# finalny model aj s kalibraciou
class FinalModel:
    def __init__(self, pipeline, calibrator=None, treshold=0.5):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.treshold = treshold

    def predict_proba(self, X):
        p = self.pipeline.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(p)
        return p

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

# finalne natrenovanie a kalibracia a evaluacia a ulozenie modelu a metrik do mlflow
def run_final_experiment(config, X_train, X_calib, X_test, Y_train, Y_calib, Y_test):
    model_name   = config["model_name"]
    experiment_nazov = config["nazov"]
    model_params = config["model_parameters"].copy()
    if model_params.get("scale_pos_weight") == "auto":
        model_params["scale_pos_weight"] = (Y_train.value_counts()[0] / Y_train.value_counts()[1])

    print(f"\n{'='*65}")
    print(f"  MODEL: {experiment_nazov}")
    print(f"{'='*65}")

    with mlflow.start_run(run_name=experiment_nazov) as run:
        mlflow.log_param("model",             model_name)
        mlflow.log_param("feature_selection", config["feature_selection"])
        mlflow.log_param("k_features",        config["k_features"])
        for k, v in model_params.items():
            mlflow.log_param(k, v)
        mlflow.set_tag("version",     config.get("version", "v1"))
        mlflow.set_tag("config_hash", hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest())

        # zakladny trening
        pipeline = create_pipeline_complete(
            model_name        = model_name,
            feature_selection = config["feature_selection"],
            k_features        = config["k_features"],
            model_parameters  = model_params,
        )
        pipeline.fit(X_train, Y_train)

        # predikcia modelu pred kalibraciou
        y_pred_test  = pipeline.predict(X_test)
        y_proba_test = pipeline.predict_proba(X_test)[:, 1]
        y_pred_calib  = pipeline.predict(X_calib)
        y_proba_calib = pipeline.predict_proba(X_calib)[:, 1]

        # vyhodnotenie modelu PRED kalibraciou
        print("\n  ------ PRED KALIBRÁCIOU -------")
        discrimination_before = evaluate_model(Y_test, y_pred_test, y_proba_test, f"{model_name}[pred kalibráciou]" )
        calibration_before  = evaluate_calibration(Y_test, y_proba_test, f"{model_name}[pred kalibráciou]" )

        for k, v in discrimination_before.items():
            if isinstance(v, float):
                mlflow.log_metric(f"pred_cal_{k}", round(v, 4))
        for k in ["ECE", "MCE", "Brier", "intercept", "slope"]:
            mlflow.log_metric(f"pred_cal_{k}", round(calibration_before[k], 4))

        mlflow.log_dict({"y_true": Y_test.tolist(), "y_prob": y_proba_test.tolist()}, "predictions.json")
        mlflow.log_dict({"y_true": Y_calib.tolist(), "y_prob": y_proba_calib.tolist()}, "predictions_validation.json")

        # KALIBRACIA – Platt scaling
        if model_name == 'svm': # na svm neaplikujem kalibraciu
            y_proba_calibrovane_test = y_proba_test      
            y_pred_calibrovanie_test  = y_pred_test
            calibration_after    = calibration_before
            discrimination_after = discrimination_before
            calibrator = None
        else:
            print("\n  ------ PLATT SCALING ------")
            platt = PlattScaling()
            platt.fit(Y_calib.values, y_proba_calib)

            # predikcie po kalibracii
            y_proba_calibrovane_test  = platt.predict_proba(y_proba_test)
            y_pred_calibrovanie_test   = platt.predict(y_proba_test)

            # vyhodnotenie modelu PO kalibracii
            print("\n  ── PO KALIBRÁCII ──")
            discrimination_after = evaluate_model(Y_test, y_pred_calibrovanie_test, y_proba_calibrovane_test, f"{model_name}[po kalibrácii]")
            calibration_after  = evaluate_calibration(Y_test, y_proba_calibrovane_test, f"{model_name}[po kalibrácii]")
            calibrator = platt


        for k, v in discrimination_after.items():
            if isinstance(v, float):
                mlflow.log_metric(f"po_cal_{k}", round(v, 4))
        for k in ["ECE", "MCE", "Brier", "intercept", "slope"]:
            mlflow.log_metric(f"po_cal_{k}", round(calibration_after[k], 4))
        
        mlflow.log_dict({"y_true": Y_test.tolist(), "y_prob": y_proba_calibrovane_test.tolist()}, "predictions_calibrated.json")

        fig_disc = make_discrimination_plots(Y_test, y_proba_test, y_proba_calibrovane_test, experiment_nazov)
        fig_cal  = make_calibration_plot(Y_test, y_proba_test, y_proba_calibrovane_test, calibration_before, calibration_after, experiment_nazov)
        
        mlflow.log_figure(fig_disc, f"FINAL_diskriminacia_{model_name}.png")
        mlflow.log_figure(fig_cal,  f"FINAL_kalibracia_{model_name}.png")
        
        fig_disc.savefig(f"FINAL_diskriminacia_{model_name}.png", dpi=150, bbox_inches="tight")
        fig_cal.savefig(f"FINAL_kalibracia_{model_name}_5H_uniform.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        os.makedirs("models", exist_ok=True)
        # ulozenie modelov bez kalibracie
        mlflow.sklearn.log_model(pipeline, name=f"{model_name}_pipeline")
        pipeline_path = os.path.join("models", f"{model_name}_pipeline.pkl")
        joblib.dump(pipeline, pipeline_path)

        # ulozenie kalibratoru
        if calibrator is not None:
            calibrator_path = os.path.join("models", f"{model_name}_calibrator.pkl")
            joblib.dump(calibrator, calibrator_path)
            mlflow.log_artifact(calibrator_path, artifact_path="calibrators")

        # ulozenie finalny model
        final_model = FinalModel(pipeline, calibrator)
        final_model_path = os.path.join("models", f"{model_name}_FINAL_MODEL.pkl")
        joblib.dump(final_model, final_model_path)
        mlflow.log_artifact(final_model_path, artifact_path="final_models")
        print(f"\nModel uložený: {final_model_path}")

        return {
            "model": experiment_nazov,
            "before_cal": {**discrimination_before, **calibration_before},
            "after_cal": {**discrimination_after,  **calibration_after},
            "pipeline": pipeline,
            "calibrator": calibrator,
        }

def train_final():
    EXPERIMENT_NAME = "NACC - finalne modely s kalibráciou"
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, Y = load_data()
    X_train, X_calib, X_test, Y_train, Y_calib, Y_test = split_data_60_20_20(X, Y)

    all_results = {}
    for config in FINAL_MODELS:
        result = run_final_experiment(config, X_train, X_calib, X_test, Y_train, Y_calib, Y_test)
        all_results[result["model"]] = result

    rows = []
    for model, res in all_results.items():
        rows.append({
            "Model":             model,
            "ROC-AUC pred":      round(res["before_cal"]["roc_auc"], 4),
            "ROC-AUC po":        round(res["after_cal"]["roc_auc"], 4),
            "ECE pred":          round(res["before_cal"]["ECE"], 4),
            "ECE po":            round(res["after_cal"]["ECE"], 4),
            "Brier pred":        round(res["before_cal"]["Brier"], 4),
            "Brier po":          round(res["after_cal"]["Brier"], 4),
            "Intercept pred":    round(res["before_cal"]["intercept"], 4),
            "Intercept po":      round(res["after_cal"]["intercept"], 4),
            "Slope pred":        round(res["before_cal"]["slope"], 4),
            "Slope po":          round(res["after_cal"]["slope"], 4),
        })

    summary = pd.DataFrame(rows).set_index("Model")
    print(f"\n{'='*65}")
    print("  SÚHRNNÉ VÝSLEDKY – PRED vs PO kalibrácii")
    print(f"{'='*65}")
    print(summary.to_string())

    return all_results, summary

if __name__ == "__main__":
    all_results, summary = train_final()