# VIZUALIZACIA A ANALÝZA THRESHOLDOV
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")

# vypocet jednotlivych metrik pre rozne thresholdy
def threshold_metrics(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.02)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        rows.append({
            "threshold":   round(t, 2),
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "precision":   precision_score(y_true, y_pred, zero_division=0),
            "f1":          f1_score(y_true, y_pred, zero_division=0),
            "accuracy":   (tp + tn) / len(y_true) if len(y_true) > 0 else 0,
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
        })
    return pd.DataFrame(rows)

# nacitanie nekalibrovanych predikcii z mlflow
def load_predictions_from_mlflow(run_id):
    client = mlflow.tracking.MlflowClient()
    path = client.download_artifacts(run_id, "predictions.json")
    import json
    with open(path) as f:
        data = json.load(f)
    return np.array(data["y_true"]), np.array(data["y_prob"])

# nacitanie kalibrovanych predikcii z mlflow
def load_calibrated_predictions_from_mlflow(run_id):
    client = mlflow.tracking.MlflowClient()
    path = client.download_artifacts(run_id, "predictions_calibrated.json")
    import json
    with open(path) as f:
        data = json.load(f)
    return np.array(data["y_true"]), np.array(data["y_prob"])

# vizualizacia 
def plot_threshold_metrics(df, model_name="model", optimal_thresholds=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    fig.suptitle(f"Threshold analýza - {model_name}", fontsize=13, fontweight="bold", y=0.95)
    for col in ["sensitivity", "specificity", "precision", "f1", "accuracy"]:
        ax.plot(df["threshold"], df[col], marker="o", markersize=4, label=col)
    ax.axvline(0.5, color="gray", linestyle="--", label="default t=0.5")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)
    if optimal_thresholds:
        opt_lines = "\n".join([   
            f"{f'Optimal threshold ({s}):':<32} {r['threshold']}  |  sens={r['sensitivity']:.3f}  spec={r['specificity']:.3f}  f1={r['f1']:.3f}"
            for s, r in optimal_thresholds.items()])
        ax.set_title(f"{opt_lines}")
    plt.tight_layout()
    return fig

# funkcia na hladanie optimalneho thresholdu podla roznych strategiach
def find_optimal_threshold(df, strategy="sensitivity"):
    if strategy == "f1":
        idx = df["f1"].idxmax()
    elif strategy == "sensitivity":
        filtered = df[df["specificity"] >= 0.70]
        idx = filtered["sensitivity"].idxmax() if not filtered.empty else df["sensitivity"].idxmax()
    elif strategy == "youden":  # Youden index = sensitivity + specificity - 1
        df = df.copy()
        df["youden"] = df["sensitivity"] + df["specificity"] - 1
        idx = df["youden"].idxmax()
    return df.loc[idx]


if __name__ == "__main__":
    EXPERIMENT_NAME = "NACC - finalne modely s kalibráciou"
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        output_format="pandas"
    )
    
    TARGET_RUNS = {"LogReg_FINAL_calibrated_MODEL", "XGBoost_FINAL_calibrated_MODEL", "SVM_FINAL_calibrated_MODEL"}
    for _, run in runs.iterrows():
        print(f"Analyzujem experiment: {EXPERIMENT_NAME}")
        run_id    = run["run_id"]
        model_name = run.get("params.model", "unknown")
        experiment_name = run.get("tags.mlflow.runName", run_id)
        if experiment_name not in TARGET_RUNS:
            continue
        print(f"\n>>>>>>>>> {model_name}")

        # y_true, y_proba = load_predictions_from_mlflow(run_id)        # pre analyzu nekalibrovaných modelov
        y_true, y_proba = load_calibrated_predictions_from_mlflow(run_id)
        df = threshold_metrics(y_true, y_proba)
        print(df.to_string(index=False))

        optimal_thresholds = {}
        for strategy in ["f1", "sensitivity", "youden"]:
            best = find_optimal_threshold(df, strategy)
            optimal_thresholds[strategy] = best.to_dict()
            print(f"  Optimálny threshold ({strategy}): {best['threshold']} " f"| sens={best['sensitivity']:.3f} spec={best['specificity']:.3f} f1={best['f1']:.3f}")

        fig = plot_threshold_metrics(df, model_name, optimal_thresholds=optimal_thresholds)
        fig.savefig(f"threshold_{experiment_name}.png", dpi=150)
        plt.close(fig)