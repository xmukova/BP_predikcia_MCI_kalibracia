# Test štatistickej významnosti rozdielu medzi k=15 a k=10  a k= 20 features
import json
import numpy as np
from scipy import stats
import pandas as pd
import mlflow
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")
EXPERIMENT_NAME = "BP:NACC - nastavenie k_features"

# definicia runov ktore porovnavame z mlflow experimentu
COMPARE_K = {
    "lr": {
        "k10": "LR_H5_sweep_k10",
        "k15": "LR_H5_sweep_k15",
        "k20": "LR_H5_sweep_k20",
    },
    "svm": {
        "k10": "SVM_H5_sweep_k10",
        "k15": "SVM_H5_sweep_k15",
        "k20": "SVM_H5_sweep_k20",
    },
    "xgboost": {
        "k10": "XGB_H5_sweep_k10",
        "k15": "XGB_H5_sweep_k15",
        "k20": "XGB_H5_sweep_k20",
    }
}

# nacitanie predikcii
def load_predictions_by_run_names(experiment_name, target_runs):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' neexistuje.")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'"
    )
    loaded = {}
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName")
        if run_name not in target_runs:
            continue

        try:
            path = client.download_artifacts(run.info.run_id, "predictions.json")
            with open(path, "r") as f:
                data = json.load(f)

            loaded[run_name] = {
                "y_true": np.array(data["y_true"]),
                "y_prob": np.array(data["y_prob"])
            }

            print(f"Načítané: {run_name}")
        except Exception as e:
            print(f"Chyba pri {run_name}: {e}")

    return loaded

# vypocet ROCUAC a variance pre DeLong test
def compute_auc(y_true, y_predicted):
    pos_scores = y_predicted[y_true == 1]
    neg_scores = y_predicted[y_true == 0]
    positive_pocet = len(pos_scores)
    negative_pocet = len(neg_scores)

    V10 = np.array([np.mean(p > neg_scores) + 0.5 * np.mean(p == neg_scores) for p in pos_scores]) # kolko negativnych predikcii porazi kazda pozitivna predikcia
    V01 = np.array([np.mean(q < pos_scores) + 0.5 * np.mean(q == pos_scores) for q in neg_scores]) # kolko pozitivnych predikcii porazi kazdu negativnu predikciu

    auc = np.mean(V10)
    variance = np.var(V10, ddof=1) / positive_pocet + np.var(V01, ddof=1) / negative_pocet
    return auc, variance, V10, V01

# DeLong test pre porovnanie statistickej vyznamnosti rozdielu dvoch AUC
def delong_test(y_true, y_predicted_1, y_predicted_2):
    pos_idx = y_true == 1
    neg_idx = y_true == 0
    positive_pocet = pos_idx.sum()
    negative_pocet = neg_idx.sum()
    auc1, var1, V10_1, V01_1 = compute_auc(y_true, y_predicted_1)
    auc2, var2, V10_2, V01_2 = compute_auc(y_true, y_predicted_2)
    cov = (np.cov(V10_1, V10_2)[0, 1] / positive_pocet) + (np.cov(V01_1, V01_2)[0, 1] / negative_pocet)
    std_dev = np.sqrt(var1 + var2 - 2 * cov)
    z = (auc1 - auc2) / std_dev             
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auc1": auc1,
        "auc2": auc2,
        "auc_diff": auc1 - auc2,
        "z_statistic": z,
        "p_value": p,
        "significant": p < 0.05
    }

# porovnanie modelov pre k=10 vs k=15 a k=15 vs k=20 features pomocou DeLong testu a vypis vysledkov
def compare_models():
    all_target_runs = set()
    for model_dict in COMPARE_K.values():
        all_target_runs.update(model_dict.values())

    predictions = load_predictions_by_run_names(EXPERIMENT_NAME, all_target_runs)

    results = []
    for model_name, config in COMPARE_K.items():
        run_k10 = config["k10"]
        run_k15 = config["k15"]
        run_k20 = config["k20"]
        if (run_k10 not in predictions or run_k15 not in predictions or run_k20 not in predictions):
            print(f"Chýbajú dáta pre {model_name}")
            continue

        y_true_10 = predictions[run_k10]["y_true"]
        y_prob_10 = predictions[run_k10]["y_prob"]

        y_true_15 = predictions[run_k15]["y_true"]
        y_prob_15 = predictions[run_k15]["y_prob"]

        y_true_20 = predictions[run_k20]["y_true"]
        y_prob_20 = predictions[run_k20]["y_prob"]
        
        if not (np.array_equal(y_true_10, y_true_15) and np.array_equal(y_true_15, y_true_20)):
            print(f"WARNING: y_true sa nezhoduje pre {model_name}")

        # DeLong test: 10 vs 15
        result_10_vs_15 = delong_test(y_true_10, y_prob_10, y_prob_15)
        p_value_10_vs_15 = result_10_vs_15["p_value"]
        results.append({
            "Model": model_name.upper(),
            "Porovnanie": "10 vs 15",
            "AUC 1": round(result_10_vs_15["auc1"], 4),
            "AUC 2": round(result_10_vs_15["auc2"], 4),
            "AUC diff": round(result_10_vs_15["auc_diff"], 4),
            "p-value": round(p_value_10_vs_15, 6),
            "Významné": "Áno" if p_value_10_vs_15 < 0.05 else "Nie"
        })    

        # DeLong test: 15 vs 20
        result_15_vs_20 = delong_test(y_true_15, y_prob_15, y_prob_20)
        p_value_15_vs_20 = result_15_vs_20["p_value"]
        results.append({
            "Model": model_name.upper(),
            "Porovnanie": "15 vs 20",
            "AUC 1": round(result_15_vs_20["auc1"], 4),
            "AUC 2": round(result_15_vs_20["auc2"], 4),
            "AUC diff": round(result_15_vs_20["auc_diff"], 4),
            "p-value": round(p_value_15_vs_20, 6),
            "Významné": "Áno" if p_value_15_vs_20 < 0.05 else "Nie"
        })

    df = pd.DataFrame(results)
    print("\n")
    print("=" * 70)
    print("DeLong test: 10 vs 15 a 15 vs 20 premenných")
    print("=" * 70)
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    df_results = compare_models()