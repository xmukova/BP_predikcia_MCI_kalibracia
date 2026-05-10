# VIZUALIZACIA VPLYVU POCTU FEATURES (k) NA VYSLEDKY MODELU
import mlflow
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")

# vizualizacia
def plot_k_sweep(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        output_format="pandas"
    )
  
    modely = [("LogReg",      "LR_H5_sweep"), 
                ("SVM",     "SVM_H5_sweep"), 
                ("XGBoost", "XGB_H5_sweep")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (label, prefix) in zip(axes, modely):
        model_runs = runs[runs["tags.mlflow.runName"].str.startswith(prefix)].copy()
        if model_runs.empty:
            ax.set_title(f"{label} – žiadne dáta")
            continue

        model_runs["k"] = model_runs["params.k_features"].astype(float)
        model_runs = model_runs.sort_values("k")

        # CV ROC AUC
        ax.plot(model_runs["k"], model_runs["metrics.cv_roc_auc_test_mean"], marker="o", label="CV ROC-AUC")
        ax.fill_between(        #95% confidence interval
            model_runs["k"],
            model_runs["metrics.cv_roc_auc_test_mean"] - (1.96 * model_runs["metrics.cv_roc_auc_test_std"]),
            model_runs["metrics.cv_roc_auc_test_mean"] + (1.96 * model_runs["metrics.cv_roc_auc_test_std"]),
            alpha=0.2
        )
        # Test ROC AUC
        ax.plot(model_runs["k"], model_runs["metrics.test_roc_auc"], marker="s", linestyle="--", label="Test ROC-AUC")
        for k_val in model_runs["k"]:
            ax.axvline(x=k_val, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{label} – vplyv počtu features")
        ax.set_xlabel("Počet features (k)")
        ax.set_ylabel("ROC-AUC")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(model_runs["k"].tolist())
        ax.set_xticklabels(model_runs["k"].astype(int).tolist())

    plt.tight_layout()
    plt.savefig("porovnanie_k_features_H5.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    # zobrazujeme vplyv poctu features z vykonanych experimentoch zachytenych v mlflow pre rozne k
    EXPERIMENT_NAME = "BP:NACC - nastavenie k_features"
    plot_k_sweep(EXPERIMENT_NAME)