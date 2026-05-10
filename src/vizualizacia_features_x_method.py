# vizualizacia vybranych features z experimentov napriec feature selection metodami
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")
EXPERIMENT_NAME = "BP:NACC - nastavenie k_features"
K = 15
MODEL = "lr"

# nacitanie vybranych features napriec metodami vyberu
def load_selected_features(experiment_name, model, k):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.status = 'FINISHED' and params.model = '{model}' and params.k_features = '{k}'",
        output_format="pandas"
    )
    runs = runs[runs["tags.mlflow.runName"].str.startswith(f"{model}_H5_fs_")] 
    result = {}
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        fs_method = run["params.feature_selection"]
        try:
            path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="selected_features.json")
            with open(path) as f:
                features = json.load(f)["selected_features"]
            result[fs_method] = features
            print(f"Načítané {len(features)} features pre {fs_method}")
        except Exception as e:
            print(f"  Nepodarilo sa načítať features pre {fs_method}: {e}")
    return result

# vizualizacia vybranych features pre kazdu metodu, na porovnanie napriec metodami
def plot_heatmap(features_by_method, k, model):
    all_features = sorted(set(f for feats in features_by_method.values() for f in feats))
    methods      = sorted(features_by_method.keys())
    
    matrix = pd.DataFrame(0, index=all_features, columns=methods)
    for method, feats in features_by_method.items():
        for f in feats:
            matrix.loc[f, method] = 1

    matrix["počet"] = matrix.sum(axis=1)
    matrix = matrix.sort_values("počet", ascending=False)  
    count_col = matrix[["počet"]]
    heatmap_data = matrix[methods]

    fig, axes = plt.subplots(1, 2, figsize=(10, len(all_features) * 0.28 + 2), gridspec_kw={"width_ratios": [5, 0.5], "wspace": 0.05})
    sns.heatmap(heatmap_data, ax=axes[0], cmap="Blues", linewidths=0.5, linecolor="lightgray", cbar=False, annot=False, )
    axes[0].set_title("Porovnanie vybraných features naprieč feature selection metódami", fontsize=12, pad=12)
    axes[0].xaxis.set_label_position("top")
    axes[0].xaxis.tick_top()
    axes[0].set_xlabel("Metóda", fontsize=10, labelpad=8)
    axes[0].set_aspect("auto") 
    axes[0].set_ylabel("Feature", fontsize=10)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].tick_params(axis="y", rotation=0, labelsize=8)
    sns.heatmap(count_col, ax=axes[1], cmap="Oranges", linewidths=0.5, linecolor="lightgray", cbar=False, annot=True, fmt="d", annot_kws={"size": 8},)
    axes[1].xaxis.set_label_position("top")
    axes[1].xaxis.tick_top()
    axes[1].set_ylabel("")
    axes[1].set_aspect("auto")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[1].tick_params(axis="x", bottom=False, labelbottom=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("porovnanie_featureselectionmetod.png", dpi=150, bbox_inches="tight") 
    plt.show()   
    

if __name__ == "__main__":
    features_by_method = load_selected_features(EXPERIMENT_NAME, MODEL, K)
    plot_heatmap(features_by_method, K, MODEL)