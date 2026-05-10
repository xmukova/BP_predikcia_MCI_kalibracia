# TRAINOVANIE  MODELOV PODĽA KONFIGURÁCIE V config_experiments.py A TRACKOVANIM VYKONNOSTI MODELOV V mlflow
import matplotlib
matplotlib.use("Agg")  
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from pipeline import create_pipeline_complete
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score, recall_score, precision_score, f1_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay,)
from config_based_on_EDA import (TARGET_COLUMN, FEATURES, CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS)
from config_experiments import EXPERIMENT, SWEEP_EXPERIMENTS, SWEEP_FEATURE_SELECTION, FINAL_MODELS, SPLIT_EXPERIMENT
import os
import json
import hashlib
import mlflow
import mlflow.sklearn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")
DATA_PATH = "data/ready_for_EDA/nacc_ready_for_eda_2025_HORIZONT_5.csv"


# nacitanie dat
def load_data():
    data = pd.read_csv(DATA_PATH, low_memory=False)
    X = data[FEATURES]
    Y = data[TARGET_COLUMN]
    return X, Y

# rozdelenie dat na trenovaci validacny a testovaci dataset (60% train, 20% val, 20% test) so stratifikaciou
def split_data_60_20_20(X, Y, random_state=42):
    # 60 % train, 20 % val, 20 % test
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    # 20 / 80 = 0.25 - validacny je 20% z celeho
    X_train, X_calib, Y_train, Y_calib = train_test_split(X_temp, Y_temp, test_size=0.25, stratify=Y_temp, random_state=42)

    print("Train:", X_train.shape[0], "zaznamov, z toho", "pozitivnych pacientov: ", Y_train[Y_train == 1].sum())
    print("Calib:", X_calib.shape[0], "zaznamov, z toho", "pozitivnych pacientov: ", Y_calib[Y_calib == 1].sum())
    print("Test:", X_test.shape[0], "zaznamov, z toho", "pozitivnych pacientov: ", Y_test[Y_test == 1].sum())
    return X_train, X_calib, X_test, Y_train, Y_calib, Y_test

# cross-validacia ktora sa vykonava pocas treningu na trenovacej mnozine a zistenie vykonnosti
def cross_validation(pipeline, X_train, Y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metriky = {
        "roc_auc":   "roc_auc",
        "avg_precision": "average_precision",
        "precision": "precision",
        "recall":    "recall",   # = sensitivity
    }
    results = cross_validate(
        pipeline, X_train, Y_train,
        cv=cv,
        scoring=metriky,
        return_train_score=True,
        n_jobs=-1       
    )
    summary = {}
    for metric in metriky:
        score = results[f"test_{metric}"]
        summary[metric] = {
            "mean": score.mean(),
            "std":  score.std(),
            "scores": score.tolist()
        }

        test_score  = results[f"test_{metric}"]
        train_score = results[f"train_{metric}"]  
        summary[metric] = {
            "test_mean":  test_score.mean(),
            "test_std":   test_score.std(),
            "train_mean": train_score.mean(),     
            "train_std":  train_score.std(),      
            "scores":     test_score.tolist()
        }
    return summary

# vypis vysledkov cross-validacie
def print_cv_results(cv_results, model_name):
    print(f"\n{'='*65}")
    print(f"  Cross-validácia: {model_name}  (5-fold)")
    print(f"{'='*65}")
    print(f"  {'Metrika':<20} {'Train ± Std':>15}  {'Val ± Std':>15}  {'Rozdiel':>8}")
    print(f"  {'-'*60}")
    for metric, vals in cv_results.items():
        diff = vals['train_mean'] - vals['test_mean']
        train_str = f"{vals['train_mean']:.4f} ±{vals['train_std']:.4f}"
        val_str   = f"{vals['test_mean']:.4f} ±{vals['test_std']:.4f}"
        print(f"  {metric:<20} {train_str:>15}  {val_str:>15}  {diff:>8.4f}")
    print(f"{'='*65}")    

# vypis vysledkov metrik na testovacej mnozine
def evaluate_model(y_true, y_pred, y_proba, model_name="model"):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metriky = {
        "roc_auc":       roc_auc_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
        "sensitivity":   recall_score(y_true, y_pred),
        "specificity":   tn / (tn + fp),
        "precision":     precision_score(y_true, y_pred),
        "f1":            f1_score(y_true, y_pred),
        "accuracy":      accuracy_score(y_true, y_pred),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }

    # výpis metrík
    print(f"\n{'='*50}")
    print(f"  Evaluácia modelu: {model_name}")
    print(f"{'='*50}")
    print(f"  ROC-AUC:        {metriky['roc_auc']:.4f}")
    print(f"  Avg Precision:  {metriky['avg_precision']:.4f}")
    print(f"  Sensitivity:    {metriky['sensitivity']:.4f}")
    print(f"  Specificity:    {metriky['specificity']:.4f}")
    print(f"  Precision:      {metriky['precision']:.4f}")
    print(f"  F1-score:       {metriky['f1']:.4f}")
    print(f"  Accuracy:       {metriky['accuracy']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred 0    Pred 1")
    print(f"  Skutočné 0:   {tn:>5}     {fp:>5}")
    print(f"  Skutočné 1:   {fn:>5}     {tp:>5}")
    print(f"{'='*50}\n")
    return metriky

# vizualizacia ROC a PR kriviek
def make_plots(y_true, y_proba, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=axes[0], name=model_name)
    axes[0].plot([0, 1], [0, 1], "k--", label="Random")
    axes[0].set_title(f"ROC krivka — {model_name}")
    axes[0].legend()
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=axes[1], name=model_name)
    axes[1].set_title(f"Precision-Recall krivka — {model_name}")
    plt.tight_layout()
    return fig

# hashovanie konfiguracie experimentu pre identifikaciu v mlflow a zistenie ci uz bol spusteny
def get_config_hash(config):
    config_copy = dict(config)
    config_copy.pop("note", None)  # ignorujem poznamku
    config_str = json.dumps(config_copy, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

# ziskanie existujucich runov v experimente v mlflow (aby sa nerobili rovnake experimenty zbytocne viac krat)
def get_existing_experiments(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return set()
    
    hotovo = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], 
        filter_string="attributes.status = 'FINISHED'",
        output_format="pandas")
    if hotovo.empty:
        return set()
    
    return set(hotovo["tags.config_hash"].dropna().tolist())

# hlavna funkcia ktora spusti experiment podla konfiguracie, zaznamena vysledky v mlflow a vrati vysledky
def run_experiment(config, X_train, X_calib, X_test, Y_train, Y_calib, Y_test):
    model_params = config["model_parameters"].copy()
    if model_params.get("scale_pos_weight") == "auto":      # nastavenie parametru scale_pos_weight pre XGBoost kvoli nevyvazenosti
        xgboost_weight = Y_train.value_counts()[0] / Y_train.value_counts()[1]
        model_params["scale_pos_weight"] = xgboost_weight
    
    with mlflow.start_run(run_name = config["nazov"]):
        # zaznamenanie parametrov
        mlflow.log_param("model",             config["model_name"])
        mlflow.log_param("feature_selection", config["feature_selection"])
        mlflow.log_param("k_features",        config["k_features"])
        for i, j in model_params.items():
            mlflow.log_param(i, j)

        mlflow.set_tag("poznamka", config.get("note", ""))
        mlflow.set_tag("version", config.get("version", "v1"))
        config_hash = get_config_hash(config)
        mlflow.set_tag("config_hash", config_hash)

        pipeline = create_pipeline_complete(
            model_name = config["model_name"],
            feature_selection = config["feature_selection"],
            k_features = config["k_features"],
            model_parameters = model_params 
        )
        # cross-validation
        cv_results = cross_validation(pipeline, X_train, Y_train)
        print_cv_results(cv_results, config["nazov"])
        for metric, vals in cv_results.items():
            mlflow.log_metric(f"cv_{metric}_test_mean",  round(vals["test_mean"], 3))
            mlflow.log_metric(f"cv_{metric}_test_std",   round(vals["test_std"], 3))
            mlflow.log_metric(f"cv_{metric}_train_mean", round(vals["train_mean"], 3)) 
            diff = vals["train_mean"] - vals["test_mean"]
            mlflow.log_metric(f"cv_{metric}_difference_between_train_test", round(diff, 3))           

        pipeline.fit(X_train, Y_train)

        # lognutie vybranych features
        try:
            all_cols = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
            feature_selection = pipeline.named_steps["feature_selection"]
            if hasattr(feature_selection, "selected_features_"):       # CustomFeatureSelector
                selected = [all_cols[i] for i in feature_selection.selected_features_]
            elif hasattr(feature_selection, "get_support"):            # SelectKBest, RFE, SelectFromModel
                selected = [all_cols[i] for i, v in enumerate(feature_selection.get_support()) if v]
            else:
                selected = all_cols
            mlflow.log_dict({"selected_features": selected}, "selected_features.json")
        except Exception as e:
            print(f"  Nepodarilo sa extrahovať features: {e}")

        # test predikcie na finalnu evaluaciu
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        test_results  = evaluate_model(Y_test, y_pred, y_proba, model_name=config["nazov"])

        for i, j in test_results.items():
            if isinstance(j, float):
                mlflow.log_metric(f"test_{i}", round(j, 3))

        mlflow.log_dict({
            "y_true": Y_test.tolist(),
            "y_prob": y_proba.tolist()
        }, "predictions.json")        

        # validacne predikcie na kalibraciu
        y_proba_calib = pipeline.predict_proba(X_calib)[:, 1]
        mlflow.log_dict({
            "y_true": Y_calib.tolist(),
            "y_prob": y_proba_calib.tolist()
        }, "predictions_validation.json")

        fig = make_plots(Y_test, y_proba, config["nazov"])
        mlflow.log_figure(fig, f"curves_{config['nazov']}.png")
        plt.close(fig)

        mlflow.sklearn.log_model(pipeline, name=config["model_name"])

        print(f"\nModel '{config['nazov']}' zaznamenany v mlflow a natrenovany.\n")
        return {"cv": cv_results, "test": test_results}


def train():
    # "EXPERIMENT_NAME" vyjadruje nazov experimentu v mlflow pod ktorym sa budu zaznamenavat vysledky.

    # EXPERIMENT_NAME = "BP:NACC - experimenty manual z configu"
    # EXPERIMENT_NAME = "BP:NACC - nastavenie k_features"
    # EXPERIMENT_NAME = "BP:NACC - finalne hodnotenie + kalibracia"
    EXPERIMENT_NAME = "NACC - experiimenty BP"
    mlflow.set_experiment(EXPERIMENT_NAME)
    X, Y = load_data()
    X_train, X_calib, X_test, Y_train, Y_calib, Y_test = split_data_60_20_20(X, Y)
    
    experimenty_hotovo = get_existing_experiments(EXPERIMENT_NAME)
    # "new_experiment" obsahuje konfiguracie experimentov, ktore chcem spustit a zaroven neboli este v mlflow
    new_experiment = [e for e in FINAL_MODELS if get_config_hash(e) not in experimenty_hotovo]
    # new_experiment = [e for e in EXPERIMENT if get_config_hash(e) not in experimenty_hotovo]
    # new_experiment = [e for e in SWEEP_EXPERIMENTS if get_config_hash(e) not in experimenty_hotovo]
    # new_experiment = [e for e in SWEEP_FEATURE_SELECTION if get_config_hash(e) not in experimenty_hotovo]

    if not new_experiment:
        print("Neexistujú žiadne nespustené experimenty")
        return
    
    print(f"Nové experimenty na spustenie: {[e['nazov'] for e in new_experiment]}")

    vysledky = {}
    for i in new_experiment:
        print(f"\n>>> Spúšťam: {i['nazov']}")
        vysledky[i["nazov"]] = run_experiment( i, X_train, X_calib, X_test, Y_train, Y_calib, Y_test )

if __name__ == "__main__":
    train()