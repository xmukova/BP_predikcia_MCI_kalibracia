# Hladanie najlepsich hyperparametrov modelu pomocou Bayesian optimization a Optuna kniznice 
import optuna
import mlflow
from sklearn.model_selection import StratifiedKFold, cross_validate
from train_model import load_data, split_data_60_20_20
from pipeline import create_pipeline_complete
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# definicia prehladavacieho priestoru hyperparametrov pre logisticku regresiu
# podla penalty sa nastavuje vhodny solver
def lr_parameters(trial):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    if penalty == "elasticnet":
        solver = "saga"
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True),
                "penalty": penalty, 
                "solver": solver, 
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)}
    elif penalty == "l1":
        solver = "saga"
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True),
                "penalty": penalty, 
                "solver": solver}
    else:  # l2
        solver = trial.suggest_categorical("solver", ["saga", "lbfgs"])
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True),
                "penalty": penalty, 
                "solver": solver}

# definicia prehladavacieho priestoru hyperparametrov pre SVM
def svm_parameters(trial): return {
        "C":      trial.suggest_float("C", 1e-2, 10, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
        "gamma":  trial.suggest_categorical("gamma", ["scale", "auto", 0.001, 0.1]),}

# definicia prehladavacieho priestoru hyperparametrov pre XGBoost
def xgboost_parameters(trial): return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 700),
        "max_depth":         trial.suggest_int("max_depth", 2, 6),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 50),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 20.0),
        "scale_pos_weight":  "auto",}

PARAMETER_SEARCH = {
    "lr":      lr_parameters,
    "svm":     svm_parameters,
    "xgboost": xgboost_parameters,
}
# tuning modelu pomocou Bayesian optimization z kniznice Optuna, 
# ktora inteligentne prehladava priestor hyperparametrov a snazi sa najst najlepsie kombinacie pre dany model
def tune_model(model_name, feature_selection="custom", k_features=30, n_trials=40, metric="roc_auc", X_train=None, Y_train=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    def objective(trial):
        parametre = PARAMETER_SEARCH[model_name](trial)
        if parametre.get("scale_pos_weight") == "auto":
            parametre["scale_pos_weight"] = Y_train.value_counts()[0] / Y_train.value_counts()[1]

        pipeline = create_pipeline_complete(
            model_name=model_name,
            feature_selection=feature_selection,
            k_features=k_features,
            model_parameters=parametre,
        )

        results = cross_validate(pipeline, X_train, Y_train, cv=cv, scoring=metric, return_train_score=True, n_jobs=-1)
        test_mean  = results["test_score"].mean()
        train_mean = results["train_score"].mean()
        gap        = abs(train_mean - test_mean)
        trial.set_user_attr("test_mean",  round(test_mean, 4))
        trial.set_user_attr("train_mean", round(train_mean, 4))
        trial.set_user_attr("gap",        round(gap, 4))
        return test_mean - 0.1 * gap #penalizujem modely, ktore maju velky gap medzi train a test aby som nevybrala overfitt modely

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))   # Bayesian optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

# ulozenie vysledkov tuningu do mlflow, vratane najlepsich parametrov
def log_tuning_results(study, model_name, feature_selection, k_features, metric):
    best = study.best_trial
    run_name = f"tuning_{model_name}_best_HORIZONT_5"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model",             model_name)
        mlflow.log_param("feature_selection", feature_selection)
        mlflow.log_param("k_features",        k_features)
        mlflow.log_param("n_trials",          len(study.trials))
        mlflow.log_param("optimized_metric",  metric)

        for k, v in best.params.items():
            mlflow.log_param(f"best_{k}", v)

        mlflow.log_metric(f"best_cv_{metric}", best.value)
        mlflow.log_dict(best.params, "best_parametre.json")
        history = [{"trial": t.number, "value": t.value} for t in study.trials if t.value is not None ]
        mlflow.log_dict({"optimization_history": history}, "optuna_history.json")

    print(f"\nNajlepšie parametre pre {model_name}:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"  CV {metric}: {best.value:.4f}")
    return best.params

# spustenie tuningu modelov podla definovanej konfiguracie
def run_tuning():
    EXPERIMENT_NAME = "BP:NACC - hyperparameter tuning"
    mlflow.set_experiment(EXPERIMENT_NAME)
    X, Y = load_data()
    X_train, X_calib, X_test, Y_train, Y_calib, Y_test = split_data_60_20_20(X, Y)
    TUNING_CONFIGS = [
        {   "model_name":        "lr",
            "feature_selection": "custom",
            "k_features":        30,
            "n_trials":          20,
            "metric":            "roc_auc",
        },
        {   "model_name":        "svm",
            "feature_selection": "custom",
            "k_features":        30,
            "n_trials":          20,
            "metric":            "roc_auc",},
        {   "model_name":        "xgboost",
            "feature_selection": "custom",
            "k_features":        30,
            "n_trials":          40,
            "metric":            "roc_auc",},
    ]

    for cfg in TUNING_CONFIGS:
        print(f"\n>>> Tuning: {cfg['model_name']} ({cfg['n_trials']} trials)")
        study = tune_model(**cfg, X_train=X_train, Y_train=Y_train)
        log_tuning_results(
            study,
            model_name=cfg["model_name"],
            feature_selection=cfg["feature_selection"],
            k_features=cfg["k_features"],
            metric=cfg["metric"],
        )

if __name__ == "__main__":
    run_tuning()