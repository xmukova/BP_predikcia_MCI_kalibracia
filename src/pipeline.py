# DEFINICIA PIPELINE PRE PREDSPRACOVANIE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
from scipy.stats import rankdata
try:
    from config_based_on_EDA import (CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, FEATURES)
except ModuleNotFoundError:
    from src.config_based_on_EDA import (CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, FEATURES)

# prva cast pipeline na preprocessing
# numericke premenne sa doplnia medianom a standardizuju pomocou RobustScaler (median + IQR)
# kategoricke premenne sa doplnia modusom.
def create_pipeline_first_part():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())  #median + IQR
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")) 
    ])
    preprocessing_pipeline = ColumnTransformer([
        ("num", numeric_pipeline, CONTINUOUS_COLUMNS),
        ("cat", categorical_pipeline, CATEGORICAL_COLUMNS)
    ])
    return preprocessing_pipeline

# vlastna implementacia feature selectora, ktory kombinuje poradie premennych z MI, RFE, Random Forest a L1 regularization. 
# Vyberie k najlepsich features na zaklade priemeru poradia v tychto 4 metodach. 
# Poradie sa pre kazdu metodu vypocita tak, ze najlepsia feature ma najvyssie poradie (vyssie poradie = lepsia feature)
class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k_features=20):
        self.k_features = k_features
        self.selected_features_ = None
        self.feature_scores_ = None

    def fit(self, X, y):
        scores = pd.DataFrame(index=range(X.shape[1]))
        
        # Mutual Information
        mi = mutual_info_classif(X, y, random_state=42)     
        scores["mi"] = self.rank(mi)

        # RFE s Logistic Regression
        rfe = RFE(estimator=LogisticRegression(max_iter=2000, random_state=42), n_features_to_select=1 )
        rfe.fit(X, y)
        scores["rfe"] = self.rank(rfe.ranking_ * -1)  # 1=najlepsi ale moj ranking funguje naopak, preto *-1

        # Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        scores["rf"] = self.rank(rf.feature_importances_)

        # L1 Logistic Regression
        l1 = LogisticRegression(penalty="l1", solver="saga", max_iter=2000, random_state=42 )
        l1.fit(X, y)
        l1_importance = np.abs(l1.coef_[0])
        scores["l1"] = self.rank(l1_importance)

        # Priemer rankingov
        scores["mean_rank"] = scores.mean(axis=1)
        self.feature_scores_ = scores
        top_k_idx = scores["mean_rank"].nlargest(self.k_features).index
        self.selected_features_ = np.array(top_k_idx)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_]
        return X[:, self.selected_features_]

    def rank(self, scores):     # prevedenie skore na poradie (1 = worst, n = best - vyssie = lepsie)
        return rankdata(scores, method="average")  
    
# definicia feature selectorov
def get_feature_selector(feature_selection=None, k_features=len(FEATURES)):    
    # Mutual Information
    if feature_selection == "mi":               
        selector = SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=42), k=k_features) 
    # Recursive Feature Elimination s Logistic Regression
    elif feature_selection == "rfe":
        selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=k_features)
    # Random Forest
    elif feature_selection == "random_forest":
        selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=150, random_state=42), max_features=k_features, threshold=-np.inf)
    # L1 regularization s Logistic Regression (LASSO)
    elif feature_selection == "l1":
        selector = SelectFromModel(estimator=LogisticRegression(penalty="l1", solver="saga", max_iter=1000, random_state=42), max_features=k_features, threshold=-np.inf)
    # Custom - kombinacia poradia v MI, RFE, Random Forest a L1 regularization
    elif feature_selection == "custom":
        selector = CustomFeatureSelector(k_features = k_features)
    else:
        selector = "passthrough"
    return selector

# definicia modelov
def get_model(model_name, model_parameters=None):
    model_parameters = model_parameters if model_parameters is not None else {}
    if model_name == "lr":
        return LogisticRegression(
            max_iter=3000,
            random_state=42,
            class_weight="balanced",
            **model_parameters
        )

    elif model_name == "svm":
        return SVC(
            probability=True,
            random_state=42,
            class_weight="balanced",
            **model_parameters
        )

    elif model_name == "xgboost":
        return XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            **model_parameters
        )

    else:
        raise ValueError("Unknown model name")

# completna pipeline = preprocessing + feature selection + model
def create_pipeline_complete(model_name, feature_selection=None, k_features=30, model_parameters=None):
    pipeline = Pipeline([
        ("preprocessing", create_pipeline_first_part()),
        ("feature_selection", get_feature_selector(feature_selection, k_features)),
        ("model", get_model(model_name, model_parameters))
    ])
    return pipeline