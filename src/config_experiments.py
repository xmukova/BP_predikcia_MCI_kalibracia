# KONFIGURACNY SUBOR EXPERIMENTOV, KTORE SA SPUSTAJU V train_model.py
EXPERIMENT = [  # zakladne rucne experimenty
    {"nazov": "LogReg_baseline",
        "model_name": "lr",
        "feature_selection": None,
        "k_features": None,
        "model_parameters": {
            "C": 1.0,
            "solver": "lbfgs"}},
    {"nazov": "SVM_baseline",
        "model_name": "svm",
        "feature_selection": None,
        "k_features": None,
        "model_parameters": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale"}},
    {"nazov": "XGB_baseline",
        "model_name": "xgboost",
        "feature_selection": None,
        "k_features": None,
        "model_parameters": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8}},
    ############################################################################
    ########################### Logistic regression - nastavenie hyperparametrov
    ############################################################################
    {"nazov": "LogReg_baseline_c1_l2_lbfgs",
     "model_name": "lr",
     "feature_selection": None,
     "k_features": None,
     "version": "v1",
     "note": "no feature selection",
     "model_parameters": {
         "C": 1.0, 
         "solver": "lbfgs", 
         "penalty": "l2"}},

    {"nazov": "LogReg_baseline_c0.1_l2_lbfgs",
     "model_name": "lr",
     "feature_selection": None,
     "k_features": None,
     "version": "v1",
     "note": "no feature selection",
     "model_parameters": {
         "C": 0.1,              # silnejsia regularizacia
         "solver": "lbfgs", 
         "penalty": "l2"}},

    {"nazov": "LogReg_baseline_c0.01_l2_lbfgs",
     "model_name": "lr",
     "feature_selection": None,
     "k_features": None,
     "version": "v1",
     "note": "no feature selection",
     "model_parameters": {
         "C": 0.01,              # silna regularizacia
         "solver": "lbfgs", 
         "penalty": "l2"}},

    {"nazov": "LogReg_baseline_c10_l2_lbfgs",
     "model_name": "lr",
     "feature_selection": None,
     "k_features": None,
     "version": "v1",
     "note": "no feature selection",
     "model_parameters": {
         "C": 10.0,              # slaba regularizacia
         "solver": "lbfgs", 
         "penalty": "l2"}},     

    {"nazov": "LogReg_baseline_c1_l1_saga",
     "model_name": "lr",
     "feature_selection": None,
     "k_features": None,
     "version": "v1",
     "note": "no feature selection",
     "model_parameters": {
         "C": 1.0,              
         "solver": "saga",          # solver saga 
         "penalty": "l1"}},         # l1 regularizacia

    {"nazov": "LogReg_baseline_c0.1_l1_saga",
     "model_name": "lr",
     "feature_selection": None,
     "k_features": None,
     "version": "v1",
     "note": "no feature selection",
     "model_parameters": {
         "C": 0.1,                  #0.1 silna regularizacia
         "solver": "saga",          # solver saga 
         "penalty": "l1"}},         # l1 regularizacia     

    {"nazov": "LogReg_baseline_c1_elasticnet_saga",
     "model_name": "lr",
     "feature_selection": None,
     "k_features": None,
     "version": "v1",
     "note": "no feature selection",
     "model_parameters": {
         "C": 1.0,              
         "solver": "saga",          
         "penalty": "elasticnet"}},         # elasticnet
    # logistic regression - feature selection
    {"nazov": "LogReg_mi30_c0.1_l1_saga",
     "model_name": "lr",
     "feature_selection": "mi",             #mi
     "k_features": 30,                      #30
     "version": "v2",
     "note": "feature selection",
     "model_parameters": {
         "C": 0.1,                  
         "solver": "saga",           
         "penalty": "l1"}},  
    
    {"nazov": "LogReg_rf40_c0.1_l1_saga",
     "model_name": "lr",
     "feature_selection": "rf",             #rf
     "k_features": 40,                      #40
     "version": "v2",
     "note": "feature selection",
     "model_parameters": {
         "C": 0.1,                  
         "solver": "saga",           
         "penalty": "l1"}}, 

    {"nazov": "LogReg_l140_c0.1_l1_saga",
     "model_name": "lr",
     "feature_selection": "l1",             #l1
     "k_features": 40,                      #40
     "version": "v2",
     "note": "feature selection",
     "model_parameters": {
         "C": 0.1,                  
         "solver": "saga",           
         "penalty": "l1"}},   

    {"nazov": "LogReg_custom20_c0.1_l1_saga",
     "model_name": "lr",
     "feature_selection": "custom",             #custom
     "k_features": 20,                      #20
     "version": "v2",
     "note": "feature selection",
     "model_parameters": {
         "C": 0.1,                  
         "solver": "saga",           
         "penalty": "l1"}}, 

     {"nazov": "LogReg_rf10_c1_l2_lbfgs",
     "model_name": "lr",
     "feature_selection": "rf",             #rf
     "k_features": 10,                          #10
     "version": "v2",
     "note": "feature selection",
     "model_parameters": {
         "C": 1,                  
         "solver": "lbfgs",           
         "penalty": "l2"}},  

    ############################################################################
    ########################### Logistic regression - nastavenie hyperparametrov
    ############################################################################     
    {"nazov": "SVM_linear",
     "model_name": "svm",
     "feature_selection": None,             
     "k_features": None,                          
     "version": "v1",
     "note": "no feature selection",
     "model_parameters": {
         "C": 1.0, 
         "kernel": "linear"}},               # linear kernel pre porovnanie

    {"nazov": "SVM_rbf_baseline",
     "model_name": "svm",
     "feature_selection": None,             
     "k_features": None,                          
     "version": "v1",
     "note": "no feature selection",
     "model_parameters": {
         "C": 1.0, 
         "kernel": "rbf",                   # rbf 
         "gamma": "scale"}},                 

    {"nazov": "SVM_rbf_highC",
    "model_name": "svm",
    "feature_selection": None,             
    "k_features": None,                          
    "version": "v1",
    "note": "no feature selection",
    "model_parameters": {
        "C": 100.0,                     # 
        "kernel": "rbf",
        "gamma": "scale"}},

    {"nazov": "SVM_rbf_lowC",
    "model_name": "svm",
    "feature_selection": None,             
    "k_features": None,                          
    "version": "v1",
    "note": "no feature selection",
    "model_parameters": {
        "C": 0.01,                  #   
        "kernel": "rbf",
        "gamma": "scale"}},   

    {"nazov": "SVM_rbf_C0.1",
    "model_name": "svm",
    "feature_selection": None,             
    "k_features": None,                          
    "version": "v1",
    "note": "no feature selection",
    "model_parameters": {
        "C": 0.1,                  #   
        "kernel": "rbf",
        "gamma": "scale"}},    

    {"nazov": "SVM_rbf_C1_highgamma",
    "model_name": "svm",
    "feature_selection": None,             
    "k_features": None,                          
    "version": "v1",
    "note": "no feature selection",
    "model_parameters": {
        "C": 1.0,                  
        "kernel": "rbf",
        "gamma": 10}},             # vplyv gamma

    {"nazov": "SVM_rbf_C1_lowgamma",
    "model_name": "svm",
    "feature_selection": None,             
    "k_features": None,                          
    "version": "v1",
    "note": "no feature selection",
    "model_parameters": {
        "C": 1.0,                  
        "kernel": "rbf",
        "gamma": 0.001}},             # vplyv gamma    

    {"nazov": "SVM_poly_deg3",
    "model_name": "svm",
    "feature_selection": None,
    "k_features": None,
    "version": "v1",
    "note": "no feature selection",
    "model_parameters": {
        "kernel": "poly",           # poly kernel
        "degree": 3,
        "C": 1.0}},

    {"nazov": "SVM_poly_deg2",
    "model_name": "svm",
    "feature_selection": None,
    "k_features": None,
    "version": "v1",
    "note": "no feature selection",
    "model_parameters": {
        "kernel": "poly",           # poly kernel
        "degree": 2,
        "C": 1.0}},    

    # SVM - feature selection
    {"nazov": "SVM_rbf_mi30",
     "model_name": "svm",
     "feature_selection": "mi",             #mi
     "k_features": 30,                      #30    
     "version": "v2",
     "note": "feature selection",
     "model_parameters": {
         "C": 1.0, 
         "kernel": "rbf",                   
         "gamma": "scale"}}, 

    {"nazov": "SVM_rbf_mi10",
     "model_name": "svm",
     "feature_selection": "mi",             #mi
     "k_features": 10,                      #10    
     "version": "v2",
     "note": "feature selection",
     "model_parameters": {
         "C": 1.0, 
         "kernel": "rbf",                   
         "gamma": "scale"}},      

    {"nazov": "SVM_rbf_l130",
     "model_name": "svm",
     "feature_selection": "l1",             #l1
     "k_features": 30,                      #30    
     "version": "v2",
     "note": "feature selection",
     "model_parameters": {
         "C": 1.0, 
         "kernel": "rbf",                   
         "gamma": "scale"}},       

    {"nazov": "SVM_rbf_rf30",
     "model_name": "svm",
     "feature_selection": "rf",             #rf
     "k_features": 30,                      #30    
     "version": "v2",
     "note": "feature selection",
     "model_parameters": {
         "C": 1.0, 
         "kernel": "rbf",                   
         "gamma": "scale"}}, 

    {"nazov": "SVM_rbf_custom30",
     "model_name": "svm",
     "feature_selection": "custom",             #custom
     "k_features": 30,                      #30    
     "version": "v2",
     "note": "feature selection",
     "model_parameters": {
         "C": 1.0, 
         "kernel": "rbf",                   
         "gamma": "scale"}},     

    ############################################################################
    ########################### XGBoost - nastavenie hyperparametrov
    ############################################################################     
    {"nazov": "XGB_baseline_with_weight",
    "model_name": "xgboost",
    "feature_selection": None,
    "k_features": None,
    "note": "pouzitie scale_pos_weight pre nevyvazeny dataset",
    "model_parameters": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": "auto"}},     # nastavenie scale_pos_weight pre nevyvazeny dataset

    {"nazov": "XGB_depth6_with_weight",
    "model_name": "xgboost",
    "feature_selection": None,
    "k_features": None,
    "note": "pouzitie scale_pos_weight pre nevyvazeny dataset, no feature selection",
    "version": "v1",
    "model_parameters": {
        "n_estimators": 100,
        "max_depth": 6,                 # depth 6
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": "auto"}},     # nastavenie scale_pos_weight pre nevyvazeny dataset    

    {"nazov": "XGB_depth8_estimators300_learningrate001_with_weight",
    "model_name": "xgboost",
    "feature_selection": None,
    "k_features": None,
    "note": "pouzitie scale_pos_weight pre nevyvazeny dataset, no feature selection",
    "version": "v1",
    "model_parameters": {
        "n_estimators": 300,            # estimators 300 
        "max_depth": 4,                 # depth 4
        "learning_rate": 0.01,          # pomala learning rate 0.01
        "subsample": 0.8,   
        "colsample_bytree": 0.8,
        "scale_pos_weight": "auto"}},     # nastavenie scale_pos_weight pre nevyvazeny dataset    

    {"nazov": "XGB_depth3_with_weight",
    "model_name": "xgboost",
    "feature_selection": None,
    "k_features": None,
    "note": "pouzitie scale_pos_weight pre nevyvazeny dataset, no feature selection",
    "version": "v1",
    "model_parameters": {
        "n_estimators": 400,
        "max_depth": 3,                 # depth 3
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": "auto"}},     # nastavenie scale_pos_weight pre nevyvazeny dataset    

    {"nazov": "XGB_subsample_colsample_with_weight",
    "model_name": "xgboost",
    "feature_selection": None,
    "k_features": None,
    "note": "pouzitie scale_pos_weight pre nevyvazeny dataset",
    "model_parameters": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 1.0,                   # 1
        "colsample_bytree": 1.0,            # 1
        "scale_pos_weight": "auto"}},     # nastavenie scale_pos_weight pre nevyvazeny dataset


    {"nazov": "XGB_lr0.01_with_weight",
    "model_name": "xgboost",
    "feature_selection": None,
    "k_features": None,
    "note": "pouzitie scale_pos_weight pre nevyvazeny dataset",
    "model_parameters": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.01,          # pomala learning rate 0.01
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": "auto"}},     # nastavenie scale_pos_weight pre nevyvazeny dataset

   {"nazov": "XGB_lr0.3_with_weight",
    "model_name": "xgboost",
    "feature_selection": None,
    "k_features": None,
    "note": "pouzitie scale_pos_weight pre nevyvazeny dataset",
    "model_parameters": {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.3,          # rychla learning rate 0.3
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": "auto"}},     # nastavenie scale_pos_weight pre nevyvazeny dataset

    {"nazov": "XGB_regularizacia_with_weight",
    "model_name": "xgboost",
    "feature_selection": None,
    "k_features": None,
    "note": "pouzitie scale_pos_weight pre nevyvazeny dataset",
    "model_parameters": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,       
        "subsample": 0.6,               # mensi subsample 0.6
        "colsample_bytree": 0.6,        # mensi colsample_bytree 0.6
        "reg_alpha": 0.1,              # L1 regularizacia
        "reg_lambda": 5.0,             # L2 regularizacia
        "scale_pos_weight": "auto"}},     # nastavenie scale_pos_weight pre nevyvazeny dataset



    # riesenie overfittingu pri XGBoost
    {"nazov": "XGB_anti_overfit_shallow",
    "model_name": "xgboost",
    "feature_selection": None,
    "k_features": None,
    "note": "riesim overfitting",
    "version": "v2",
    "model_parameters": {
        "n_estimators": 500,
        "max_depth": 3,              # plytky strom
        "learning_rate": 0.01,       # pomala LR
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,      
        "gamma": 1.0,                
        "reg_alpha": 0.1,            
        "reg_lambda": 5.0,           
        "scale_pos_weight": "auto"}},

    {"nazov": "XGB_anti_overfit_minchild",
    "model_name": "xgboost",
    "feature_selection": None,
    "k_features": None,
    "note": "riesim overfitting",
    "version": "v2",
    "model_parameters": {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 20,      
        "gamma": 0.5,
        "reg_alpha": 0.05,
        "reg_lambda": 3.0,
        "scale_pos_weight": "auto"}},


    {"nazov": "XGB_anti_overfit_slowlearn",
    "model_name": "xgboost",
    "feature_selection": None,
    "k_features": None,
    "note": "riesim overfitting",
    "version": "v2",
    "model_parameters": {
        "n_estimators": 800,         
        "max_depth": 3,
        "learning_rate": 0.005,      
        "subsample": 0.6,            
        "colsample_bytree": 0.6,
        "colsample_bylevel": 0.6,    
        "min_child_weight": 15,
        "gamma": 1.0,
        "reg_lambda": 5.0,
        "scale_pos_weight": "auto"}},

    {"nazov": "XGB_anti_overfit_all",
    "model_name": "xgboost",
    "feature_selection": None,
    "k_features": None,
    "note": "riesim overfitting",
    "version": "v2",
    "model_parameters": {
        "n_estimators": 600,
        "max_depth": 2,              
        "learning_rate": 0.01,
        "subsample": 0.65,
        "colsample_bytree": 0.65,
        "colsample_bylevel": 0.65,
        "min_child_weight": 30,      
        "gamma": 2.0,                
        "reg_alpha": 0.5,
        "reg_lambda": 10.0,
        "scale_pos_weight": "auto"}},


    {"nazov": "XGB_optuna_1",
    "model_name": "xgboost",
    "feature_selection": "custom",
    "k_features": 30,
    "note": "hyperparameter tuning",
    "version": "v3",
    "model_parameters":{
        "n_estimators": 410,
        "max_depth": 2,
        "learning_rate": 0.021,
        "subsample": 0.904,
        "colsample_bytree": 0.9955,
        "colsample_bylevel": 0.901,
        "min_child_weight": 43,
        "gamma": 1.1357,
        "reg_alpha": 0.3,
        "reg_lambda": 7,
        "scale_pos_weight": "auto"
        }},

    {"nazov": "SVM_optuna_1",
    "model_name": "svm",
    "feature_selection": "custom",
    "k_features": 30,
    "note": "hyperparameter tuning",
    "version": "v3",
    "model_parameters":{
        "C": 9.4,
        "kernel": "rbf",
        "gamma": 0.001
        }},

    {"nazov": "LogReg_optuna_1",
    "model_name": "lr",
    "feature_selection": "custom",
    "k_features": 30,
    "note": "hyperparameter tuning",
    "version": "v3",
    "model_parameters":{
        "C": 0.01,
        "penalty": "l2",
        "solver": "saga"
    }},

    {"nazov": "XGB_kandidat_best",
    "model_name": "xgboost",
    "feature_selection": "custom",
    "k_features": 30,
    "note": "",
    "version": "v3",
    "model_parameters":{
        "n_estimators": 600,
        "max_depth": 2,              
        "learning_rate": 0.01,
        "subsample": 0.65,
        "colsample_bytree": 0.65,
        "colsample_bylevel": 0.65,
        "min_child_weight": 30,      
        "gamma": 2.0,                
        "reg_alpha": 0.5,
        "reg_lambda": 10.0,
        "scale_pos_weight": "auto"
        }},

    {"nazov": "SVM_kandidat_best",
    "model_name": "svm",
    "feature_selection": "custom",
    "k_features": 30,
    "note": "",
    "version": "v3",
    "model_parameters":{
        "C": 1.0,                  
        "kernel": "rbf",
        "gamma": 0.001
        }},    

    {"nazov": "LogReg_kandidat_best",
    "model_name": "lr",
    "feature_selection": "custom",
    "k_features": 30,
    "note": "",
    "version": "v3",
    "model_parameters":{
        "C": 0.1,                  
        "solver": "lbfgs",           
        "penalty": "l2"
        }},    

    #-------- oversampling --------------
    {"nazov": "SVM_oversampling_OK_smote",
    "model_name": "svm",
    "feature_selection": "custom",
    "k_features": 30,
    "note": "oversampling CORRECT",
    "version": "v5",
    "model_parameters":{
        "C": 9.4,
        "kernel": "rbf",
        "gamma": 0.001
        }},

    {"nazov": "LogReg_oversampling_OK_smote",
    "model_name": "lr",
    "feature_selection": "custom",
    "k_features": 30,
    "note": "oversampling CORRECT",
    "version": "v5",
    "model_parameters":{
        "C": 0.01,
        "penalty": "l2",
        "solver": "saga"
    }},

    {"nazov": "XGB_oversampling_OK_smote",
    "model_name": "xgboost",
    "feature_selection": "custom",
    "k_features": 30,
    "note": "oversampling CORRECT",
    "version": "v5",
    "model_parameters":{
        "n_estimators": 600,
        "max_depth": 2,              
        "learning_rate": 0.01,
        "subsample": 0.65,
        "colsample_bytree": 0.65,
        "colsample_bylevel": 0.65,
        "min_child_weight": 30,      
        "gamma": 2.0,                
        "reg_alpha": 0.5,
        "reg_lambda": 10.0,
        #"scale_pos_weight": "auto"
        }},

    {"nazov": "SVM_oversampling_OK_smoteNC",
    "model_name": "svm",
    "feature_selection": "custom",
    "k_features": 30,
    "note": "oversampling CORRECT - smotenc",
    "version": "v5",
    "model_parameters":{
        "C": 9.4,
        "kernel": "rbf",
        "gamma": 0.001
        }},

    {"nazov": "LogReg_oversampling_OK_smoteNC",
    "model_name": "lr",
    "feature_selection": "custom",
    "k_features": 30,
    "note": "oversampling CORRECT - smotenc",
    "version": "v5",
    "model_parameters":{
        "C": 0.01,
        "penalty": "l2",
        "solver": "saga"
    }},

    {"nazov": "XGB_oversampling_OK_smoteNC",
    "model_name": "xgboost",
    "feature_selection": "custom",
    "k_features": 30,
    "note": "oversampling CORRECT - smotenc",
    "version": "v5",
    "model_parameters":{
        "n_estimators": 600,
        "max_depth": 2,              
        "learning_rate": 0.01,
        "subsample": 0.65,
        "colsample_bytree": 0.65,
        "colsample_bylevel": 0.65,
        "min_child_weight": 30,      
        "gamma": 2.0,                
        "reg_alpha": 0.5,
        "reg_lambda": 10.0,
        #"scale_pos_weight": "auto"
        }},        

]

# ========= Nastavenie najlepsich hyperparametrov =========

BEST_LR_PARAMS_HORIZONT_5 = {"penalty": "l1",
                            "C": 0.057,
                            "solver": "saga"}
BEST_SVM_PARAMS_HORIZONT_5 = {"C": 0.133,
                            "kernel": "rbf",
                            "gamma": "scale"}
BEST_XGB_PARAMS_HORIZONT_5 ={"n_estimators": 467,
                            "max_depth": 2,
                            "learning_rate": 0.027,
                            "subsample": 0.74,
                            "colsample_bytree": 0.75,
                            "colsample_bylevel": 0.914,
                            "min_child_weight": 15,
                            "gamma": 2.57,
                            "reg_alpha": 0.59,
                            "reg_lambda": 0.93,
                            "scale_pos_weight": "auto"}

# ========= Porovnanie vykonnosti feature selection metod =========
FEATURE_SELECTION_METHODS = ["mi", "rfe", "random_forest", "l1", "custom"]
SWEEP_FEATURE_SELECTION = [{
        "nazov":             f"{model_name}_H5_fs_{fs_method}_k15",
        "model_name":        model_name,
        "feature_selection": fs_method,
        "k_features":        15,
        "version":           "fs_sweep",
        "note":              f"Feature selection sweep",
        "model_parameters":  parametre,
    }
    for model_name, parametre in [
        ("lr",  BEST_LR_PARAMS_HORIZONT_5),
        ("svm",     BEST_SVM_PARAMS_HORIZONT_5),
        ("xgboost", BEST_XGB_PARAMS_HORIZONT_5),
    ]
    for fs_method in FEATURE_SELECTION_METHODS
]

# === Nastavenie optimalneho poctu features v modeli =====
K_VALUES = [5, 10, 15, 20, 25, 30, 40, 60, 100]
SWEEP_EXPERIMENTS = [
    *[{"nazov": f"LR_H5_sweep_k{k}",
       "model_name": "lr",
       "feature_selection": "custom",
       "k_features": k,
       "model_parameters": BEST_LR_PARAMS_HORIZONT_5}
      for k in K_VALUES],

    *[{"nazov": f"SVM_H5_sweep_k{k}",
       "model_name": "svm",
       "feature_selection": "custom",
       "k_features": k,
       "model_parameters": BEST_SVM_PARAMS_HORIZONT_5}
      for k in K_VALUES],

    *[{"nazov": f"XGB_H5_sweep_k{k}",
       "model_name": "xgboost",
       "feature_selection": "custom",
       "k_features": k,
       "model_parameters": BEST_XGB_PARAMS_HORIZONT_5}
      for k in K_VALUES],
]

# ====== Porovnanie modelov na roznych train/test splitoch ======
SPLIT_EXPERIMENT = [
     {"nazov": "LogReg_EXPERIMENT_final_SPLIT60_20_20",
     "model_name": "lr",
     "feature_selection": "custom",
     "k_features": 30,
     "version": "split_train_60",
     "note": "",
     "model_parameters": BEST_LR_PARAMS_HORIZONT_5},

    {"nazov": "SVM_EXPERIMENT_final_SPLIT60_20_20",
     "model_name": "svm",
     "feature_selection": "custom",
     "k_features": 30,
     "version": "split_train_60",
     "note": "",
     "model_parameters": BEST_SVM_PARAMS_HORIZONT_5},

    {"nazov": "XGBoost_EXPERIMENT_final_SPLIT60_20_20",
     "model_name": "xgboost",
     "feature_selection": "custom",
     "k_features": 30,
     "version": "split_train_60",
     "note": "",
     "model_parameters": BEST_XGB_PARAMS_HORIZONT_5},
]

# ====== Finalne modely ======
FINAL_MODELS = [
    {"nazov": "LogReg_FINAL_calibrated_MODEL",
     "model_name": "lr",
     "feature_selection": "custom",
     "k_features": 15,
     "version": "FINAL - v1",
     "note": "",
     "model_parameters": BEST_LR_PARAMS_HORIZONT_5},

    {"nazov": "SVM_FINAL_calibrated_MODEL",
     "model_name": "svm",
     "feature_selection": "custom",
     "k_features": 15,
     "version": "FINAL - v1",
     "note": "",
     "model_parameters": BEST_SVM_PARAMS_HORIZONT_5},

    {"nazov": "XGBoost_FINAL_calibrated_MODEL",
     "model_name": "xgboost",
     "feature_selection": "custom",
     "k_features": 15,
     "version": "FINAL - v1",
     "note": "",
     "model_parameters": BEST_XGB_PARAMS_HORIZONT_5},
]